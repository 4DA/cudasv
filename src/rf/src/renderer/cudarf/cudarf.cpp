#include <memory>
#include <cassert>
#include <cmath>

#include <tinygltf/stb_image.h>

#include <rf/renderer/glm_common.hpp>
#include <rf/renderer/cudarf/cudarf.hpp>
#include <rf/renderer/cudarf/cudarf_camera.hpp>
#include <rf/renderer/virtual_camera.hpp>

#include "cudarf_rast.hpp"
#include "helpers_cudavec.inl"
#include "TAA_common.hpp"
#include "helpers.hpp"

#include "vecglm.inl"
#include "types.hpp"

#include <cuda_gl_interop.h>
#include "glcommon.hpp"

// #define CUDARF_CPU_PROFILING
#ifdef CUDARF_CPU_PROFILING
#include <chrono>
#include <spdlog/spdlog.h>
#endif

using namespace cudarf;
using namespace cudarf::rast;

namespace cudarf
{
namespace pipe
{
struct Ctx;
void init_tile_queue_static(cudarf::pipe::Ctx *desc, const cudaStream_t &cuStream);
}
}


static float halton(int prime, int index = 1 /* index is 1-based */)
{
    float r = 0.0f;
    float f = 1.0f;
    int i = index;

    while (i > 0) {
        f /= prime;
        r += f * (i % prime);
        i = (int) std::floor(i / (float)prime);
    }

    return r;
}

static void initialize_halton_2_3(std::vector<float2> &out)
{
    for (int i = 0, n = out.size(); i != n; i++) {
        float u = halton(2, i + 1) - 0.5f;
        float v = halton(3, i + 1) - 0.5f;
        out[i] = {u, v};
    }
}

cudarf::Uniforms cudarf::make_uniforms(const glm::mat4 &P,
                                       const glm::mat4 &V,
                                       const glm::mat4 &M)
{
    glm::mat4 PVM = P * V * M;
    glm::mat4 N = glm::mat4(glm::transpose(glm::inverse(glm::mat3(M))));
    return cudarf::Uniforms{PVM, M, N};
}

cudarf::CommonUniforms cudarf::make_common(const glm::mat4 &P,
                                           const glm::mat4 &V)
{
    glm::mat4 PV = P * V;
    return cudarf::CommonUniforms{PV, P, V, glm::inverse(V)};
}

cudarf::CommonUniforms cudarf::make_common(const rf::VirtualCamera *camera)
{
    return cudarf::make_common(camera->projection_matrix,
                               get_camera_matrix(camera->transform));
}

unsigned int cudarf::pipe::alloc_draw_packet(cudarf::pipe::Ctx *desc)
{
    return desc->drawPacketCount++;
}

void cudarf::pipe::set_draw_packet_buffers(cudarf::pipe::Ctx *desc,
                                           const int *bufIdx,
                                           int index_count,
                                           const Vec3f *bufPos,
                                           int vertCount,
                                           const Vec4f *bufCol,
                                           const Vec3f *bufNor,
                                           const Vec2f *bufTex,
                                           const Vec4f *bufTan,
                                           unsigned int drawPacketId,
                                           const cudaStream_t &cuStream)
{
#ifdef CUDARF_CPU_PROFILING
    auto t1 = std::chrono::high_resolution_clock::now();
#endif

    assert(drawPacketId < CUDARF_MAX_DRAW_PACKETS);

    if (!index_count) {
        SPDLOG_DEBUG("Zero index count for drawPacketId: {}", drawPacketId);
    }

    DrawPacket *drawPacket = &desc->drawPackets[drawPacketId];

    drawPacket->index_count = index_count;
    drawPacket->vertCount = vertCount;

    if (bufIdx == nullptr) {
        desc->drawPacketIdxBuffers[drawPacketId].reset();
        drawPacket->dev_bufIdx = nullptr;
        drawPacket->indexCapacity = 0;
    } else {
        if (index_count > drawPacket->indexCapacity) {
            desc->drawPacketIdxBuffers[drawPacketId].reset(drawPacket->index_count);
            drawPacket->dev_bufIdx = desc->drawPacketIdxBuffers[drawPacketId].get();
            drawPacket->indexCapacity = index_count;
        }

        CUDA_CHK(cudaMemcpyAsync(drawPacket->dev_bufIdx, bufIdx,
                                 drawPacket->index_count * sizeof(PrimitiveIndex),
                                 cudaMemcpyHostToDevice,
                                 cuStream));
    }

    SPDLOG_DEBUG("set_draw_packet_buffers [drawPacketId:{}]: indices: {}, sz: {}",
                 drawPacketId, (size_t) drawPacket->index_count,
                 drawPacket->index_count * sizeof(PrimitiveIndex), sizeof(PrimitiveIndex));

    if (vertCount > drawPacket->vertexCapacity) {
        desc->drawPacketVertexBuffers[drawPacketId].reset(drawPacket->vertCount);
        drawPacket->dev_bufVertex = desc->drawPacketVertexBuffers[drawPacketId].get();

        desc->drawPacketVertexStagingBuffers[drawPacketId].reset(drawPacket->vertCount);
        drawPacket->stagingBufVertex =
            desc->drawPacketVertexStagingBuffers[drawPacketId].get();

        drawPacket->vertexCapacity = vertCount;
    }

    VertexIn *bufVertex = drawPacket->stagingBufVertex;

    for (int i = 0; i < drawPacket->vertCount; i++) {
        bufVertex[i].pos = bufPos[i];

        if (bufCol) {
            bufVertex[i].col = make_float4(bufCol[i].x, bufCol[i].y, bufCol[i].z, bufCol[i].w);
        }
        else {
            bufVertex[i].col = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        }

        if (bufNor) {
            bufVertex[i].nor = bufNor[i];
        }

        if (bufTex) {
            bufVertex[i].tex = bufTex[i];
        }

        if (bufTan) {
            bufVertex[i].tan = bufTan[i];
        }
    }

    // DEBUG: print vertices and indices
    // for (int i = 0; i < drawPacket->vertCount; i++) {
    //     printf("vertex_in[%d] pos = %f, %f, %f | col = %f, %f, %f\n", i,
    //            bufVertex[i].pos.x, bufVertex[i].pos.y, bufVertex[i].pos.z,
    //            bufVertex[i].col.x, bufVertex[i].col.y, bufVertex[i].col.z);
    // }

    // for (int i = 0; i < drawPacket->index_count/3; i++) {
    //     printf("triangle_in[%d] (%d, %d, %d) | col = %f, %f, %f\n", i,
    //            bufIdx[3*i], bufIdx[3*i+1], bufIdx[3*i+2]);
    // }


    CUDA_CHK(cudaMemcpyAsync(drawPacket->dev_bufVertex, drawPacket->stagingBufVertex,
                             drawPacket->vertCount * sizeof(VertexIn), cudaMemcpyHostToDevice, cuStream));

    SPDLOG_DEBUG("Device vertex buffer [sz: {} = {} * {}] set", drawPacket->vertCount * sizeof(VertexIn), (size_t) drawPacket->vertCount, sizeof(VertexIn));

#ifdef CUDARF_CPU_PROFILING
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    SPDLOG_DEBUG("{}[#idx: {}] time: {:f} ms\n", __func__, index_count, ms_double);
#endif
}

void cudarf::pipe::set_draw_packet_buffers(cudarf::pipe::Ctx *desc,
                                           const rf::GltfMesh &mesh,
                                           unsigned int drawPacketId,
                                           const cudaStream_t &cuStream)
{
    int index_count = mesh.count;
    int vertCount = mesh.vertices.count;
    assert(drawPacketId < CUDARF_MAX_DRAW_PACKETS);

    if (!index_count) {
        SPDLOG_DEBUG("Zero index count for drawPacketId: {}", drawPacketId);
    }

    DrawPacket *drawPacket = &desc->drawPackets[drawPacketId];

    drawPacket->index_count = index_count;
    drawPacket->vertCount = vertCount;

    if (index_count > drawPacket->indexCapacity) {
        desc->drawPacketIdxBuffers[drawPacketId].reset(drawPacket->index_count);
        drawPacket->dev_bufIdx = desc->drawPacketIdxBuffers[drawPacketId].get();
        drawPacket->indexCapacity = index_count;
    }

    if (mesh.convertIdxTo32) {
        uint32_t *tmp = new uint32_t[drawPacket->index_count];

        for (int i = 0; i < drawPacket->index_count; i++) {
            tmp[i] = ((uint16_t *)mesh.indexPtr)[i];
        }

        CUDA_CHK(cudaMemcpyAsync(drawPacket->dev_bufIdx, tmp,
                                 drawPacket->index_count * sizeof(PrimitiveIndex),
                                 cudaMemcpyHostToDevice,
                                 cuStream));

        delete[] tmp;
    }
    else {
        CUDA_CHK(cudaMemcpyAsync(drawPacket->dev_bufIdx, mesh.indexPtr,
                                 drawPacket->index_count * sizeof(PrimitiveIndex),
                                 cudaMemcpyHostToDevice,
                                 cuStream));
    }

    SPDLOG_DEBUG("set_buffers [drawPacketId:{}]: indices: {}, sz: {}", drawPacketId, (size_t) drawPacket->index_count, drawPacket->index_count * sizeof(PrimitiveIndex), sizeof(PrimitiveIndex));

    if (vertCount > drawPacket->vertexCapacity) {
        desc->drawPacketVertexBuffers[drawPacketId].reset(drawPacket->vertCount);
        drawPacket->dev_bufVertex = desc->drawPacketVertexBuffers[drawPacketId].get();

        desc->drawPacketVertexStagingBuffers[drawPacketId].reset(drawPacket->vertCount);
        drawPacket->stagingBufVertex =
            desc->drawPacketVertexStagingBuffers[drawPacketId].get();

        drawPacket->vertexCapacity = vertCount;
    }

    VertexIn *bufVertex = drawPacket->stagingBufVertex;

    for (int i = 0; i < drawPacket->vertCount; i++) {
        bufVertex[i].pos = to_vec3f(get_attribute3(&mesh.vertices, i));
        bufVertex[i].col = make_float4(1.0f, 1.0f, 1.0f, 1.0f);

        if (mesh.normals.count) {
            bufVertex[i].nor = to_vec3f(get_attribute3(&mesh.normals, i));
        }

        if (mesh.texcoords.count) {
            bufVertex[i].tex = to_vec2f(get_attribute2(&mesh.texcoords, i));
        }

        if (mesh.tangents.count) {
            bufVertex[i].tan = to_vec4f(get_attribute4(&mesh.tangents, i));
        }
    }

    CUDA_CHK(cudaMemcpyAsync(drawPacket->dev_bufVertex, drawPacket->stagingBufVertex,
                             drawPacket->vertCount * sizeof(VertexIn), cudaMemcpyHostToDevice, cuStream));

    SPDLOG_DEBUG("Device vertex buffer [sz: {} = {} * {}] set", drawPacket->vertCount * sizeof(VertexIn), (size_t) drawPacket->vertCount, sizeof(VertexIn));
}

cudarf::pipe::Ctx::Ctx(int window_width,
                       int window_height,
                       int tileQLimit,
                       int SMPCount,
                       bool TAAEnabled,
                       int clockRate,
                       const cudaStream_t &cuStream)
    : dev_pipeStatic(1),
      dev_pipeFrame(1),
      dev_pipeSubmission(1),
      dev_pipeParams(1),
      dev_pipeAtomics(1)
#ifdef WITH_TAA
    , framebufferResources{
          cudarf::memory::ArraySurfaceTexture(
              window_width, window_height, cudarf::memory::rgba8_channel_desc(),
              cudaAddressModeClamp, cudaReadModeNormalizedFloat,
              cudarf::memory::TextureSampling::NormalizedLinear),
          cudarf::memory::ArraySurfaceTexture(
              window_width, window_height, cudarf::memory::rgba8_channel_desc(),
              cudaAddressModeClamp, cudaReadModeNormalizedFloat,
              cudarf::memory::TextureSampling::NormalizedLinear)}
    , rasterResource(
          window_width, window_height, cudarf::memory::rgba8_channel_desc(),
          cudaAddressModeClamp, cudaReadModeNormalizedFloat,
          cudarf::memory::TextureSampling::NormalizedLinear)
    , uiFramebufferResource(
          window_width, window_height, cudarf::memory::rgba8_channel_desc())
#endif
{
    width = window_width;
    height = window_height;
    this->SMPCount = SMPCount;
    this->clockRate = clockRate;
    this->TAAEnabled = TAAEnabled;

    unsigned long rasterizerW = round_up_to_mult_pwr(width, CUDARF_BIN_LOG2 + CUDARF_TILE_LOG2);
    unsigned long rasterizerH = round_up_to_mult_pwr(height, CUDARF_BIN_LOG2 + CUDARF_TILE_LOG2);

    cudarf::rast::PipeStaticContext pipeStatic{};
    pipeStatic.windowWidth = width;
    pipeStatic.windowHeight = height;
    pipeStatic.rasterizerSize = make_int2(rasterizerW, rasterizerH);
    pipeStatic.clockRate = clockRate;
    CUDA_CHK(cudaMemcpyAsync(dev_pipeStatic.get(),
                             &pipeStatic,
                             sizeof(cudarf::rast::PipeStaticContext),
                             cudaMemcpyHostToDevice,
                             cuStream));

    SPDLOG_INFO("\nInitializing rasterization descriptor [TAA: {}] ...", TAAEnabled);
    SPDLOG_INFO("----------------------------------------");
    SPDLOG_INFO("Rasterizer virtual viewport: {} x {} (mult of 2^{}), {} x {} tiles | bin size: {} x {} ({} x {} tiles) | tile sz: {} x {}", rasterizerW, rasterizerH, CUDARF_BIN_LOG2 + CUDARF_TILE_LOG2, rasterizerW / CUDARF_TILE_SZ, rasterizerH / CUDARF_TILE_SZ, rasterizerW / CUDARF_BIN_COUNT, rasterizerH / CUDARF_BIN_COUNT, rasterizerW / CUDARF_BIN_COUNT / CUDARF_TILE_SZ, rasterizerH / CUDARF_BIN_COUNT / CUDARF_TILE_SZ, CUDARF_TILE_SZ, CUDARF_TILE_SZ);

    int tilesInBin = rasterizerW / CUDARF_BIN_COUNT / CUDARF_TILE_SZ * rasterizerH / CUDARF_BIN_COUNT / CUDARF_TILE_SZ;

    // this will cause artifacts in coarse tiler
    if (tilesInBin > CUDARF_MAX_TILES) {
        SPDLOG_ERROR("Tiles in bin({}) > CUDARF_MAX_TILES({})", tilesInBin, CUDARF_MAX_TILES);
        assert(false);
    }


    // Vertex transform
    // ---------------------------------
    dev_uniforms.reset(CUDARF_DRAW_PACKET_BATCH_LIMIT);

#ifdef WITH_TAA
    dev_uniformsHist.reset(CUDARF_DRAW_PACKET_BATCH_LIMIT);
#endif

    // Bin tiler
    // ---------------------------------
    dev_binFirstSeg.reset(CUDARF_MAXBINS_SQR * CUDARF_BIN_STREAMS_SIZE);

    dev_binTotal.reset(CUDARF_MAXBINS_SQR * CUDARF_BIN_STREAMS_SIZE);

    SPDLOG_INFO("Bin tiler (only fixed size): {} KB", (CUDARF_MAXBINS_SQR * CUDARF_BIN_STREAMS_SIZE * sizeof(int32_t) +
           CUDARF_MAXBINS_SQR * CUDARF_BIN_STREAMS_SIZE * sizeof(int32_t)) / 1024);

    // CUDA_CHK(cudarf_cuda_free(dev_binQueues));
    // CUDA_CHK(cudarf_cuda_malloc(&dev_binQueues, CUDARF_MAXBINS_SQR * sizeof(SimpleQueue::Segment<NQ_BINSEG_SIZE>)));

    // Coarse tiler
    // ---------------------------------
    unsigned int tileCount = rasterizerW / CUDARF_TILE_SZ * rasterizerH / CUDARF_TILE_SZ;

    this->tileQLimit = tileQLimit;

    dev_tileQData.reset(tileCount * tileQLimit);

    dev_tileQHeaders.reset(tileCount);

    SPDLOG_INFO("Coarse tiler: {} KB", tileCount * (tileQLimit * sizeof(int32_t) + sizeof(SimpleQueue::Segment)) /  1024);

    init_tile_queue_static(this, cuStream);

    // Output framebuffer & depth
    // ---------------------------------
    // TODO: write method to initialize depth and set it here
    dev_depthbuffer.reset(width * height);
    dev_geom_output.reset(width * height);
    dev_xyCommands.reset(width * height);

    SPDLOG_INFO("Depth buffer: {} KB", width * height * sizeof(DepthValue) / 1024);

#if defined(WITH_TAA)

    for (std::size_t i = 0; i < framebufferResources.size(); ++i) {
        dev_framebuffer[i] = framebufferResources[i].surface();
        dev_framebufferTex[i] = framebufferResources[i].texture();
    }

    dev_velocityTex.reset(width * height);

    SPDLOG_INFO("Internal framebuffer {} x {} @ 32: {} KB", width, height, (width * height * sizeof(ColorN)) / 1024);

    SPDLOG_INFO("Output framebuffer[0] {} x {} @ 32: {} KB", width, height, (width * height * sizeof(ColorN)) / 1024);

    SPDLOG_INFO("Output framebuffer[1] {} x {} @ 32: {} KB", width, height, (width * height * sizeof(ColorN)) / 1024);

    SPDLOG_INFO("UI framebuffer {} x {} @ 32: {} KB", width, height, (width * height * sizeof(ColorN)) / 1024);

    SPDLOG_INFO("Velocity texture {} x {}: {} KB", width, height, (width * height * sizeof(cudarf::Velocity)) / 1024);
#else

    free_surface(dev_framebuffer);
    create_surface(dev_framebuffer, dev_framebufferTex, width, height, cuStream);
    free_surface(uiFramebuffer);
    create_surface(uiFramebuffer, width, height, cuStream);

    SPDLOG_INFO("Output framebuffer {} x {} @ 32: {} KB", width, height, (width * height * sizeof(ColorN)) / 1024);

    SPDLOG_INFO("UI framebuffer {} x {} @ 32: {} KB", width, height, (width * height * sizeof(ColorN)) / 1024);
#endif

#ifdef WITH_TAA
    TAA.pointsHalton.resize(HALTON_POINTS);
    initialize_halton_2_3(TAA.pointsHalton);

    std::string points;
    points.reserve(20);

    for (float2 pt: TAA.pointsHalton) {
        points += ("(" + std::to_string(pt.x) + ", " + std::to_string(pt.y) + ") ");
    }
    SPDLOG_INFO("Halton points: {}", points.c_str());
#endif
}

cudarf::pipe::Ctx::~Ctx()
{
    // The user is responsible for cleaning draw packets.
#ifndef WITH_TAA
    cudarf::pipe::Ctx *desc = this;
    free_surface(desc->dev_framebuffer);
    free_surface(desc->uiFramebuffer);
#endif
}

void cudarf::pipe::draw_triangles(cudarf::pipe::Ctx* rasterization_desc,
                                  const std::vector<glm::vec3> &triangles,
                                  const std::vector<glm::vec4> &colors,
                                  const glm::mat4 &M,
                                  const cudarf::CommonUniforms &common,
                                  bool face_culling,
                                  bool blending,
                                  unsigned int frameCounter,
                                  const cudaStream_t &cuStream)
{
    auto uniforms = cudarf::make_uniforms(common.P, common.V, M);
    std::vector<int> idx;
    idx.resize(triangles.size());

    for (unsigned int i = 0; i < triangles.size(); i++) {
        idx[i] = i;
    }

    #warning TODO reinterpret_cast
    cudarf::pipe::set_draw_packet_buffers(rasterization_desc,
                                          idx.data(),
                                          idx.size(),
                                          reinterpret_cast<const cudarf::Vec3f *>(triangles.data()),
                                          triangles.size(),
                                          reinterpret_cast<const cudarf::Color *>(colors.data()),
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          CUDARF_SCRATCH_DRAW_PACKET,
                                          cuStream);

    auto  material = std::make_shared<cudarf::Material>();

    material->baseColor = make_float4(1.0, 1.0, 1.0, 1.0);
    material->emissive = make_float3(0.0, 0.0, 0.0);
    material->metallic = 0.0f;
    material->roughness = 1.0f;
    material->type = cudarf::SHADER_TYPE_UNLIT;

    cudarf::MaterialMap materialMap = {{0, material}};
    std::vector<unsigned int> materialIds = {0};
    std::vector<unsigned int> drawPacketIds = {CUDARF_SCRATCH_DRAW_PACKET};

    std::vector<cudarf::CUDARFLight> lights;

    cudarf::pipe::LaunchConfig launchConfig(
        false,
        false,
        frameCounter,
        cudarf::pipe::get_internal_fb(rasterization_desc, frameCounter),
        nullptr);

    cudarf::pipe::run_pipe(rasterization_desc,
                   cudarf::RenderParams {
                       face_culling,
                       blending,
                   },
                   uniforms,
#ifdef WITH_TAA
                   uniforms,
#endif
                   drawPacketIds,
                   materialIds,
                   materialMap,
                   launchConfig,
                   cuStream
     );

}

void cudarf::pipe::begin_frame(cudarf::pipe::Ctx *desc,
                               const rf::VirtualCamera &camera,
                               const cudarf::PBRParams &pbr,
                               const CommonUniforms &commonHist,
                               unsigned int frameCounter,
                               cudaStream_t cuStream)
{
    PipeFrameContext pipeFrame;

    CommonUniforms common = cudarf::make_common(&camera);

#ifdef WITH_TAA
    if (desc->TAAEnabled) {
        pipeFrame.common = prepare_for_TAA(
            *desc,
            frameCounter,
            4,
            common);
    }
    else {
        pipeFrame.common = common;
    }

    pipeFrame.taa.commonHist = (frameCounter > 0) ? commonHist : common;
    pipeFrame.taa.uniformsHist = desc->dev_uniformsHist.get();
    pipeFrame.taa.velocityTex = desc->dev_velocityTex.get();
    pipeFrame.taa.velocityThreshold = desc->TAA.velocityThreshold;
#else
    pipeFrame.common = common;
#endif

    // set PBR context
    {
        pipeFrame.camera = make_float3(camera.transform.translation.x,
                                       camera.transform.translation.y,
                                       camera.transform.translation.z);

        pipeFrame.exposure = pbr.exposure;
        pipeFrame.lightCount = pbr.lights.size();

        for (int i = 0; i < pipeFrame.lightCount; i++) {
            pipeFrame.lights[i] = pbr.lights[i];
        }

        pipeFrame.sphericalHarmonics = pbr.sphericalHarmonics;
        pipeFrame.brdfLUT = pbr.brdfLUT;
        pipeFrame.specular = pbr.specular;
    }

    CUDA_CHK(cudaMemcpyAsync(desc->dev_pipeFrame.get(), &pipeFrame, sizeof(cudarf::rast::PipeFrameContext), cudaMemcpyHostToDevice, cuStream));

    cudarf::pipe::clear_depth(desc, cuStream);

    CUDA_CHK(cudaMemsetAsync(desc->dev_geom_output.get(),
                             0xFF,
                             sizeof(cudarf::visibuf::GeomOutput) * desc->width * desc->height,
                             cuStream));

    CUDA_CHK(cudaMemsetAsync(desc->dev_xyCommands.get(),
                             0xFF,
                             sizeof(cudarf::visibuf::XYCommand) * desc->width * desc->height,
                             cuStream));

#ifdef WITH_TAA
    cudarf::pipe::clear_framebuffer(desc, desc->rasterResource.surface(), make_uchar4(0, 0, 0, 0), cuStream);
    CUDA_CHK(cudaMemsetAsync(desc->dev_velocityTex.get(), 0, sizeof(cudarf::Velocity) * desc->width * desc->height, cuStream));
#else
    cudarf::pipe::clear_framebuffer(desc,
                                    cudarf::pipe::get_output_fb(desc, frameCounter),
                                    make_uchar4(0, 0, 0, 0),
                                    cuStream);
#endif
    cudarf::pipe::clear_framebuffer(desc,
                                    cudarf::pipe::get_ui_fb(desc),
                                    make_uchar4(0, 0, 0, 0),
                                    cuStream);

}

cudarf::Framebuffer cudarf::pipe::get_output_fb(cudarf::pipe::Ctx *desc, unsigned int frameCounter)
{
#ifdef WITH_TAA
    if (desc->TAAEnabled) {
        return desc->dev_framebuffer[frameCounter % 2];
    } else {
        return desc->rasterResource.surface();
    }
#else
    return desc->dev_framebuffer;
#endif
}

cudarf::FBTexture cudarf::pipe::get_output_tex(cudarf::pipe::Ctx *desc, unsigned int frameCounter)
{
#ifdef WITH_TAA
    if (desc->TAAEnabled) {
        if (frameCounter % 2 == 0) {
            return desc->dev_framebufferTex[0];
        }
        else {
            return desc->dev_framebufferTex[1];
        }
    } else {
        return desc->rasterResource.texture();
    }
#else
    return desc->dev_framebufferTex;
#endif
}

#ifdef WITH_TAA

cudarf::Framebuffer cudarf::pipe::get_internal_fb(cudarf::pipe::Ctx *desc, unsigned int frameCounter)
{
    return desc->rasterResource.surface();
}

cudarf::Framebuffer cudarf::pipe::get_ui_fb(cudarf::pipe::Ctx *desc)
{
    return desc->uiFramebufferResource.surface();
}

cudarf::Framebuffer cudarf::pipe::get_history_fb(cudarf::pipe::Ctx *desc, unsigned int frameCounter)
{
    if (frameCounter % 2 == 0) {
        return desc->dev_framebuffer[1];
    }
    else {
        return desc->dev_framebuffer[0];
    }
}

cudarf::FBTexture cudarf::pipe::get_history_tex(cudarf::pipe::Ctx *desc, unsigned int frameCounter)
{
    if (frameCounter % 2 == 0) {
        return desc->dev_framebufferTex[1];
    }
    else {
        return desc->dev_framebufferTex[0];
    }
}

#else
cudarf::Framebuffer cudarf::pipe::get_internal_fb(cudarf::pipe::Ctx *desc, unsigned int frameCounter)
{
    return desc->dev_framebuffer;
}

cudarf::Framebuffer cudarf::pipe::get_ui_fb(cudarf::pipe::Ctx *desc)
{
    return desc->uiFramebuffer;
}

#endif
