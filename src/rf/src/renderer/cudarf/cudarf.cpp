#include <memory>
#include <cassert>
#include <cmath>

#include <tinygltf/stb_image.h>

#include <rf/renderer/glm_common.hpp>
#include <rf/renderer/cudarf/cudarf.hpp>
#include <rf/renderer/cudarf/cudarf_camera.hpp>
#include <rf/renderer/virtual_camera.hpp>
#include "helpers.hpp"

#include "vecglm.inl"
#include "types.hpp"

#include <cuda_gl_interop.h>
#include "glcommon.hpp"

// #define CUDARF_CPU_PROFILING
#ifdef CUDARF_CPU_PROFILING
#include <chrono>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/printf.h>
#endif

using namespace cudarf;
using namespace cudarf::rast;


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
    return cudarf::Uniforms{PVM, M};
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
                                           unsigned int drawPacketId,
                                           const cudaStream_t &cuStream)
{
#ifdef CUDARF_CPU_PROFILING
    auto t1 = std::chrono::high_resolution_clock::now();
#endif

    assert(drawPacketId < CUDARF_MAX_DRAW_PACKETS);

    if (!index_count) {
        SPDLOG_DEBUG("{}", fmt::sprintf("Zero index count for drawPacketId: %u", drawPacketId));
    }

    DrawPacket *drawPacket = &desc->drawPackets[drawPacketId];

    drawPacket->index_count = index_count;
    drawPacket->vertCount = vertCount;

    if (bufIdx == nullptr) {
        CUDA_CHK(cudarf_cuda_free(drawPacket->dev_bufIdx));
        drawPacket->dev_bufIdx = nullptr;
        drawPacket->indexCapacity = 0;
    } else {
        if (index_count > drawPacket->indexCapacity) {
            CUDA_CHK(cudarf_cuda_free(drawPacket->dev_bufIdx));
            CUDA_CHK(cudarf_cuda_malloc(&drawPacket->dev_bufIdx, drawPacket->index_count * sizeof(PrimitiveIndex)));
            drawPacket->indexCapacity = index_count;
        }

        CUDA_CHK(cudaMemcpyAsync(drawPacket->dev_bufIdx, bufIdx,
                                 drawPacket->index_count * sizeof(PrimitiveIndex),
                                 cudaMemcpyHostToDevice,
                                 cuStream));
    }

    SPDLOG_DEBUG("{}",
                 fmt::sprintf("set_draw_packet_buffers [drawPacketId:%d]: indices: %zu, sz: %zu",
                 drawPacketId,
                 (size_t) drawPacket->index_count,
                 drawPacket->index_count * sizeof(PrimitiveIndex),
                 sizeof(PrimitiveIndex)));

    if (vertCount > drawPacket->vertexCapacity) {
        CUDA_CHK(cudarf_cuda_free(drawPacket->dev_bufVertex));
        CUDA_CHK(cudarf_cuda_malloc(&drawPacket->dev_bufVertex, drawPacket->vertCount * sizeof(VertexIn)));

        CUDA_CHK(cudarf_cuda_free_host(drawPacket->stagingBufVertex));
        CUDA_CHK(cudarf_cuda_malloc_host((void **)&drawPacket->stagingBufVertex, drawPacket->vertCount * sizeof(VertexIn)));

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

    SPDLOG_DEBUG("{}", fmt::sprintf("Device vertex buffer [sz: %zu = %zu * %zu] set",
                drawPacket->vertCount * sizeof(VertexIn),
                (size_t) drawPacket->vertCount, sizeof(VertexIn)));

#ifdef CUDARF_CPU_PROFILING
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    SPDLOG_DEBUG("{}", fmt::sprintf("%s[#idx: %d] time: %lf ms\n", __func__, index_count, ms_double));
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
        SPDLOG_DEBUG("{}", fmt::sprintf("Zero index count for drawPacketId: %u", drawPacketId));
    }

    DrawPacket *drawPacket = &desc->drawPackets[drawPacketId];

    drawPacket->index_count = index_count;
    drawPacket->vertCount = vertCount;

    if (index_count > drawPacket->indexCapacity) {
        CUDA_CHK(cudaFree(drawPacket->dev_bufIdx));
        CUDA_CHK(cudaMalloc(&drawPacket->dev_bufIdx, drawPacket->index_count * sizeof(PrimitiveIndex)));
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

    SPDLOG_DEBUG("{}", fmt::sprintf("set_buffers [drawPacketId:%d]: indices: %zu, sz: %zu",
                 drawPacketId,
                 (size_t) drawPacket->index_count,
                 drawPacket->index_count * sizeof(PrimitiveIndex),
                 sizeof(PrimitiveIndex)));

    if (vertCount > drawPacket->vertexCapacity) {
        CUDA_CHK(cudarf_cuda_free(drawPacket->dev_bufVertex));
        CUDA_CHK(cudarf_cuda_malloc(&drawPacket->dev_bufVertex, drawPacket->vertCount * sizeof(VertexIn)));

        CUDA_CHK(cudarf_cuda_free_host(drawPacket->stagingBufVertex));
        CUDA_CHK(cudarf_cuda_malloc_host((void **)&drawPacket->stagingBufVertex, drawPacket->vertCount * sizeof(VertexIn)));

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
    }

    CUDA_CHK(cudaMemcpyAsync(drawPacket->dev_bufVertex, drawPacket->stagingBufVertex,
                             drawPacket->vertCount * sizeof(VertexIn), cudaMemcpyHostToDevice, cuStream));

    SPDLOG_DEBUG("{}", fmt::sprintf("Device vertex buffer [sz: %zu = %zu * %zu] set",
                drawPacket->vertCount * sizeof(VertexIn),
                (size_t) drawPacket->vertCount, sizeof(VertexIn)));
}

void cudarf::create_surface(cudaSurfaceObject_t &fb,
                            int width,
                            int height,
                            const cudaStream_t &cuStream)
{
    cudaArray *cuArray;
    const int channels = 4;
    const int compSz = 1;

    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);

    CUDA_CHK(cudaMallocArray(&cuArray, &channelDesc, width, height, 0));

    cudaResourceDesc surfRes;
    std::memset(&surfRes, 0, sizeof(cudaResourceDesc));
    surfRes.resType = cudaResourceTypeArray;
    surfRes.res.array.array = cuArray;

    CUDA_CHK(cudaCreateSurfaceObject(&fb, &surfRes));
}


void cudarf::create_surface(cudaSurfaceObject_t &outSurface,
                            cudaTextureObject_t &outTexture,
                            int width,
                            int height,
                            const cudaStream_t &cuStream)
{
    cudaArray *cuArray;
    const int channels = 4;
    const int compSz = 1;

    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);

    CUDA_CHK(cudaMallocArray(&cuArray, &channelDesc, width, height, 0));

    cudaResourceDesc surfRes;
    std::memset(&surfRes, 0, sizeof(cudaResourceDesc));
    surfRes.resType = cudaResourceTypeArray;
    surfRes.res.array.array = cuArray;

    CUDA_CHK(cudaCreateSurfaceObject(&outSurface, &surfRes));

    // create texture for reading
    // --
    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = cuArray;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = true;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.addressMode[1] =  cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeNormalizedFloat;

    CUDA_CHK(cudaCreateTextureObject(&outTexture, &texRes, &texDescr, NULL));

    return;
}

void cudarf::create_surface(cudarf::LinearSurface &outSurface,
                            int width,
                            int height,
                            const cudaStream_t &cuStream)
{
    CUDA_CHK(cudaMalloc(&outSurface.devPtr, width * height * sizeof(ColorN)));
    outSurface.w = width;
    outSurface.h = height;
}

void cudarf::create_surface(cudarf::LinearSurface &outSurface,
                            cudaTextureObject_t &outTexture,
                            int width,
                            int height,
                            const cudaStream_t &cuStream)
{
    CUDA_CHK(cudaMalloc(&outSurface.devPtr, width * height * sizeof(ColorN)));
    outSurface.w = width;
    outSurface.h = height;

    const int channels = 4;
    const int compSz = 1;

    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);

    CUDA_CHK(cudaMallocArray(&outSurface.texArray, &channelDesc, width, height, 0));

    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = outSurface.texArray;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = true;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.addressMode[1] =  cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeNormalizedFloat;

    CUDA_CHK(cudaCreateTextureObject(&outTexture, &texRes, &texDescr, NULL));

    return;
}

void cudarf::free_surface(cudarf::LinearSurface &fb)
{
    CUDA_CHK(cudaFree(fb.devPtr));
    fb.devPtr = nullptr;
}

void cudarf::free_surface(cudaSurfaceObject_t &fb)
{
    // TODO
    fb = 0;
}

void cudarf::pipe::init(cudarf::pipe::Ctx *desc, int window_width, int window_height,
                        int tileQLimit, int SMPCount, bool TAAEnabled, int clockRate, const cudaStream_t &cuStream)
{
    desc->width = window_width;
    desc->height = window_height;
    desc->SMPCount = SMPCount;
    desc->clockRate = clockRate;
    desc->TAAEnabled = TAAEnabled;

    CUDA_CHK(cudaMalloc((void **)&desc->dev_pipeParams, sizeof(cudarf::rast::PipeParams)));
    CUDA_CHK(cudarf_cuda_malloc(&desc->dev_pipeAtomics, sizeof(cudarf::pipe::Atomics)));

    unsigned long rasterizerW = round_up_to_mult_pwr(desc->width, CUDARF_BIN_LOG2 + CUDARF_TILE_LOG2);
    unsigned long rasterizerH = round_up_to_mult_pwr(desc->height, CUDARF_BIN_LOG2 + CUDARF_TILE_LOG2);

    SPDLOG_INFO("{}", fmt::sprintf("\nInitializing rasterization descriptor [TAA: %d] ...", TAAEnabled));
    SPDLOG_INFO("{}", fmt::sprintf("----------------------------------------"));
    SPDLOG_INFO("{}", fmt::sprintf("Rasterizer virtual viewport: %lu x %lu (mult of 2^%d), %lu x %lu tiles | bin size: %lu x %lu (%lu x %lu tiles) | tile sz: %d x %d",
                rasterizerW, rasterizerH, CUDARF_BIN_LOG2 + CUDARF_TILE_LOG2,
                rasterizerW / CUDARF_TILE_SZ, rasterizerH / CUDARF_TILE_SZ,
                rasterizerW / CUDARF_BIN_COUNT, rasterizerH / CUDARF_BIN_COUNT,
                rasterizerW / CUDARF_BIN_COUNT / CUDARF_TILE_SZ,
                rasterizerH / CUDARF_BIN_COUNT / CUDARF_TILE_SZ,
                CUDARF_TILE_SZ, CUDARF_TILE_SZ
        ));

    int tilesInBin = rasterizerW / CUDARF_BIN_COUNT / CUDARF_TILE_SZ * rasterizerH / CUDARF_BIN_COUNT / CUDARF_TILE_SZ;

    // this will cause artifacts in coarse tiler
    if (tilesInBin > CUDARF_MAX_TILES) {
        SPDLOG_ERROR("{}", fmt::sprintf("Tiles in bin(%d) > CUDARF_MAX_TILES(%d)", tilesInBin, CUDARF_MAX_TILES));
        assert(false);
    }


    // Vertex transform
    // ---------------------------------
    CUDA_CHK(cudarf_cuda_free(desc->dev_uniforms));
    CUDA_CHK(cudarf_cuda_malloc(&desc->dev_uniforms, CUDARF_DRAW_PACKET_BATCH_LIMIT * sizeof(cudarf::Uniforms)));

#ifdef WITH_TAA
    CUDA_CHK(cudarf_cuda_free(desc->dev_uniformsHist));
    CUDA_CHK(cudarf_cuda_malloc(&desc->dev_uniformsHist, CUDARF_DRAW_PACKET_BATCH_LIMIT * sizeof(cudarf::Uniforms)));
#endif

    // Bin tiler
    // ---------------------------------
    CUDA_CHK(cudarf_cuda_free(desc->dev_binFirstSeg));
    CUDA_CHK(cudarf_cuda_malloc(&desc->dev_binFirstSeg, CUDARF_MAXBINS_SQR * CUDARF_BIN_STREAMS_SIZE * sizeof(int32_t)));

    CUDA_CHK(cudarf_cuda_free(desc->dev_binTotal));
    CUDA_CHK(cudarf_cuda_malloc(&desc->dev_binTotal, CUDARF_MAXBINS_SQR * CUDARF_BIN_STREAMS_SIZE * sizeof(int32_t)));

    SPDLOG_INFO("{}", fmt::sprintf("Bin tiler (only fixed size): %lu KB", (CUDARF_MAXBINS_SQR * CUDARF_BIN_STREAMS_SIZE * sizeof(int32_t) +
           CUDARF_MAXBINS_SQR * CUDARF_BIN_STREAMS_SIZE * sizeof(int32_t)) / 1024));

    // CUDA_CHK(cudarf_cuda_free(desc->dev_binQueues));
    // CUDA_CHK(cudarf_cuda_malloc(&desc->dev_binQueues, CUDARF_MAXBINS_SQR * sizeof(SimpleQueue::Segment<NQ_BINSEG_SIZE>)));

    // Coarse tiler
    // ---------------------------------
    unsigned int tileCount = rasterizerW / CUDARF_TILE_SZ * rasterizerH / CUDARF_TILE_SZ;

    desc->tileQLimit = tileQLimit;

    CUDA_CHK(cudarf_cuda_free(desc->dev_tileQData));
    CUDA_CHK(cudarf_cuda_malloc(&desc->dev_tileQData, tileCount * sizeof(int32_t) * tileQLimit));

    CUDA_CHK(cudarf_cuda_free(desc->dev_tileQHeaders));
    CUDA_CHK(cudarf_cuda_malloc(&desc->dev_tileQHeaders, tileCount * sizeof(SimpleQueue::Segment)));

    SPDLOG_INFO("{}", fmt::sprintf("Coarse tiler: %lu KB", tileCount * (tileQLimit * sizeof(int32_t) + sizeof(SimpleQueue::Segment)) /  1024));


    // Output framebuffer & depth
    // ---------------------------------
    CUDA_CHK(cudarf_cuda_free(desc->dev_depthbuffer));
    // TODO: write method to initialize depth and set it here
    CUDA_CHK(cudarf_cuda_malloc(&desc->dev_depthbuffer,   desc->width * desc->height * sizeof(DepthValue)));
    CUDA_CHK(cudarf_cuda_free(desc->dev_geom_output));
    CUDA_CHK(cudarf_cuda_malloc(&desc->dev_geom_output,
                                desc->width * desc->height * sizeof(cudarf::visibuf::GeomOutput)));
    CUDA_CHK(cudarf_cuda_free(desc->dev_materialOffsets));
    CUDA_CHK(cudarf_cuda_malloc(&desc->dev_materialOffsets,
                                CUDARF_MAX_DRAW_PACKETS * sizeof(cudarf::visibuf::MaterialOffset)));

    SPDLOG_INFO("{}", fmt::sprintf("Depth buffer: %lu KB", desc->width * desc->height * sizeof(DepthValue) / 1024));

#if defined(WITH_TAA)

    create_surface(desc->dev_framebuffer[0], desc->dev_framebufferTex[0], desc->width, desc->height, cuStream);
    create_surface(desc->dev_framebuffer[1], desc->dev_framebufferTex[1], desc->width, desc->height, cuStream);
    create_surface(desc->rasterSurface, desc->rasterTexture, desc->width, desc->height, cuStream);

    CUDA_CHK(cudaFree(desc->dev_velocityTex));
    CUDA_CHK(cudaMalloc(&desc->dev_velocityTex, sizeof(cudarf::Velocity) * desc->width * desc->height));

    SPDLOG_INFO("{}", fmt::sprintf("Internal framebuffer %d x %d @ 32: %lu KB",
                desc->width, desc->height,
                (desc->width * desc->height * sizeof(ColorN)) / 1024));

    SPDLOG_INFO("{}", fmt::sprintf("Output framebuffer[0] %d x %d @ 32: %lu KB",
                desc->width, desc->height,
                (desc->width * desc->height * sizeof(ColorN)) / 1024));

    SPDLOG_INFO("{}", fmt::sprintf("Output framebuffer[1] %d x %d @ 32: %lu KB",
                desc->width, desc->height,
                (desc->width * desc->height * sizeof(ColorN)) / 1024));

    SPDLOG_INFO("{}", fmt::sprintf("Velocity texture %d x %d: %lu KB",
                desc->width, desc->height,
                (desc->width * desc->height * sizeof(cudarf::Velocity)) / 1024));
#else

    free_surface(desc->dev_framebuffer);
    create_surface(desc->dev_framebuffer, desc->dev_framebufferTex, desc->width, desc->height, cuStream);

    SPDLOG_INFO("{}", fmt::sprintf("Output framebuffer %d x %d @ 32: %lu KB",
                desc->width, desc->height,
                (desc->width * desc->height * sizeof(ColorN)) / 1024));
#endif

#ifdef WITH_TAA
    desc->TAA.pointsHalton.resize(HALTON_POINTS);
    initialize_halton_2_3(desc->TAA.pointsHalton);

    std::string points;
    points.reserve(20);

    for (float2 pt: desc->TAA.pointsHalton) {
        points += ("(" + std::to_string(pt.x) + ", " + std::to_string(pt.y) + ") ");
    }
    SPDLOG_INFO("{}", fmt::sprintf("Halton points: %s", points.c_str()));
#endif
}

void cudarf::pipe::destroy(cudarf::pipe::Ctx *desc) {
    // user is responsible for cleaning draw packets

    CUDA_CHK(cudarf_cuda_free(desc->dev_depthbuffer));
    desc->dev_depthbuffer = NULL;
    CUDA_CHK(cudarf_cuda_free(desc->dev_geom_output));
    desc->dev_geom_output = NULL;
    CUDA_CHK(cudarf_cuda_free(desc->dev_materialOffsets));
    desc->dev_materialOffsets = NULL;

#ifdef WITH_TAA
    free_surface(desc->dev_framebuffer[0]);
    free_surface(desc->dev_framebuffer[1]);
#else
    free_surface(desc->dev_framebuffer);
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
        frameCounter,
        cudarf::pipe::get_internal_fb(rasterization_desc, frameCounter),
        nullptr);

    cudarf::pipe::run_pipe(rasterization_desc,
                   cudarf::RenderParams {
                       face_culling,
                       blending,
                       common,
#ifdef WITH_TAA
                       common,
#endif
                       RenderParams::PBR()
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

void cudarf::pipe::begin_frame(cudarf::pipe::Ctx *desc, unsigned int frameCounter, cudaStream_t cuStream)
{
    cudarf::pipe::clear_depth(desc, cuStream);
    CUDA_CHK(cudaMemsetAsync(desc->dev_geom_output,
                             0xFF,
                             sizeof(cudarf::visibuf::GeomOutput) * desc->width * desc->height,
                             cuStream));
    CUDA_CHK(cudaMemsetAsync(desc->dev_materialOffsets,
                             0xFF,
                             sizeof(cudarf::visibuf::MaterialOffset) * CUDARF_MAX_DRAW_PACKETS,
                             cuStream));

#ifdef WITH_TAA
    cudarf::pipe::clear_framebuffer(desc, desc->rasterSurface, make_uchar4(0, 0, 0, 0), cuStream);
    CUDA_CHK(cudaMemsetAsync(desc->dev_velocityTex, 0, sizeof(cudarf::Velocity) * desc->width * desc->height, cuStream));
#else
    cudarf::pipe::clear_framebuffer(desc,
                                    cudarf::pipe::get_output_fb(desc, frameCounter),
                                    make_uchar4(0, 0, 0, 0),
                                    cuStream);
#endif

}

cudarf::Framebuffer cudarf::pipe::get_output_fb(cudarf::pipe::Ctx *desc, unsigned int frameCounter)
{
#ifdef WITH_TAA
    if (desc->TAAEnabled) {
        return desc->dev_framebuffer[frameCounter % 2];
    } else {
        return desc->rasterSurface;
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
        return desc->rasterTexture;
    }
#else
    return desc->dev_framebufferTex;
#endif
}

#ifdef WITH_TAA

cudarf::Framebuffer cudarf::pipe::get_internal_fb(cudarf::pipe::Ctx *desc, unsigned int frameCounter)
{
    return desc->rasterSurface;
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

#endif
