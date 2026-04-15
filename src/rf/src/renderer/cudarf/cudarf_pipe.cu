// -*- mode: c++ -*-

#ifndef _CUDARF_PIPELINE_CU_
#define _CUDARF_PIPELINE_CU_

// enable for debugging
// #define DUMP_STAGE_OUTPUT
// #define VISUALIZE_BINS
// #define VISUALIZE_TILES
// #define TEST_TILER_BIN

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <set>

#include <spdlog/spdlog.h>

#include <cuda_runtime.h>

#include <rf/renderer/cudarf/cudarf.hpp>
#include "helpers.hpp"

#include "helpers_cudavec.inl"
#include "utils.inl"
#include "types.hpp"
#include "triangle.inl"
#include "tiler_bin.inl"
#include "tiler_coarse.inl"
#include "color_space.inl"
#include "cubemap.inl"
#include "frame_conventions.inl"
#include "framebuffer.inl"
#include "fragment.inl"
#include "TAA.inl"
#include "raster_naive.inl"
#include "test.inl"
#include "visibuf.inl"

#include <cudarf/cudarf_profile.hpp>

// profiling utilities
// -------------------
#define CUDA_TIME_BEGIN(name)                                                   \
    int __it_##name = -1;                                                       \
    if constexpr (CUDARF_ENABLE_CUDA_PROFILING) {                               \
        if (launchConfig.eventDB) {                                              \
            __it_##name = launchConfig.eventDB->start_interval(#name, cuStream); \
        }                                                                        \
    }

#define CUDA_TIME_END(name)                                                     \
    if constexpr (CUDARF_ENABLE_CUDA_PROFILING) {                               \
        if (__it_##name >= 0 && launchConfig.eventDB) {                          \
            launchConfig.eventDB->stop_interval(__it_##name);                    \
        }                                                                        \
    }

using namespace cudarf;
using namespace cudarf::rast;


#ifdef DUMP_STAGE_OUTPUT
VertexOut verts[30000000];
uint8_t subtris[1000000];
cudarf::rast::Triangle triags[1000000];
#endif

__device__ int32_t cudarf::rast::SimpleQueue::push(cudarf::rast::SimpleQueue::Segment &seg, unsigned int limit, int32_t val)
{
    unsigned int writeIdx = atomicAdd(&(seg.queueSize), 1);
    if (writeIdx < limit){
        seg.queue[writeIdx] = val;
        return writeIdx;
    } else {
        return -1;
    }
}

__device__ int32_t cudarf::rast::SimpleQueue::push_unprotected(cudarf::rast::SimpleQueue::Segment &seg, unsigned int limit, int32_t val)
{
    unsigned int writeIdx = seg.queueSize++;
    if (writeIdx < limit){
        seg.queue[writeIdx] = val;
        return writeIdx;
    } else {
        return -1;
    }
}

__device__ void cudarf::rast::SimpleQueue::clear(cudarf::rast::SimpleQueue::Segment &seg)
{
    atomicExch(&(seg.queueSize), 0);
}

__global__
static void init_tilequeues(SimpleQueue::Segment *tileQHeaders, int32_t *tileQData, unsigned int limit, int w, int h) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
        tileQHeaders[index].queueSize = 0;
        tileQHeaders[index].queue = &tileQData[index * limit];
    }
}



__global__
void visualize_bins(const cudarf::rast::PipeParams *params, cudarf::Framebuffer framebuffer)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (!(x < params->windowWidth && y < params->windowHeight)) {return;}

    int index = x + (y * params->windowWidth);

    int binX = x / params->binCtx.binW;
    int binY = y / params->binCtx.binH;
    int binIdx = binX + params->binCtx.binsX * binY;

    bool binOccupied = false;

    for (int SM = 0; SM < CUDARF_BIN_STREAMS_SIZE; SM++) {
        if (params->binCtx.binTotal[(binIdx << CUDARF_BIN_STREAMS_LOG2) + SM] > 0) {
            binOccupied = true;
            break;
        }
    }

    bool isEdge = (x % params->binCtx.binW == 0 || y % params->binCtx.binH == 0);

    cudarf::Color color;
    fb::load(framebuffer, x, y, color);

    if (binOccupied) {
        color.x = 0.2f;
        color.w = 1.0f;
    }
    if (isEdge) {
        color.x = 1.0f;
        color.w = 1.0f;
    }

    fb::store(framebuffer, x, y, color);
}

__global__
void visualize_tiles(const cudarf::rast::PipeParams *params, cudarf::Framebuffer framebuffer)
{
    const BinTilerCtx &ctx = params->binCtx;

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (!(x < params->windowWidth && y < params->windowHeight)) {return;}

    int index = x + (y * params->windowWidth);

    int tileX = x / CUDARF_TILE_SZ;
    int tileY = y / CUDARF_TILE_SZ;

    int binW = params->binCtx.binW;
    int binH = params->binCtx.binH;

    int2 tilesInBin = make_int2(binW / CUDARF_TILE_SZ, binH / CUDARF_TILE_SZ);

	int tileId = tileX + tileY * ctx.binsX * tilesInBin.x;

    cudarf::Color color;
    fb::load(framebuffer, x, y, color);

    if (x % CUDARF_TILE_SZ == 0 || y % CUDARF_TILE_SZ == 0) {
        color.y = 0.2f;
    }
    if (params->tileQHeaders[tileId].queueSize > 0) {
        color.y += 0.2f;
    }
    fb::store(framebuffer, x, y, color);
}

__global__
void visualize_velocity(const cudarf::Velocity *velocity, int width, int height, cudarf::Framebuffer framebuffer)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (!(x < width && y < height)) {return;}

    int index = x + (y * width);

    cudarf::Color color;
    color.x = std::abs(velocity[index].x / 20.0f);
    color.y = std::abs(velocity[index].y / 20.0f);
    color.z = 0.0f;
    color.w = 1.0f;

    fb::store(framebuffer, x, y, color);
}


static void prepare_internal_buffers(PipeInternalBufferSet &bufferSet, std::size_t totalVertices, std::size_t indexCount)
{
    std::size_t total_triangles = indexCount;

    if (totalVertices <= bufferSet.maxVertexCount && indexCount <= bufferSet.maxIndexCount) {
        return;
    }

    // SPDLOG_INFO("reallocated internal buffers: {} vertices, {} indices",
    //             totalVertices, indexCount);

    bufferSet.maxVertexCount = totalVertices;
    bufferSet.maxIndexCount = indexCount;

    int numBins = CUDARF_MAXBINS_SQR;

    // initialize triangle setup output
    // DEBUG override to draw specific number of triangles
    // --

    reinit_buf(&bufferSet.dev_bufVertexOut,
               totalVertices * sizeof(VertexOut));

    reinit_buf(&bufferSet.dev_tri_subtris,  total_triangles * sizeof(uint8_t));


    // printf("%s[#triangles: %d]\n", __func__, triangle_count);

    // allocate buffers for vertex and triangle setup stage
    // --
    reinit_buf(&bufferSet.dev_triangles, total_triangles * sizeof(cudarf::rast::Triangle));

    // estimate bin buffer sizes
    // --
    // TODO justify this number
    int maxBinSegsSlack  = 1024;      // x 2137B  = 534KB
    bufferSet.maxBinSegs = std::max(bufferSet.maxBinSegs,
                   (int) std::max(numBins * CUDARF_BIN_STREAMS_SIZE,
                                  (int) (total_triangles - 1) / CUDARF_BIN_SEG_SIZE + 1) + maxBinSegsSlack);

    reinit_buf(&bufferSet.dev_binSegData, bufferSet.maxBinSegs * CUDARF_BIN_SEG_SIZE * sizeof(int32_t));
    reinit_buf(&bufferSet.dev_binSegNext, bufferSet.maxBinSegs * sizeof(int32_t));
    reinit_buf(&bufferSet.dev_binSegCount, bufferSet.maxBinSegs * sizeof(int32_t));

    // allocate debug buffers
    // --
    // CUDA_CHK(cudarf_cuda_free(bufferSet.dev_dbgbuf));
    // CUDA_CHK(cudarf_cuda_malloc(&bufferSet.dev_dbgbuf, total_triangles * sizeof(int32_t)));
    // CUDA_CHK(cudaMemset(bufferSet.dev_dbgbuf, 0,
    //                     total_triangles * sizeof(int32_t)));

    // CUDA_CHK(cudarf_cuda_free(bufferSet.dev_dbgtiles));
    // CUDA_CHK(cudarf_cuda_malloc(&bufferSet.dev_dbgtiles, bufferSet.width * bufferSet.height * sizeof(int32_t)));
    // CUDA_CHK(cudaMemset(bufferSet.dev_dbgtiles, 0,
    //                     bufferSet.width * bufferSet.height * sizeof(int32_t)));

}

// rasterization pipeline for flat rendering
void cudarf::pipe::run_pipe(cudarf::pipe::Ctx *desc,
                            const cudarf::RenderParams &params,
                            const cudarf::Uniforms &uniforms,
#ifdef WITH_TAA
                            const cudarf::Uniforms &uniformsHist,
#endif
                            const std::vector<unsigned int> &drawPacketIds,
                            const std::vector<unsigned int> &matIds,
                            const cudarf::MaterialMap &materials,
                            const cudarf::pipe::LaunchConfig &launchConfig,
                            const cudaStream_t &cuStream)
{
    std::vector<cudarf::Uniforms> drawPacketUniforms(drawPacketIds.size(), uniforms);

#ifdef WITH_TAA
    std::vector<cudarf::Uniforms> drawPacketUniformsHist(drawPacketIds.size(), uniformsHist);
    cudarf::pipe::run_pipe(desc, params, drawPacketUniforms, drawPacketUniformsHist, drawPacketIds, matIds, materials, launchConfig, cuStream);
#else
    cudarf::pipe::run_pipe(desc, params, drawPacketUniforms, drawPacketIds, matIds, materials, launchConfig, cuStream);
#endif

}

// rasterization pipeline
void cudarf::pipe::run_pipe(cudarf::pipe::Ctx *desc,
                            const cudarf::RenderParams &params,
                            const std::vector<cudarf::Uniforms> &uniforms,
#ifdef WITH_TAA
                            const std::vector<cudarf::Uniforms> &uniformsHist,
#endif
                            const std::vector<unsigned int> &drawPacketIds,
                            const std::vector<unsigned int> &matIds,
                            const cudarf::MaterialMap &materials,
                            const cudarf::pipe::LaunchConfig &launchConfig,
                            const cudaStream_t &cuStream)
{
    assert(drawPacketIds.size() > 0);
    assert(drawPacketIds.size() == matIds.size());
    assert(drawPacketIds.size() < CUDARF_DRAW_PACKET_BATCH_LIMIT);
    assert(uniforms.size() == drawPacketIds.size());

#ifndef NDEBUG
    {
        // visibuf material-offset generation assumes the global material ids
        // form a dense [0, materialCount) range.
        assert(!materials.empty());

        std::set<unsigned int> materialIdsDense;
        for (const auto &[materialId, material]: materials) {
            (void)material;
            assert(materialId >= 0);
            materialIdsDense.insert(static_cast<unsigned int>(materialId));
        }

        assert(materialIdsDense.size() == materials.size());

        unsigned int expectedId = 0;
        for (unsigned int materialId: materialIdsDense) {
            assert(materialId == expectedId);
            expectedId++;
        }

        for (unsigned int materialId: matIds) {
            assert(materialId < materials.size());
        }
    }
#endif

    cudarf::rast::PipeParams pipe;

    int rasterizerW = round_up_to_mult_pwr(desc->width, CUDARF_BIN_LOG2 + CUDARF_TILE_LOG2);
    int rasterizerH = round_up_to_mult_pwr(desc->height, CUDARF_BIN_LOG2 + CUDARF_TILE_LOG2);


    unsigned int *drawPacketOrder = pipe.drawPacketOrder;

    pipe.vtxOffsets[0] = 0;
    pipe.idxOffsets[0] = 0;

    unsigned int total_indices = 0;
    unsigned int totalVertices = 0;

    unsigned int vtxPrev = desc->drawPackets[drawPacketIds[0]].vertCount;
    unsigned int idxPrev = desc->drawPackets[drawPacketIds[0]].index_count;

    int ord = 0;

    cudarf::ShaderType commonShaderType = materials.at(matIds[0])->type;

    std::fill_n(pipe.drawPacketMaterials, CUDARF_MAX_DRAW_PACKETS, 0xFFFFFFFFu);

    // build running sums of index and vertex count over draw packets, which
    // correspond to per-packet vtx/idx offsets within the combined streams
    for (auto id: drawPacketIds) {
        assert(id < CUDARF_MAX_DRAW_PACKETS);
        drawPacketOrder[ord] = id;
        pipe.drawPacketMaterials[id] = matIds[ord];
        totalVertices += desc->drawPackets[id].vertCount;
        total_indices += desc->drawPackets[id].index_count;

        // run_pipe supports only one shader type
        assert (materials.at(matIds[ord])->type == commonShaderType);

        if (ord == 0) {ord++; continue; }

        pipe.vtxOffsets[ord] = vtxPrev;
        pipe.idxOffsets[ord] = idxPrev;

        vtxPrev += desc->drawPackets[id].vertCount;
        idxPrev += desc->drawPackets[id].index_count;
        ord++;
    }

    unsigned int total_triangles = total_indices / PRIM_ELEMS;
    SPDLOG_DEBUG("PIPE: draw packets to render: {} | triangles: {}\n", drawPacketIds.size(), total_triangles);

    // printf("----------------------------------\ndraw packets: %zu | total vertices = %u, indices = %u\n",
    //        drawPacketIds.size(), totalVertices, total_indices);

    CUDA_TIME_BEGIN(init_buffers);

    // initialize all uniforms
    CUDA_CHK(cudaMemcpyAsync(desc->dev_uniforms, uniforms.data(), sizeof(Uniforms) * drawPacketIds.size(),
                             cudaMemcpyHostToDevice, cuStream));

#ifdef WITH_TAA
    assert(uniforms.size() == uniformsHist.size());
    assert(uniformsHist.size());
    CUDA_CHK(cudaMemcpyAsync(desc->dev_uniformsHist, uniformsHist.data(), sizeof(Uniforms) * drawPacketIds.size(),
                             cudaMemcpyHostToDevice, cuStream));
#endif

    PipeInternalBufferSet &bufferSet = desc->internalBufs;

    prepare_internal_buffers(bufferSet, totalVertices, total_indices);

    // no necessity in cleaning this memory, it will be overwritten completely
    // CUDA_CHK(cudaMemsetAsync(bufferSet.dev_triangles, 0,
    //                          total_triangles * sizeof(cudarf::Triangle),
    //                          cuStream));

    CUDA_CHK(cudaMemsetAsync(bufferSet.dev_tri_subtris, 0, total_triangles * sizeof(uint8_t), cuStream));

    if (desc->dev_geom_output) {
        // Geom output is consumed by visibuf stages in the current run_pipe call.
        // Clear it here so stale draw-packet ids from earlier passes do not leak in.
        CUDA_CHK(cudaMemsetAsync(desc->dev_geom_output,
                                 0xFF,
                                 sizeof(cudarf::visibuf::GeomOutput) * desc->width * desc->height,
                                 cuStream));
    }

    // init bin tiler buffers
    {
        CUDA_CHK(cudaMemsetAsync(desc->dev_binTotal, 0, CUDARF_MAXBINS_SQR * CUDARF_BIN_STREAMS_SIZE * sizeof(int32_t), cuStream));
    }

    // init coarse tiler buffers
    // --
    {
        dim3 blockSize2d(8, 8);
        dim3 blockCount2d((rasterizerW / CUDARF_TILE_SZ  - 1) / blockSize2d.x + 1,
                          (rasterizerH / CUDARF_TILE_SZ - 1) / blockSize2d.y + 1);

        init_tilequeues<<<blockCount2d, blockSize2d, 0, cuStream>>>(
            desc->dev_tileQHeaders, desc->dev_tileQData, desc->tileQLimit,
            rasterizerW / CUDARF_TILE_SZ, rasterizerH / CUDARF_TILE_SZ);
        CUDA_CHK_ERROR("init_tilequeues");
    }

    // not needed for now, as we clean rasterbuf in rasterizer
    // init fine raster buffers
    // --
    /*
    {
        dim3 blockSize2d(8, 8);
        dim3 blockCount2d((rasterizerW  - 1) / blockSize2d.x + 1,
                          (rasterizerH - 1) / blockSize2d.y + 1);

        CUDA_TIME_BEGIN(init_raster);
        init_rasterbuf<<<blockCount2d, blockSize2d, 0, cuStream>>>(desc->dev_rasterOut, rasterizerW, rasterizerH);
        CUDA_TIME_END(init_raster);
    }
    */

    // reinit_buf(&bufferSet.dev_dbgbuf, total_triangles * sizeof(int32_t));

    // initialize constant memory
    // --
    pipe.withFaceCulling = params.face_culling;
    pipe.withBlending = params.with_blending;

    pipe.windowWidth = desc->width;
    pipe.windowHeight = desc->height;

    pipe.rasterizerSize = make_int2(round_up_to_mult_pwr(desc->width, CUDARF_BIN_LOG2 + CUDARF_TILE_LOG2),
                                     round_up_to_mult_pwr(desc->height, CUDARF_BIN_LOG2 + CUDARF_TILE_LOG2));

    for (auto id: drawPacketIds) {
        pipe.drawPackets[id] = desc->drawPackets[id];
    }

    pipe.uniforms = desc->dev_uniforms;
    pipe.vertexOut = bufferSet.dev_bufVertexOut;
    pipe.drawPacketCount = drawPacketIds.size();

#ifdef WITH_TAA
    if (desc->TAAEnabled && !params.with_blending) {
        pipe.common = prepare_for_TAA(
            *desc,
            launchConfig.frameCounter,
            4,
            params.common);
    }
    else {
        pipe.common = params.common;
    }

    pipe.taa.commonHist = params.commonHist;
    pipe.taa.uniformsHist = desc->dev_uniformsHist;
#else
    pipe.common = params.common;
#endif

    pipe.tris = bufferSet.dev_triangles;
    pipe.numTriangles = total_triangles;
    pipe.maxSubtris = total_triangles; // TODO: revise when we implement clipping

    pipe.triSubtris = static_cast<uint8_t *>(bufferSet.dev_tri_subtris);
    pipe.dbgbuf = bufferSet.dev_dbgbuf;
    pipe.clockRate = desc->clockRate;

    // Select batch size for bin tiler and estimate buffer sizes
    // --
    // TODO draw scheme of all buffers in bin tiler
    {
        pipe.binCtx.binsX = CUDARF_BIN_COUNT;
        pipe.binCtx.binsY = CUDARF_BIN_COUNT;
        pipe.binCtx.binW = pipe.rasterizerSize.x / pipe.binCtx.binsX;
        pipe.binCtx.binH = pipe.rasterizerSize.y / pipe.binCtx.binsY;

        pipe.binCtx.numBins = pipe.binCtx.binsX * pipe.binCtx.binsY;

        // how many triangles are to be processed on one SM in one cycle (512) 
        int roundSize  = CUDARF_BIN_WARPS * 32;

        // minimal number of batches in tiler_bin kernel invocation
        int minBatches = CUDARF_BIN_STREAMS_SIZE * 2;
        int maxRounds  = 32;

        // number of triangles to be processed in tiler_bin kernel invocation
        pipe.binCtx.binBatchSize =
            iclamp(total_triangles / (roundSize * minBatches), 1, maxRounds) * roundSize;

        pipe.binCtx.maxBinSegs = bufferSet.maxBinSegs;
        pipe.binCtx.binFirstSeg = desc->dev_binFirstSeg;
        pipe.binCtx.binTotal = desc->dev_binTotal;
        pipe.binCtx.binSegData = bufferSet.dev_binSegData;
        pipe.binCtx.binSegNext = bufferSet.dev_binSegNext;
        pipe.binCtx.binSegCount = bufferSet.dev_binSegCount;
        // pipe.dbgtiles = bufferSet.dev_dbgtiles;
    }

    // coarse tiler
    {
        pipe.tileQHeaders = desc->dev_tileQHeaders;
        pipe.tileQData = desc->dev_tileQData;
        pipe.tileQLimit = desc->tileQLimit;
    }

    assert(materials.size() < CUDARF_MAX_MATERIALS);

    for (const auto &it: materials) {
        pipe.materials[it.first] = *it.second;
    }

    // set PBR context
    {
        pipe.camera = params.pbr.camera;
        pipe.exposure = params.pbr.exposure;
        pipe.lightCount = params.pbr.lights.size();

        for (int i = 0; i < pipe.lightCount; i++) {
            pipe.lights[i] = params.pbr.lights[i];
        }

        pipe.sphericalHarmonics = params.pbr.sphericalHarmonics;
        pipe.brdfLUT = params.pbr.brdfLUT;
        pipe.specular = params.pbr.specular;
    }

#ifdef WITH_TAA
    pipe.taa.velocityTex = desc->dev_velocityTex;
    pipe.taa.velocityThreshold = desc->TAA.velocityThreshold;
#endif

    CUDA_CHK(cudaMemcpyAsync(desc->dev_pipeParams, &pipe, sizeof(cudarf::rast::PipeParams), cudaMemcpyHostToDevice, cuStream));

    // initialize atomics
    // --
    cudarf::pipe::Atomics pipe_atomics{};
    pipe_atomics.subtris_count = total_triangles;
    pipe_atomics.null_tris = 0;

    pipe_atomics.bin_counter = 0;
    pipe_atomics.numBinSegs = 0;

    pipe_atomics.coarseCounter = 0;

    pipe_atomics.dbg_oft = 0;

    CUDA_CHK(cudaMemcpyAsync(desc->dev_pipeAtomics, &pipe_atomics, sizeof(cudarf::pipe::Atomics), cudaMemcpyHostToDevice, cuStream));

    CUDA_TIME_END(init_buffers);

    CUDA_TIME_BEGIN(total_vertex_transform);

    {
        int threadsPerBlock = 256;
        int blocksPerGrid = (totalVertices + threadsPerBlock - 1) / threadsPerBlock;

        if (desc->TAAEnabled) {
            if (launchConfig.withTexturing) {
                if (commonShaderType == SHADER_TYPE_PBR) {
                    vertex_transform<cudarf::SHADER_TYPE_PBR, true, true>
                        <<<blocksPerGrid, threadsPerBlock, 0, cuStream>>>
                        (desc->dev_pipeParams, totalVertices);
                }
                else {
                    vertex_transform<cudarf::SHADER_TYPE_UNLIT, true, true>
                        <<<blocksPerGrid, threadsPerBlock, 0, cuStream>>>
                        (desc->dev_pipeParams, totalVertices);
                }
            }
            else {
                if (commonShaderType == SHADER_TYPE_PBR) {
                    vertex_transform<cudarf::SHADER_TYPE_PBR, false, true>
                        <<<blocksPerGrid, threadsPerBlock, 0, cuStream>>>
                        (desc->dev_pipeParams, totalVertices);
                }
                else {
                    vertex_transform<cudarf::SHADER_TYPE_UNLIT, false, true>
                        <<<blocksPerGrid, threadsPerBlock, 0, cuStream>>>
                        (desc->dev_pipeParams, totalVertices);
                }
            }
        }
        else {
            if (launchConfig.withTexturing) {
                if (commonShaderType == SHADER_TYPE_PBR) {
                    vertex_transform<cudarf::SHADER_TYPE_PBR, true, false>
                        <<<blocksPerGrid, threadsPerBlock, 0, cuStream>>>
                        (desc->dev_pipeParams, totalVertices);
                }
                else {
                    vertex_transform<cudarf::SHADER_TYPE_UNLIT, true, false>
                        <<<blocksPerGrid, threadsPerBlock, 0, cuStream>>>
                        (desc->dev_pipeParams, totalVertices);
                }
            }
            else {
                if (commonShaderType == SHADER_TYPE_PBR) {
                    vertex_transform<cudarf::SHADER_TYPE_PBR, false, false>
                        <<<blocksPerGrid, threadsPerBlock, 0, cuStream>>>
                        (desc->dev_pipeParams, totalVertices);
                }
                else {
                    vertex_transform<cudarf::SHADER_TYPE_UNLIT, false, false>
                        <<<blocksPerGrid, threadsPerBlock, 0, cuStream>>>
                        (desc->dev_pipeParams, totalVertices);
                }
            }
        }
        CUDA_CHK_ERROR("vertex_transform");
    }

    CUDA_TIME_END(total_vertex_transform);

    // dump vertex transform output
    #ifdef DUMP_STAGE_OUTPUT
    {
        CUDA_CHK(cudaMemcpyAsync(verts, bufferSet.dev_bufVertexOut,
                            bufferSet.vertCount * sizeof(VertexOut), cudaMemcpyDeviceToHost));

        for (int i = 0; i < bufferSet.vertCount; i++) {
            if (i >= 3) break;
            dump_vert("after VT", " | ", verts[i].pos);
            dump_vert("col", "\n", verts[i].col);
        }
        printf("\n");
    }
    #endif

    CUDA_TIME_BEGIN(total_triangle_assembly);

    {
        int threadsPerBlock = 256;
        int blocksPerGrid = (total_triangles + threadsPerBlock - 1) / threadsPerBlock;

        if (launchConfig.withTexturing) {
            if (commonShaderType == SHADER_TYPE_PBR) {
                triangle_assembly<cudarf::SHADER_TYPE_PBR, true><<<blocksPerGrid, threadsPerBlock, 0, cuStream>>>
                    (desc->dev_pipeParams, total_triangles);
            }
            else {
                triangle_assembly<cudarf::SHADER_TYPE_UNLIT, true><<<blocksPerGrid, threadsPerBlock, 0, cuStream>>>
                    (desc->dev_pipeParams, total_triangles);
            }
        }
        else {
            if (commonShaderType == SHADER_TYPE_PBR) {
                triangle_assembly<cudarf::SHADER_TYPE_PBR, false><<<blocksPerGrid, threadsPerBlock, 0, cuStream>>>
                    (desc->dev_pipeParams, total_triangles);
            }
            else {
                triangle_assembly<cudarf::SHADER_TYPE_UNLIT, false><<<blocksPerGrid, threadsPerBlock, 0, cuStream>>>
                    (desc->dev_pipeParams, total_triangles);
            }
        }

        CUDA_CHK_ERROR("triangle_assembly");
    }

    CUDA_TIME_END(total_triangle_assembly);

    #ifdef DUMP_STAGE_OUTPUT
    {
        CUDA_CHK(cudaMemcpyAsync(&subtris, bufferSet.dev_tri_subtris,
                            triangle_count * sizeof(uint8_t), cudaMemcpyDeviceToHost));

        CUDA_CHK(cudaMemcpyAsync(&triags, bufferSet.dev_triangles,
                            triangle_count * sizeof(cudarf::rast::Triangle), cudaMemcpyDeviceToHost));

        for (unsigned int i = 0; i < triangle_count; i++) {
            if (i >= 3) break;

            if (!subtris[i]) {continue;}
            char pfx[128]; snprintf(pfx, 128, "TRI[%u]", i);
            dump_triag(pfx, triags[i]);
        }
    }
    #endif

    {
        // one bin rasterizer block is run per SM, each containing 32 warps =>
        // 512 threads
        CUDA_TIME_BEGIN(tiler_bin);
        tiler_bin<<<dim3(CUDARF_BIN_STREAMS_SIZE), dim3(32, CUDARF_BIN_WARPS), 0, cuStream>>>
            (desc->dev_pipeParams, desc->dev_pipeAtomics, bufferSet.dev_triangles);

        CUDA_CHK_ERROR("tiler_bin");

        CUDA_TIME_END(tiler_bin);

        // TODO do this once for every input size change in prepare internal buffers
        // check that bin tiler buffers haven't overflown

        CUDA_TIME_BEGIN(memcpy_d2h);
        CUDA_CHK(cudaMemcpyAsync(&pipe_atomics, desc->dev_pipeAtomics, sizeof(cudarf::pipe::Atomics),
                                 cudaMemcpyDeviceToHost, cuStream));

        CUDA_TIME_END(memcpy_d2h);

        // TODO: re-launch pipe if bin segments have overflown
        if(pipe_atomics.numBinSegs > (unsigned int) bufferSet.maxBinSegs) {
            printf("%s:%d not implemented (pipe_atomics.numBinSegs(%d) > bufferSet.maxBinSegs(%d))\n",
                   __FILE__, __LINE__, pipe_atomics.numBinSegs, bufferSet.maxBinSegs);
            fflush(stdout);
            assert(false);
            return;
        }

#ifdef TEST_TILER_BIN
        test_bin_output(*desc, pipe, pipe_atomics, cuStream);
#endif
    }

    {
        // dim3 gridSize(1);         // DEBUG: use 1x1 grid to launch only on one thread block
        dim3 gridSize(desc->SMPCount, 1);
        dim3 blockSize(32, CUDARF_COARSE_WARPS);
        CUDA_TIME_BEGIN(tilerCoarse);

        if (params.with_blending) {
            tilerCoarse<true>
            <<<gridSize, blockSize, 0, cuStream >>>(desc->dev_pipeParams, desc->dev_pipeAtomics);
        } else {
            tilerCoarse<false>
            <<<gridSize, blockSize, 0, cuStream >>>(desc->dev_pipeParams, desc->dev_pipeAtomics);
        }

        CUDA_CHK_ERROR("tilerCoarse");
        CUDA_TIME_END(tilerCoarse);

        // DEBUG: test to check that coarse tiler has processed all triangles
        // will flag if any triangle is culled/clipped
        // CUDA_CHK(cudaMemcpyAsync(&bin_tiler_mask, pipe.dbgbuf,
        //                          total_triangles * sizeof(int32_t), cudaMemcpyDeviceToHost, cuStream));

        // std::size_t unprocessedCount = 0;

        // for (uint i = 0; i < total_triangles; i++) {
        //     if (bin_tiler_mask[i] == 0) {unprocessedCount++;}
        // }

        // if (unprocessedCount) {
        //     printf("WARNING coarse tiler: %zu triangles not consumed\n", unprocessedCount);
        // }
    }

#ifdef WITH_TAA
        // passing cudarf::LinearSurface increases register pressure, so pass
        // pointer here

        cudarf::Framebuffer framebuffer;

        if (launchConfig.nativeOutput) {
            framebuffer = launchConfig.nativeOutput;
        }
        else {
            framebuffer = desc->rasterSurface;
        }
#else
        cudarf::Framebuffer framebuffer;

        if (launchConfig.nativeOutput) {
            framebuffer = launchConfig.nativeOutput;
        }
        else {
            framebuffer = desc->dev_framebuffer;
        }
#endif
        assert(framebuffer);
    // fine raster phase
    {
        dim3 blockSize2d(8, 8);
        dim3 blockCount2d((desc->width - 1) / blockSize2d.x + 1,
                          (desc->height - 1) / blockSize2d.y + 1);


        CUDA_TIME_BEGIN(fine_raster_naive);

        if (launchConfig.withTexturing) {
            if (params.with_blending) {
                if (commonShaderType == SHADER_TYPE_PBR) {
                    fine_raster_naive<true, SHADER_TYPE_PBR, true><<<blockCount2d, blockSize2d, 0, cuStream>>>
                        (desc->dev_pipeParams, framebuffer, desc->dev_depthbuffer, desc->dev_geom_output);
                } else {
                    fine_raster_naive<true, SHADER_TYPE_UNLIT, true><<<blockCount2d, blockSize2d, 0, cuStream>>>
                        (desc->dev_pipeParams, framebuffer,  desc->dev_depthbuffer, desc->dev_geom_output);
                }
            }
            else {
                if (commonShaderType == SHADER_TYPE_PBR) {
                    fine_raster_naive<false, SHADER_TYPE_PBR, true><<<blockCount2d, blockSize2d, 0, cuStream>>>
                        (desc->dev_pipeParams, framebuffer, desc->dev_depthbuffer, desc->dev_geom_output);
                } else {
                    fine_raster_naive<false, SHADER_TYPE_UNLIT, true><<<blockCount2d, blockSize2d, 0, cuStream>>>
                        (desc->dev_pipeParams, framebuffer, desc->dev_depthbuffer, desc->dev_geom_output);
                }
            }
        }
        else {
            if (params.with_blending) {
                if (commonShaderType == SHADER_TYPE_PBR) {
                    fine_raster_naive<true, SHADER_TYPE_PBR, false><<<blockCount2d, blockSize2d, 0, cuStream>>>
                        (desc->dev_pipeParams, framebuffer, desc->dev_depthbuffer, desc->dev_geom_output);
                } else {
                    fine_raster_naive<true, SHADER_TYPE_UNLIT, false><<<blockCount2d, blockSize2d, 0, cuStream>>>
                        (desc->dev_pipeParams, framebuffer, desc->dev_depthbuffer, desc->dev_geom_output);
                }
            }
            else {
                if (commonShaderType == SHADER_TYPE_PBR) {
                    fine_raster_naive<false, SHADER_TYPE_PBR, false><<<blockCount2d, blockSize2d, 0, cuStream>>>
                        (desc->dev_pipeParams, framebuffer, desc->dev_depthbuffer, desc->dev_geom_output);
                } else {
                    fine_raster_naive<false, SHADER_TYPE_UNLIT, false><<<blockCount2d, blockSize2d, 0, cuStream>>>
                        (desc->dev_pipeParams, framebuffer, desc->dev_depthbuffer, desc->dev_geom_output);
                }
            }
        }

        CUDA_TIME_END(fine_raster_naive);

        CUDA_CHK_ERROR("fine_raster_naive");
    }

    // visibuf pixel countig stage
    if (desc->dev_geom_output)
    {
        dim3 blockSize2d(8, 8);
        dim3 blockCount2d((desc->width - 1) / blockSize2d.x + 1,
                          (desc->height - 1) / blockSize2d.y + 1);

        CUDA_TIME_BEGIN(visibuf_count_pixels);
        assert(desc->dev_geom_output);
        visibuf_count_pixels<<<blockCount2d, blockSize2d, 0, cuStream>>>
            (desc->dev_pipeParams, desc->dev_geom_output, desc->dev_pipeAtomics);

        CUDA_TIME_END(visibuf_count_pixels);

        CUDA_CHK_ERROR("visibuf_count_pixels");

        // CUDA_CHK(cudaMemcpyAsync(&pipe_atomics,
        //                          desc->dev_pipeAtomics,
        //                          sizeof(cudarf::pipe::Atomics),
        //                          cudaMemcpyDeviceToHost,
        //                          cuStream));
        // CUDA_CHK(cudaStreamSynchronize(cuStream));

        // std::vector<bool> printedMaterialIds(CUDARF_MAX_DRAW_PACKETS, false);
        // printf("visibuf material pixel counts:");
        // for (unsigned int matId: matIds) {
        //     if (matId >= CUDARF_MAX_DRAW_PACKETS || printedMaterialIds[matId]) {
        //         continue;
        //     }

        //     printedMaterialIds[matId] = true;
        //     printf(" [mat %u]=%u", matId, pipe_atomics.visibuf.materialPixelCount[matId]);
        // }
        // printf("\n");
    }

    // build material offsets using prefix sums
    if (desc->dev_materialOffsets && desc->dev_xyCommands)
    {
        dim3 blockSize2d(materials.size(), 1);
        dim3 blockCount2d(1, 1);

        CUDA_TIME_BEGIN(visibuf_material_offsets);

        visibuf_build_material_offsets<<<blockCount2d, blockSize2d, 0, cuStream>>>
            (static_cast<uint32_t>(materials.size()), desc->dev_pipeAtomics, desc->dev_materialOffsets);

        CUDA_TIME_END(visibuf_material_offsets);

        CUDA_CHK_ERROR("visibuf_build_material_offsets");
    }

    // visibuf pixel counting stage
    if (desc->dev_geom_output && desc->dev_materialOffsets && desc->dev_xyCommands)
    {
        dim3 blockSize2d(8, 8);
        dim3 blockCount2d((desc->width - 1) / blockSize2d.x + 1,
                          (desc->height - 1) / blockSize2d.y + 1);

        CUDA_TIME_BEGIN(visibuf_build_xy_lists);
        assert(desc->dev_geom_output);
        visibuf_build_xy_lists<<<blockCount2d, blockSize2d, 0, cuStream>>>
            (desc->dev_pipeParams, materials.size(), desc->dev_geom_output, desc->dev_pipeAtomics,
             desc->dev_materialOffsets, desc->dev_xyCommands);

        CUDA_TIME_END(visibuf_build_xy_lists);

        CUDA_CHK_ERROR("visibuf_build_xy_lists");
    }

    // fetch pixel counts for each material
    // TODO: launch cuda kernels using smth like vulkan indirect dispatch
    {
        CUDA_CHK(cudaMemcpyAsync(&pipe_atomics,
                                 desc->dev_pipeAtomics,
                                 sizeof(cudarf::pipe::Atomics),
                                 cudaMemcpyDeviceToHost,
                                 cuStream));

        CUDA_CHK(cudaStreamSynchronize(cuStream));
    }

    // build list of all material ids that participate in this draw call
    std::set<uint32_t> activeMaterials;
    std::copy(std::begin(matIds), std::end(matIds), std::inserter(activeMaterials, activeMaterials.end()));

    if (desc->dev_geom_output && desc->dev_materialOffsets && desc->dev_xyCommands)
    {
        CUDA_TIME_BEGIN(visibuf_material_pass);

        for (auto i: activeMaterials) {
            dim3 blockSize(256);
            dim3 blockCount(pipe_atomics.visibuf.materialPixelCount[i] - 1 / blockSize.x + 1);

            visibuf_material_pass<<<blockCount, blockSize, 0, cuStream>>>
                (desc->dev_pipeParams, desc->dev_geom_output, desc->dev_pipeAtomics,
                 desc->dev_materialOffsets, desc->dev_xyCommands, i, framebuffer);

            CUDA_CHK_ERROR("visibuf_material_pass");
        }

        CUDA_TIME_END(visibuf_material_pass);
    }

    // DEBUG: separate fragment shader stage
    // if (!params.with_blending) {
    //     dim3 blockSize2d(8, 8);
    //     dim3 blockCount2d((desc->width - 1) / blockSize2d.x + 1,
    //                       (desc->height - 1) / blockSize2d.y + 1);

    //     CUDA_TIME_BEGIN(fragmentShaderPBR);

    //     if (commonShaderType == SHADER_TYPE_PBR) {
    //         fragmentShaderPBR<<<blockCount2d, blockSize2d, 0, cuStream>>>
    //             (desc->dev_pipeParams, desc->dev_framebuffer);
    //     } else {
    //         fragmentShaderFlat<<<blockCount2d, blockSize2d, 0, cuStream>>>
    //             (desc->dev_pipeParams, desc->dev_framebuffer);
    //     }

    //     CUDA_TIME_END(fragmentShaderPBR);

    //     CUDA_CHK_ERROR("fragmentShaderPBR");
    // }

    // DEBUG: triangle rasterizer
    #if 0
    {
        RasterizationOverrides overrides;
        int threadsPerBlock = 256;
        int blocksPerGrid = (total_triangles + threadsPerBlock - 1) / threadsPerBlock;

        render_naive<<<blocksPerGrid, threadsPerBlock, 0, cuStream>>>(
            desc->dev_pipeParams, desc->dev_pipeAtomics,
            desc->width, desc->height, bufferSet.dev_triangles, total_triangles,
            desc->dev_depthbuffer, desc->dev_framebuffer, overrides);
    }
    #endif

    #ifdef VISUALIZE_BINS
    {
        int sideLength2d = 8;
        dim3 blockSize2d(sideLength2d, sideLength2d);
        dim3 blockCount2d((desc->width - 1) / blockSize2d.x + 1,
                          (desc->height - 1) / blockSize2d.y + 1);

#ifdef WITH_TAA
        cudarf::Framebuffer framebuffer = desc->rasterSurface;
#else
        cudarf::Framebuffer framebuffer = desc->dev_framebuffer;
#endif

        visualizeBins<<<blockCount2d, blockSize2d, 0, cuStream>>>(
            desc->dev_pipeParams, framebuffer);

        CUDA_CHK_ERROR("visualizeBins");
    }
#endif

#ifdef VISUALIZE_TILES
    {
#ifdef WITH_TAA
        cudarf::Framebuffer framebuffer = desc->rasterSurface;
#else
        cudarf::Framebuffer framebuffer = desc->dev_framebuffer;
#endif
        dim3 blockSize2d(8, 8);
        dim3 blockCount2d((desc->width  - 1) / blockSize2d.x + 1,
                          (desc->height - 1) / blockSize2d.y + 1);

        visualizeTiles <<< blockCount2d, blockSize2d, 0, cuStream >>>(desc->dev_pipeParams, framebuffer);

        CUDA_CHK_ERROR("visualizeTiles");
    }
#endif

}

#ifdef WITH_TAA
void cudarf::pipe::TAA(cudarf::pipe::Ctx *desc,
               const cudarf::CommonUniforms &frameUniforms,
               const cudarf::CommonUniforms &histUniforms,
               const cudarf::ProjectionParams &projection,
               unsigned int frameCounter,
               cudaStream_t cuStream)
{
    assert(desc->TAAEnabled);

    auto jitter = get_jitter(*desc, frameCounter);

    dim3 blockSize2d(8, 8);
    dim3 blockCount2d((desc->width - 1) / blockSize2d.x + 1,
                      (desc->height - 1) / blockSize2d.y + 1);

    SPDLOG_DEBUG("TAA begin: frame: {}, Q: {}, jitter: ({}, {})\n",
                 frameCounter, frameCounter % 4, jitter.x, jitter.y);

    resolve_TAA<true> <<<blockCount2d, blockSize2d, 0, cuStream>>> (
        projection,
        desc->width,
        desc->height,
        desc->rasterTexture,
        get_history_tex(desc, frameCounter),
        desc->dev_velocityTex,
        frameUniforms,
        histUniforms,
        get_output_fb(desc, frameCounter),
        desc->dev_depthbuffer,
        desc->TAA.feedback,
        jitter);

    CUDA_CHK_ERROR("resolve_TAA");

    // DEBUG: visualize velocity buffer
    // visualize_velocity<<<blockCount2d, blockSize2d, 0, cuStream>>>
    //     (desc->dev_velocityTex, desc->width, desc->height, get_output_fb(desc, frameCounter));
}
#endif


#endif
