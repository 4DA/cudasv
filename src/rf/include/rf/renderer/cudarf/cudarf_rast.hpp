#ifndef CUDARF_RAST_HPP
#define CUDARF_RAST_HPP

// cudarf_rast.hpp — GPU kernel-internal rasterization types (cudarf::rast)
//
// Self-contained: does not include cudarf.hpp. The cudarf types that rast
// depends on by value (Uniforms, CommonUniforms, CUDARFLight, DrawPacket)
// are defined here directly; types used only by pointer are forward-declared.
//
// cudarf.hpp includes this file and builds the host-facing API on top of it.

#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>

#include <rf/renderer/cudarf/material.hpp>
#include <rf/renderer/cudarf/types.hpp>
#include <rf/renderer/glm_common.hpp>

// quality knobs
// -----------------------------------------------------------------------------

// pipeline capacity limits
// -----------------------------------------------------------------------------

#define CUDARF_MAX_LIGHTS           10
#define HALTON_POINTS               16
#define CUDARF_MAX_MATERIALS        128
#define CUDARF_SCRATCH_DRAW_PACKET   0
#define CUDARF_MAX_DRAW_PACKETS      512
#define CUDARF_DRAW_PACKET_BATCH_LIMIT 512

namespace cudarf
{

// ---------------------------------------------------------------------------
// Shared host/device types referenced by value inside cudarf::rast::PipeParams.
// Defined here so that cudarf_rast.hpp stays self-contained.
// ---------------------------------------------------------------------------

// uniforms specific to one draw packet (vertex transform)
struct Uniforms {
    glm::mat4 PVM;
    glm::mat4 M;         // used for PBR world-space shading
    glm::mat4 N;         // inverse-transpose normal matrix
};

struct CommonUniforms {
    glm::mat4 PV;
    glm::mat4 P;
    glm::mat4 V;
    glm::mat4 V_inv;
};

struct CUDARFLight {
    float intensity;
    float3 position;
    float range;
};

// ---------------------------------------------------------------------------
// cudarf::rast — GPU kernel-internal types
// (first block: per-vertex / per-triangle / tiler types, no host deps)
// ---------------------------------------------------------------------------

namespace rast
{

/// Simple queue with atomic operations
///
namespace SimpleQueue {
    struct Segment {
        int32_t queueSize = 0;
        int32_t *queue;
    };

    __device__ int32_t push(SimpleQueue::Segment &seg, unsigned int limit, int32_t val);
    __device__ int32_t push_unprotected(SimpleQueue::Segment &seg, unsigned int limit, int32_t val);
    __device__ void clear(SimpleQueue::Segment &seg);
}

struct VertexIn {
    Vec3f pos;
    Vec3f nor;
    Vec2f tex;
    Vec4f tan;
    Color col;
};

struct VertexOut {
    Vec4f pos_3dhp;      // vertex position in homogeneous 3DHP space

#ifdef WITH_TAA
    float2 pos_ss_hist;  // history vertex position in [0; W) x [0; H) screen space
#endif

    Color col;
    Vec3f pos_world;
    float3 nor;
    Vec3f tan;
    float bitan_sgn;
    float2 tex;
    float2 jitter;
};

struct AABB {
    float3 min;
    float3 max;
};

struct Triangle {
    // P0, P1, P2 scaled to [0; screen width] x [0; screen height]
    float2 sP0;
    float2 sP1;
    float2 sP2;

#ifdef WITH_TAA
    float2 v_ss_hist[3];
#endif

    Vec3f zw;        // coefficients for z/w
    Vec3f w_rcp;     // coefficients for 1/w
    float zw_min;    // = min(zw0, zw1, zw2)
    float2 flo;
    float2 fhi;
    float area_rcp;  // 1 / 2area(P0, P1, P2)

    float3 v_world[3];
    Color col[3];
    float3 normal[3];
    float3 tangent[3];
    int bitan_sgn;
    float2 tex[3];

    unsigned char materialId;
    uint8_t isBackFacing;
    uint32_t id;
    uint32_t drawPacketId;

    Triangle() = default;
};

struct Fragment {
    unsigned char materialId;

    cudarf::Color vertexColor;
    float3 pos_global;

    float3 normal;
    float3 tangent;
    float3 bitangent;

    float2 tex;

    float albedoLod;
    float normalLod;
    float emissiveLod;
    float metRoughLod;

#ifdef WITH_TAA
    float2 pos_ss_hist;
#endif
};

struct BinTilerCtx {
    uint32_t binBatchSize;
    int32_t maxBinSegs;
    int32_t numBins;

    int32_t binsX;
    int32_t binsY;
    int32_t binW;
    int32_t binH;

    // size: CUDARF_MAXBINS_SQR * CUDARF_BIN_STREAMS_SIZE * (S32 segIdx), -1 = none
    int32_t *binFirstSeg;
    // size: CUDARF_MAXBINS_SQR * CUDARF_BIN_STREAMS_SIZE * (S32 numTris)
    int32_t *binTotal;
    // size: maxBinSegs * CUDARF_BIN_SEG_SIZE * (S32 triIdx)
    void *binSegData;
    // size: maxBinSegs * (S32 segIdx), -1 = none
    void *binSegNext;
    // size: maxBinSegs * (S32 numEntries)
    void *binSegCount;
};

} // namespace rast (first block)

// ---------------------------------------------------------------------------
// DrawPacket — host/device buffer descriptor, defined between the two rast
// blocks because it depends on rast::VertexIn and is in turn needed by
// rast::PipeParams (by-value array member).
// ---------------------------------------------------------------------------

struct DrawPacket
{
    cudarf::PrimitiveIndex *dev_bufIdx = nullptr; ///< borrowed device index buffer
    rast::VertexIn *dev_bufVertex = nullptr;      ///< borrowed device vertex buffer
    rast::VertexIn *stagingBufVertex = nullptr;   ///< borrowed host-pinned staging buffer

    int index_count = 0;
    int vertCount = 0;
    int indexCapacity = 0;
    int vertexCapacity = 0;
};

// ---------------------------------------------------------------------------
// cudarf::rast (second block) — pipeline buffer structs that reference
// DrawPacket and the shared host types above
// ---------------------------------------------------------------------------

namespace rast
{

/// Internal GPU buffers for tiled rasterization, sized by primitive count.
///
struct PipeInternalBufferSet
{
    std::size_t maxVertexCount = 0;
    std::size_t maxIndexCount = 0;

    VertexOut *dev_bufVertexOut = NULL;   // vertex shader output
    Triangle  *dev_triangles   = NULL;   // triangle setup output
    void      *dev_tri_subtris;

    int32_t maxBinSegs = 0;
    void *dev_binSegData  = NULL;   // maxBinSegs * CUDARF_BIN_SEG_SIZE * S32
    void *dev_binSegNext  = NULL;   // maxBinSegs * S32 segIdx, -1 = none
    void *dev_binSegCount = NULL;   // maxBinSegs * S32 numEntries

    int32_t *dev_dbgbuf = NULL;
};

/// Persistent render-target and tiling state. These fields change only when
/// the rasterizer context or render-target dimensions are recreated.
struct PipeStaticContext {
    int32_t windowWidth;
    int32_t windowHeight;
    int2    rasterizerSize;   // multiple of 128 for current impl

    int     clockRate;
};

/// Per-frame state shared by all submissions in a frame.
struct PipeFrameContext {
    cudarf::CommonUniforms common;

#ifdef WITH_TAA
    struct {
        cudarf::CommonUniforms commonHist;
        cudarf::Uniforms *uniformsHist;
        cudarf::Velocity *velocityTex;
        float velocityThreshold;
    } taa;
#endif

    float               exposure;
    cudarf::CUDARFLight lights[CUDARF_MAX_LIGHTS];
    int32_t             lightCount;
    float3              camera;
    glm::mat4           sphericalHarmonics;
    cudaTextureObject_t brdfLUT = 0;
    cudaTextureObject_t specular = 0;
    unsigned int        specularMipCount = 0;
};

/// Per-draw/submission state: render flags, active draw packets, materials,
/// and batch-local geometry counts.
struct PipeSubmissionContext {
    bool withFaceCulling;
    bool withBlending;
    bool withDepthWriting;
    bool withDepthTest;

    int32_t  numTriangles;
    uint32_t maxSubtris;

    cudarf::DrawPacket drawPackets[CUDARF_MAX_DRAW_PACKETS];
    cudarf::Uniforms *uniforms;

    unsigned int drawPacketOrder[CUDARF_DRAW_PACKET_BATCH_LIMIT];
    unsigned int vtxOffsets[CUDARF_DRAW_PACKET_BATCH_LIMIT];   // prefix sum of vertex counts
    unsigned int drawPacketCount;

    unsigned int idxOffsets[CUDARF_DRAW_PACKET_BATCH_LIMIT];   // prefix sum of index counts
    unsigned int drawPacketMaterials[CUDARF_MAX_DRAW_PACKETS];
    bool         drawPacketDoubleSided[CUDARF_MAX_DRAW_PACKETS];

    cudarf::Material    materials[CUDARF_MAX_MATERIALS];
};

/// Submission scratch buffers and queue state. These are persistent
/// allocations in `pipe::Ctx`, but their active capacities/contents are
/// submission-local.
struct PipeScratchContext {
    VertexOut *vertexOut;

    Triangle *tris;           // indices_count/3 * sizeof(Triangle)
    uint8_t  *triSubtris;

    BinTilerCtx binCtx;

    SimpleQueue::Segment *tileQHeaders = NULL;
    int32_t *tileQData = NULL;
    unsigned int tileQLimit;

    int     *dbgbuf;
    int     *dbgbins;
    int32_t *dbgtiles;
};

/// Rasterization pipeline GPU-side parameter block. Assembled on the host and
/// uploaded to device memory before each draw call. The lifetime-specific base
/// structs keep the current flat field access intact while making the next
/// refactor step explicit: persist or upload these groups independently.
///
/// All pointers point to device memory unless specified otherwise.
///

struct PipeParams {
    const PipeStaticContext *stat;
    const PipeFrameContext *frame;
    const PipeSubmissionContext *submission;
    PipeScratchContext scratch;
};

} // namespace rast (second block)

} // namespace cudarf

#endif
