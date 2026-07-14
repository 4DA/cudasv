#ifndef CUDARF_HPP
#define CUDARF_HPP

#include <array>
#include <vector>
#include <memory>
#include <unordered_map>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>

#include <rf/renderer/cudarf/cudarf_rast.hpp>
#include <rf/renderer/cudarf/cudarf_profile.hpp>
#include <rf/renderer/cudarf/array_surface.hpp>
#include <rf/renderer/cudarf/memory.hpp>
#include <rf/renderer/mesh_geometry.hpp>
#include <rf/renderer/virtual_camera.hpp>
#include <rf/camera_control/viewpoint_animation.hpp>

namespace cudarf
{

// ---------------------------------------------------------------------------
// Framebuffer / surface types
// Framebuffer: read/write individual pixels without filtering
// FBTexture:   sample-only with interpolation
// ---------------------------------------------------------------------------

struct LinearSurface
{
    cudarf::ColorN *devPtr = nullptr;
    cudaArray *texArray = nullptr;
    unsigned int w = 0;
    unsigned int h = 0;

    __host__ __device__ __inline__
    operator bool() const
    {
        return (devPtr != nullptr && w != 0 && h != 0);
    }
};

#if defined(WITH_TAA)
using Framebuffer = cudaSurfaceObject_t;
#else
using Framebuffer = LinearSurface;
#endif

enum class TAA_Pattern {
    Center,
    Dirty,
    Helix,
    Halton,
};

struct CudaStreams {
    cudaStream_t rendering;
    // cudaStream_t uploadStream;
    // cudaStream_t downloadStream;
};

struct PBRParams
{
    float3 camera;
    float exposure;
    std::vector<CUDARFLight> lights;
    glm::mat4 sphericalHarmonics;
    cudaTextureObject_t brdfLUT;
    CubeMap specular;
};


struct RenderParams {
    bool face_culling;
    bool with_blending;
};

struct ProjectionParams {
    float near;
    float far;
    float fovY;
};

namespace pipe
{

using cudarf::memory::DeviceBuffer;

struct Atomics {
    uint32_t subtris_count;  // triangle setup
    uint32_t null_tris;

    uint32_t bin_counter;    // bin tiler
    uint32_t numBinSegs;

    uint32_t coarseCounter;  // coarse tiler

    uint32_t dbg_oft;        // misc

    struct {
        // total number of visible fragments
        uint32_t totalVisibleFrags;
    } visibuf;
};

struct Ctx
{
    int width;
    int height;
    int SMPCount;
    int clockRate;           // SM clock frequency in kilohertz
    bool TAAEnabled;

    DeviceBuffer<cudarf::rast::PipeStaticContext> dev_pipeStatic;
    DeviceBuffer<cudarf::rast::PipeFrameContext> dev_pipeFrame;
    DeviceBuffer<cudarf::rast::PipeSubmissionContext> dev_pipeSubmission;
    DeviceBuffer<cudarf::rast::PipeParams> dev_pipeParams;

    DeviceBuffer<Atomics> dev_pipeAtomics;

    cudarf::DrawPacket drawPackets[CUDARF_MAX_DRAW_PACKETS];
    DeviceBuffer<cudarf::PrimitiveIndex> drawPacketIdxBuffers[CUDARF_MAX_DRAW_PACKETS];
    DeviceBuffer<cudarf::rast::VertexIn> drawPacketVertexBuffers[CUDARF_MAX_DRAW_PACKETS];

    cudarf::rast::PipeInternalBufferSet internalBufs;

    unsigned int drawPacketCount = 1;  // zero draw packet is for dynamic geometry

    DeviceBuffer<cudarf::Uniforms> dev_uniforms;
#ifdef WITH_TAA
    DeviceBuffer<cudarf::Uniforms> dev_uniformsHist;
#endif

    DeviceBuffer<int32_t> dev_binFirstSeg;
    DeviceBuffer<int32_t> dev_binTotal;

    unsigned int tileQLimit;
    DeviceBuffer<cudarf::rast::SimpleQueue::Segment> dev_tileQHeaders;
    DeviceBuffer<int32_t> dev_tileQData;

    DeviceBuffer<cudarf::DepthValue> dev_depthbuffer;
    DeviceBuffer<cudarf::visibuf::GeomOutput> dev_geom_output;
    DeviceBuffer<cudarf::visibuf::XYCommand> dev_xyCommands;

#ifdef WITH_TAA
    std::array<cudarf::memory::ArraySurfaceTexture, 2> framebufferResources;
    cudarf::memory::ArraySurfaceTexture rasterResource;
    cudarf::memory::ArraySurface uiFramebufferResource;

    cudarf::Framebuffer dev_framebuffer[2] = {0};
    cudarf::FBTexture dev_framebufferTex[2] = {0};

    DeviceBuffer<cudarf::Velocity> dev_velocityTex;

    struct TAA_Params {
        float scale = 1.0f;
        float feedback = 0.8f;
        float velocityThreshold = 0.1f;
        cudarf::TAA_Pattern pattern = cudarf::TAA_Pattern::Helix;
        std::vector<float2> pointsHalton;
    } TAA;
#else
    cudarf::Framebuffer dev_framebuffer;
    cudarf::FBTexture dev_framebufferTex;
    cudarf::Framebuffer uiFramebuffer;
#endif

    Ctx(int screenWidth,
        int screenHeight,
        int tileQLimit,
        int SMPCount,
        bool TAAEnabled,
        int clockRate,
        const cudaStream_t &cuStream);

    ~Ctx();
};


struct LaunchConfig
{
    bool withTexturing;
    bool withOpaqueVisibuf;
    unsigned int frameCounter;

    cudarf::Framebuffer nativeOutput;  // if set, used as output instead of internal fb

    std::shared_ptr<cudarf::profiling::Events> eventDB;
    bool testBinTiler;

    LaunchConfig(bool withTexturing,
                 bool withOpaqueVisibuf,
                 unsigned int frameCounter,
                 cudarf::Framebuffer nativeOutput,
                 std::shared_ptr<cudarf::profiling::Events> eventDB = nullptr,
                 bool testBinTiler = false):
        withTexturing(withTexturing),
        withOpaqueVisibuf(withOpaqueVisibuf),
        frameCounter(frameCounter),
        nativeOutput(nativeOutput),
        eventDB(eventDB),
        testBinTiler(testBinTiler)
        {}
};

void clear_framebuffer(Ctx *desc, cudarf::Framebuffer fb, cudarf::ColorN color, cudaStream_t cuStream);

void copy_framebuffer(Ctx *desc,
                      cudarf::Framebuffer src,
                      cudarf::Framebuffer dst,
                      cudaStream_t cuStream);

void clear_depth(Ctx *desc, cudaStream_t stream);

void copy_to_pbo(Ctx *desc, cudarf::Framebuffer src, uchar4 *pbo);

void copy_depth_to_pbo(Ctx *desc, uchar4 *pbo);

unsigned int alloc_draw_packet(Ctx *desc);

void set_draw_packet_buffers(Ctx *desc, const int *bufIdx,
                             int index_count, const cudarf::Vec3f *bufPos,
                             int vertCount, const cudarf::Vec4f *bufCol,
                             const cudarf::Vec3f *bufNor = nullptr,
                             const cudarf::Vec2f *bufTex = nullptr,
                             const Vec4f *bufTan = nullptr,
                             unsigned int drawPacketId = CUDARF_SCRATCH_DRAW_PACKET,
                             const cudaStream_t &cuStream = 0);

void set_draw_packet_buffers(Ctx *desc,
                             const rf::GltfMesh &mesh,
                             unsigned int drawPacketId = CUDARF_SCRATCH_DRAW_PACKET,
                             const cudaStream_t &cuStream = 0);

void draw_triangles(Ctx *desc,
                    const std::vector<glm::vec3> &triangles,
                    const std::vector<glm::vec4> &colors,
                    const glm::mat4 &M,
                    const cudarf::CommonUniforms &common,
                    bool face_culling,
                    bool blending,
                    unsigned int frameCounter,
                    const cudaStream_t &cuStream = 0);

void run_pipe(Ctx *desc,
              const cudarf::RenderParams &params,
              const cudarf::Uniforms &uniforms,
#ifdef WITH_TAA
              const cudarf::Uniforms &uniformsHist,
#endif
              const std::vector<unsigned int> &drawPacketIds,
              const std::vector<unsigned int> &matIds,
              const cudarf::MaterialMap &materials,
              const LaunchConfig &launchConfig,
              const cudaStream_t &cuStream);

void run_pipe(Ctx *desc,
                    const cudarf::RenderParams &params,
                    const std::vector<cudarf::Uniforms> &uniforms,
#ifdef WITH_TAA
                    const std::vector<cudarf::Uniforms> &uniformsHist,
#endif
                    const std::vector<unsigned int> &drawPacketIds,
                    const std::vector<unsigned int> &matIds,
                    const cudarf::MaterialMap &materials,
                    const LaunchConfig &launchConfig,
                    const cudaStream_t &cuStream);

void begin_frame(cudarf::pipe::Ctx *desc,
                 const rf::VirtualCamera &camera,
                 const cudarf::PBRParams &pbr,
                 const CommonUniforms &commonHist,
                 unsigned int frameCounter,
                 cudaStream_t cuStream);

cudarf::Framebuffer get_output_fb(Ctx *desc, unsigned int frameCounter);
cudarf::FBTexture   get_output_tex(Ctx *desc, unsigned int frameCounter);

cudarf::Framebuffer get_internal_fb(Ctx *desc, unsigned int frameCounter);
cudarf::Framebuffer get_ui_fb(Ctx *desc);

#ifdef WITH_TAA
cudarf::Framebuffer get_history_fb(Ctx *desc, unsigned int frameCounter);
cudarf::FBTexture   get_history_tex(Ctx *desc, unsigned int frameCounter);

void TAA(Ctx *desc,
         const cudarf::CommonUniforms &frameUniforms,
         const cudarf::CommonUniforms &histUniforms,
         const cudarf::ProjectionParams &projection,
         unsigned int frameCounter,
         cudaStream_t cuStream);
#endif

void generate_checkers(Ctx *desc, cudarf::Framebuffer fb, cudaStream_t cuStream);

} // namespace pipe

cudarf::Uniforms make_uniforms(const glm::mat4 &P,
                               const glm::mat4 &V,
                               const glm::mat4 &M);

cudarf::CommonUniforms make_common(const glm::mat4 &P,
                                   const glm::mat4 &V);

void resample(cudarf::FBTexture src, int outputWidth, int outputHeight, uchar4 *pbo);

void compose(Framebuffer lower,
             Framebuffer upper,
             Framebuffer overlay,
             float exposure,
             unsigned int width, unsigned int height,
             float fadeMinY, float fadeMaxY,
             uchar4 *out, cudaStream_t cuStream);

void compose(Framebuffer lower,
             Framebuffer upper,
             Framebuffer overlay,
             unsigned int width, unsigned int height,
             float fadeMinY, float fadeMaxY,
             uchar4 *out, cudaStream_t cuStream);

void compose(cudarf::Framebuffer fb,
             unsigned int width, unsigned int height,
             uchar4 *dev_out, cudaStream_t cuStream);

void create_surface(cudarf::LinearSurface &outSurface,
                    cudaTextureObject_t &outTexture,
                    int width,
                    int height,
                    const cudaStream_t &cuStream);

void create_surface(cudarf::LinearSurface &outSurface,
                    int width,
                    int height,
                    const cudaStream_t &cuStream);

void free_surface(cudarf::LinearSurface &fb);

} // namespace cudarf

#endif
