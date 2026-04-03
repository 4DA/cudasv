#ifndef RF_RENDERER_SURROUND_VIEW_PROJECTOR_HPP
#define RF_RENDERER_SURROUND_VIEW_PROJECTOR_HPP

#include <array>
#include <cstdint>

#include <cuda_runtime.h>

#include <rf/renderer/cudarf/cudarf.hpp>
#include <rf/renderer/glm_common.hpp>

#define SURROUND_VIEW_MAX_CAMERAS 4
#define SURROUND_VIEW_TEXTURE_SETS 2

namespace rf::surround_view
{

enum DebugMode : uint32_t
{
    DEBUG_MODE_NORMAL = 0,
    DEBUG_MODE_CAMERA_ROLES = 1,
    DEBUG_MODE_COVERAGE_MASK = 2,
    DEBUG_MODE_REPROJECTION_GRID = 3,
};

struct CameraProjectionParams {
    glm::mat4 world_to_camera;
    glm::mat3 intrinsics;
    float distortion_coeffs[8];
};

using InputFrames =
    std::array<cudaTextureObject_t, SURROUND_VIEW_MAX_CAMERAS>;

using InputFrameArrays =
    std::array<cudaArray_t, SURROUND_VIEW_MAX_CAMERAS>;

struct Projector {
    int tex_width = 0;
    int tex_height = 0;
    int active_texture_set = 0;
    uint64_t prevFrameSeq = ~uint64_t{0};

    std::array<InputFrames, SURROUND_VIEW_TEXTURE_SETS> cuda_textures;
    std::array<InputFrameArrays, SURROUND_VIEW_TEXTURE_SETS> cuda_arrays = {0};

    InputFrames load_rgb(
        std::array<uint8_t *, SURROUND_VIEW_MAX_CAMERAS> rgb,
        unsigned int frame_set,
        int w,
        int h,
        int byte_stride);
};

struct RigParams {
    CameraProjectionParams projections[SURROUND_VIEW_MAX_CAMERAS];
    float front_extent;
    float rear_extent;
    float left_extent;
    float right_extent;
    float front_left_start_rad;
    float front_left_end_rad;
    float right_front_start_rad;
    float right_front_end_rad;
    float rear_right_start_rad;
    float rear_right_end_rad;
    float left_rear_start_rad;
    float left_rear_end_rad;
    uint32_t debug_mode = DEBUG_MODE_NORMAL;
    uint32_t debug_camera_index = 2;
};

struct TexturePack {
    cudaTextureObject_t texObj[SURROUND_VIEW_MAX_CAMERAS];
};

struct ViewParams {
    glm::vec3 origin;
    static constexpr float ipd = 100.0f;
    glm::vec3 horizontal;
    glm::vec3 vertical;
    glm::vec3 lower_left;
};

void project_async(Projector *ctx,
                   int width,
                   int height,
                   ViewParams view,
                   RigParams params,
                   cudarf::Framebuffer dev_framebuffer,
                   cudaStream_t stream);

void project_async(std::array<cudaTextureObject_t, SURROUND_VIEW_MAX_CAMERAS> cuda_textures,
                   int width,
                   int height,
                   int texture_width,
                   int texture_height,
                   ViewParams view,
                   RigParams params,
                   cudarf::Framebuffer dev_framebuffer,
                   cudaStream_t stream);

} // namespace rf::surround_view

#endif
