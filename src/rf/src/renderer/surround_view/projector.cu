#include <cuda_runtime.h>

#include <rf/renderer/surround_view/projector.hpp>
#include <rf/renderer/cuda_helpers.hpp>

#include <rf/renderer/cudarf/cudarf.hpp>

#include "../cudarf/helpers_cudavec.inl"
#include "../cudarf/framebuffer.inl"

#include <ctype.h>
#include <stdio.h>

using namespace cudarf;

using uint = unsigned int;
using ushort = unsigned short;

namespace rf::surround_view
{

// TODO: pass these values from CMake configuration (ce_view_mesh)
static const uint MESH_CS_LOG2 = 4u;
static const uint SCV_CS_LOG2  = 4u;

inline static int clamp_int(int x, int xmin, int xmax) { return (x < xmin) ? xmin : ((x > xmax) ? xmax : x); }

uint32_t yuv_2_rgba(uint8_t yy, uint8_t uu, uint8_t vv)
{
	int R = clamp_int((float)yy + 1.402f   * ((float)vv - 128), 0, 255);
	int G = clamp_int((float)yy - 0.34414f * ((float)uu - 128) - 0.71414f * ((float)vv - 128), 0, 255);
	int B = clamp_int((float)yy + 1.772f   * ((float)uu - 128), 0, 255);

	return ((R) | (G << 8) | (B << 16));
}

uint32_t get_texel(uint8_t* tex, int tw, int th, int x, int y)
{
	if (x < 0) x = 0;
	if (y < 0) y = 0;
	if (y >= th) y = th - 1;
	if (x >= tw) x = tw - 1;

	uint32_t ofs = y * tw + x;
	uint32_t ofs_uv = (y >> 1) * (tw >> 1) + (x >> 1);
	uint8_t  cc = tex[ofs];

	uint8_t uu = *(tex + tw * th + ofs_uv);
	uint8_t vv = *(tex + tw * th + ((tw * th) / 4) + ofs_uv);
	return yuv_2_rgba(cc, uu, vv);
}

void convert_yuv_to_rgba(uint8_t* in_tex, uint8_t* in_tex_rgba, int TW, int TH, int pitch)
{
    for (int y = 0 ; y < TH ; y++)
    {
        for (int x = 0 ; x < TW ; x++)
        {
            int ofs = y * pitch + x;
            uint32_t color = get_texel(in_tex, TW, TH, x, y);
            ((uint32_t*)in_tex_rgba)[ofs] = color;
        }
    }
}

__global__ void copy_loaded_texture(
                    uchar4* buffer,
                    int pitch,
                    int w, int h, uchar4* source)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    // TODO: these should come from intrinsics, right now we have hardcoded 2048x1280 source frame and 1936x1036 in config
    const uint yuv_w = 2048;
    const uint yuv_h = 1280;

    if (x < w && y < h)
    {
        buffer[y * pitch + x] = source[y * pitch + x];
    }
}

__global__ void rgb_to_rgba_surface(
    const uint8_t *rgb,
    int width,
    int height,
    int pitchBytes,
    cudaSurfaceObject_t out)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const uint8_t *p = rgb + y * pitchBytes + x * 3;
    uchar4 rgba = make_uchar4(p[0], p[1], p[2], 255);

    surf2Dwrite(rgba, out, x * sizeof(uchar4), y);
}

InputFrames Projector::load_rgb(
    std::array<uint8_t *, SURROUND_VIEW_MAX_CAMERAS> rgb,
    unsigned int fs, // frame set
    int w,
    int h,
    int pitch,
    cudaStream_t cuStream)
{
    for (int i = 0; i < 4; i++) {
        if (cuda_arrays[fs][i] == 0) {
            // create RGB GPU staging buffer
            CUDA_CHK(cudaMalloc(&cuda_frames_staging[i], pitch * h));

            // create array-surface-texture triplet for camera frames
            auto desc = cudaCreateChannelDesc<uchar4>();
            CUDA_CHK(cudaMallocArray(&cuda_arrays[fs][i], &desc, w, h, cudaArraySurfaceLoadStore));

            cudarf::create_array_texture(cuda_textures[fs][i],
                                         cuda_arrays[fs][i],
                                         cudaAddressModeClamp,
                                         false);

            cudarf::create_array_surface(cuda_surfaces[fs][i], cuda_arrays[fs][i]);
        }

        assert(cuda_frames_staging[i]);

        // upload frame CPU data to GPU RGB staing buffer
        // --
        CUDA_CHK(cudaMemcpy(cuda_frames_staging[i],
                            rgb[i],
                            pitch * h,
                            cudaMemcpyHostToDevice));

        // convert rgb -> rgba on GPU
        // --
        dim3 blockSize(16, 16);
        dim3 gridSize((w-1) / 16 + 1, (h-1) / 16 + 1);

        rgb_to_rgba_surface<<<gridSize, blockSize, 0, cuStream>>>(
            cuda_frames_staging[i], w, h, pitch, cuda_surfaces[fs][i]);

        CUDA_CHK_ERROR("rgb_to_rgba_surface");
    }

    tex_width = w;
    tex_height = h;

    return cuda_textures[fs];
}


#if 0
// This kernel generates a test pattern with uniform color gradient

__global__ void generate_rgba_surf(cudaSurfaceObject_t outputSurfObj, int w, int h)
{
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < w && y < h) {
        unsigned char yy = x ^ y;
        unsigned char uu = ((float)x / (float)w) * (255.0f);
        unsigned char vv = ((float)y / (float)h) * (255.0f);

        // yuv_2_rgba
        uchar4 value;
        value.x = (float)yy + 1.402f   * ((float)vv - 128);
        value.y = (float)yy - 0.34414f * ((float)uu - 128) - 0.71414f * ((float)vv - 128);
        value.z = (float)yy + 1.772f   * ((float)uu - 128);
        value.w = 0xFF;
        surf2Dwrite/*<uint4>*/(value, outputSurfObj, x * 4, y);
    }
}

extern "C" void generate_rgba_surface(cudaSurfaceObject_t surf, int w, int h)
{
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((w - 1) / blockSize2d.x + 1, (h - 1) / blockSize2d.y + 1);

    generate_rgba_surf<<<blockCount2d, blockSize2d>>>(surf, w, h);
}

// Method to generate test (RG gradient) pattern
__global__ void copy_to_surface_rgba(cudaSurfaceObject_t outputSurfObj, int w, int h, uint8_t* rgba_plane)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < w && y < h) {
        uchar4 value;

        int ofs = y * w + x;
        unsigned char R = rgba_plane[ofs * 4 + 0];
        unsigned char G = rgba_plane[ofs * 4 + 1];
        unsigned char B = rgba_plane[ofs * 4 + 2];

        value.x = (float)x / (float)w; // R;
        value.y = (float)y / (float)h; // G;
        value.z = 0.f;                 // B;
        value.w = 0xFF;

        surf2Dwrite(value, outputSurfObj, x * 4, y);
    }
}

extern "C" void copy_surface_rgba(cudaSurfaceObject_t surf, int w, int h, uint8_t* rgba_plane)
{
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((w - 1) / blockSize2d.x + 1, (h - 1) / blockSize2d.y + 1);

    copy_to_surface_rgba<<<blockCount2d, blockSize2d>>>(surf, w, h, rgba_plane);
}
#endif

__device__
glm::vec3 project_to_normalized_uv(glm::vec3 projected_coords,
                                   glm::mat3 intrinsics,
                                   float imageWidth,
                                   float imageHeight)
{
    glm::vec3 coord;

    float skew = intrinsics[1][0];
    float fx = intrinsics[0][0];
    float fy = intrinsics[1][1];
    float cx = intrinsics[2][0];
    float cy = intrinsics[2][1];

    coord.x = (projected_coords.x + skew * projected_coords.y) * fx + cx;
    coord.y = projected_coords.y * fy + cy;

    coord.x /= imageWidth;
    coord.y /= imageHeight;

    coord.z = 1.0;

    if (coord.x < 0.0 ||
        coord.x > 1.0 ||
        coord.y < 0.0 ||
        coord.y > 1.0 ||
        projected_coords.z < 0.0)
    {
        coord.z = 0.0;
    }
    return coord;
}

const float EPSILON = 1e-10;

__device__ float evaluate_theta_polynomial_fisheye_scale(float radialDistance,
                                                         const float *distortionCoeffs,
                                                         float depth)
{
    float theta1 = std::atan(radialDistance / depth);

    float theta2 = theta1 * theta1,
          theta3 = theta2 * theta1,
          theta4 = theta2 * theta2,
          theta5 = theta3 * theta2,
          theta7 = theta5 * theta2,
          theta9 = theta5 * theta4;

    float tg = theta1 +
        distortionCoeffs[0] * theta3 +
        distortionCoeffs[1] * theta5 +
        distortionCoeffs[2] * theta7 +
        distortionCoeffs[3] * theta9;

    return std::abs(radialDistance) < EPSILON ? 1.0f : (tg / radialDistance);
}

__device__ glm::vec3 project_world_point_to_fisheye_plane(glm::vec3 worldPoint,
                                                          glm::mat4 worldToCamera,
                                                          const float *distortionCoeffs)
{
    glm::vec3 cameraPoint = glm::vec3(worldToCamera * glm::vec4(worldPoint, 1.0f));
    float scale = evaluate_theta_polynomial_fisheye_scale(
        glm::length(glm::vec2(cameraPoint)),
        distortionCoeffs,
        cameraPoint.z);
    return glm::vec3(scale * glm::vec2(cameraPoint), cameraPoint.z);
}

// Ray o + t d vs. cylinder x^2+y^2 = R^2 with z >= 0.
// If hitCap=true, the bottom disk at z=0 is included.
// Returns nearest t > 0. normalOut is the outward unit normal.
__device__
inline bool intersectSemiInfCylinderZ(
    const glm::vec3& o,               // ray origin
    const glm::vec3& d,               // ray direction (need not be unit)
    float R,
    bool hitCap,
    float& tHit,
    glm::vec3* normalOut = nullptr)
{
    const float eps = 1e-8f;
    bool  hit = false;
    float bestT = INFINITY;
    glm::vec3 bestN(0);

    // --- Lateral surface (solve in XY plane) ---
    const float A = d.x*d.x + d.y*d.y;
    const float B = 2.0f * (o.x*d.x + o.y*d.y);
    const float C = o.x*o.x + o.y*o.y - R*R;

    if (A > eps) {
        const float disc = B*B - 4.0f*A*C;
        if (disc >= 0.0f) {
            const float sqrtD = std::sqrt(disc);
            const float inv2A = 0.5f / A;
            float t0 = (-B - sqrtD) * inv2A;
            float t1 = (-B + sqrtD) * inv2A;
            if (t0 > t1) {
                float temp = t0;
                t0 = t1;
                t1 = temp;
            }

            // Check the smaller positive root first, then the other.
            for (float t : {t0, t1}) {
                if (t > eps) {
                    float z = o.z + t * d.z;
                    if (z >= 0.0f - 1e-6f) { // accept tiny numerical slop
                        if (t < bestT) {
                            glm::vec3 p = o + t * d;
                            glm::vec3 n = glm::normalize(glm::vec3(p.x, p.y, 0.0f)); // outward radial
                            bestT = t; bestN = n; hit = true;
                        }
                        break; // for lateral surface, first valid root is nearest
                    }
                }
            }
        }
    }
    // else: ray parallel to axis -> no lateral hit

    // --- Bottom cap at z=0 (optional) ---
    if (hitCap && std::fabs(d.z) > eps) {
        float tCap = (0.0f - o.z) / d.z;
        if (tCap > eps) {
            glm::vec3 p = o + tCap * d;
            if (p.x*p.x + p.y*p.y <= R*R + 1e-6f) {
                if (tCap < bestT) {
                    bestT = tCap;
                    // Outward normal of the bottom face (outside of the cylinder): -Z
                    bestN = glm::vec3(0, 0, -1);
                    hit = true;
                }
            }
        }
    }

    if (!hit) return false;
    tHit = bestT;
    if (normalOut) *normalOut = bestN;
    return true;
}

enum CameraID
{
    CAMERA_RIGHT = 0,
    CAMERA_LEFT,
    CAMERA_FRONT,
    CAMERA_REAR
};


__device__ float4 overlap(glm::vec2 p,
                          float a1,
                          float a2,
                          cudarf::Color s1,
                          cudarf::Color s2)
{
    float angle = std::fmod(2.0f * glm::pi<float>() + std::atan2(p.y, p.x),
                            2.0f * glm::pi<float>());

    float4 blend_val;

    if (angle < a1) {
        blend_val = s1;
    }
    else if (angle > a2) {
        blend_val = s2;
    }
    else {
        float t = (angle - a1) / (a2 - a1);
        blend_val = my::lerp(s1, s2, t);
    }

    return blend_val;
}

__device__ float4 debug_role_color(int cameraIndex)
{
    switch (cameraIndex) {
    case CAMERA_FRONT:
        return make_float4(1.0f, 0.0f, 0.0f, 1.0f);
    case CAMERA_LEFT:
        return make_float4(0.0f, 1.0f, 0.0f, 1.0f);
    case CAMERA_RIGHT:
        return make_float4(0.0f, 0.0f, 1.0f, 1.0f);
    case CAMERA_REAR:
        return make_float4(1.0f, 1.0f, 0.0f, 1.0f);
    default:
        return make_float4(1.0f, 0.0f, 1.0f, 1.0f);
    }
}

__device__ float4 debug_coverage_color(int visibleCount)
{
    switch (visibleCount) {
    case 0:
        return make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    case 1:
        return make_float4(0.0f, 0.5f, 1.0f, 1.0f);
    case 2:
        return make_float4(1.0f, 0.75f, 0.0f, 1.0f);
    case 3:
        return make_float4(1.0f, 0.0f, 1.0f, 1.0f);
    default:
        return make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    }
}

__device__ float4 apply_reprojection_grid(float4 rgba, glm::vec3 uv)
{
    const float gridCells = 12.0f;
    const float lineWidth = 0.04f;
    const float gridU = uv.x * gridCells - floorf(uv.x * gridCells);
    const float gridV = uv.y * gridCells - floorf(uv.y * gridCells);
    const bool onGrid = (gridU < lineWidth) || (gridU > (1.0f - lineWidth)) ||
                        (gridV < lineWidth) || (gridV > (1.0f - lineWidth));

    if (!onGrid) {
        return rgba;
    }

    return make_float4(1.0f, 1.0f, 1.0f, 1.0f);
}

__global__
void render_mesh_gpu(int w,
                     int h,
                     TexturePack texPack,
                     int tex_w,
                     int tex_h,
                     ViewParams virtCamera,
                     RigParams cameraParams,
                     cudarf::Framebuffer framebuffer)
{
    assert(tex_w > 0);
    assert(tex_h > 0);
    assert(w > 0);
    assert(h > 0);

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) {return;}

    glm::vec2 screen_ray(x / (float) w, y / (float) h);

    glm::vec3 dir = virtCamera.lower_left + screen_ray.x * virtCamera.horizontal
        + screen_ray.y * virtCamera.vertical - virtCamera.origin;

    float t;
    glm::vec3 N;
    bool hit = intersectSemiInfCylinderZ(
        virtCamera.origin, dir, 10000.0f, true, t, &N);

    if (!hit) {
        fb::store(framebuffer, x,  y, make_float4(0.0f, 1.0f, 0.0f, 1.0f));
        return;
    }

    glm::vec3 pWorld = virtCamera.origin + t * dir;

    float4 cam_samples[4] = {0.0f};
    glm::vec3 cam_uvs[4] = {glm::vec3(0.0f)};
    bool camera_visible[4] = {false, false, false, false};

    for (int i = 0; i < 4; i++) {
        const auto& projection = cameraParams.projections[i];

        glm::vec3 pImage = project_world_point_to_fisheye_plane(
            pWorld, projection.world_to_camera, projection.distortion_coeffs);

        glm::vec3 uv = project_to_normalized_uv(
            pImage, projection.intrinsics, tex_w, tex_h);

        float4 rgba = tex2D<float4>(texPack.texObj[i], uv.x, uv.y);

        if (uv.z > 0) {
            cam_samples[i] = rgba;
            cam_uvs[i] = uv;
            camera_visible[i] = true;
        }
    }

    float4 blend_val = {0.0f};
    int visibleCameraCount = 0;
    for (int i = 0; i < 4; ++i) {
        visibleCameraCount += camera_visible[i] ? 1 : 0;
    }

    const float frontExtent = cameraParams.front_extent;
    const float rearExtent = cameraParams.rear_extent;
    const float leftExtent = cameraParams.left_extent;
    const float rightExtent = cameraParams.right_extent;

    // front camera
    if (pWorld.x > 0.0f && pWorld.y <= leftExtent && pWorld.y >= -rightExtent) {
        blend_val = cam_samples[CAMERA_FRONT];
    } // front-left overlap
    else if (pWorld.x > 0.0f && pWorld.y > 0.0f) {

        glm::vec2 C = glm::vec2(pWorld) - glm::vec2(frontExtent, leftExtent);

        blend_val = overlap(C,
                            cameraParams.front_left_start_rad, cameraParams.front_left_end_rad,
                            cam_samples[CAMERA_FRONT], cam_samples[CAMERA_LEFT]);
    } // right-front overlap
    else if (pWorld.x > 0.0f && pWorld.y < 0.0f) {
        glm::vec2 C = glm::vec2(pWorld) - glm::vec2(frontExtent, -rightExtent);

        blend_val = overlap(C,
                            cameraParams.right_front_start_rad, cameraParams.right_front_end_rad,
                            cam_samples[CAMERA_RIGHT], cam_samples[CAMERA_FRONT]);
    } // rear-right overlap
    else if (pWorld.x < 0.0f && pWorld.y < 0.0f) {
        glm::vec2 C = glm::vec2(pWorld) - glm::vec2(-rearExtent, -rightExtent);

        blend_val = overlap(C,
                            cameraParams.rear_right_start_rad, cameraParams.rear_right_end_rad,
                            cam_samples[CAMERA_REAR], cam_samples[CAMERA_RIGHT]);
    }
    // left-rear overlap
    else if (pWorld.x < 0.0f && pWorld.y > 0.0f) {
        glm::vec2 C = glm::vec2(pWorld) - glm::vec2(-rearExtent, leftExtent);

        blend_val = overlap(C,
                            cameraParams.left_rear_start_rad, cameraParams.left_rear_end_rad,
                            cam_samples[CAMERA_LEFT], cam_samples[CAMERA_REAR]);
    }

    // DEBUG: draw single camera
    // blend_val = cam_samples[CAMERA_FRONT];

    if (cameraParams.debug_mode == DEBUG_MODE_CAMERA_ROLES) {
        if (pWorld.x > 0.0f && pWorld.y <= leftExtent && pWorld.y >= -rightExtent) {
            blend_val = debug_role_color(CAMERA_FRONT);
        } else if (pWorld.x > 0.0f && pWorld.y > 0.0f) {
            glm::vec2 C = glm::vec2(pWorld) - glm::vec2(frontExtent, leftExtent);
            blend_val = overlap(C,
                                cameraParams.front_left_start_rad, cameraParams.front_left_end_rad,
                                debug_role_color(CAMERA_FRONT), debug_role_color(CAMERA_LEFT));
        } else if (pWorld.x > 0.0f && pWorld.y < 0.0f) {
            glm::vec2 C = glm::vec2(pWorld) - glm::vec2(frontExtent, -rightExtent);
            blend_val = overlap(C,
                                cameraParams.right_front_start_rad, cameraParams.right_front_end_rad,
                                debug_role_color(CAMERA_RIGHT), debug_role_color(CAMERA_FRONT));
        } else if (pWorld.x < 0.0f && pWorld.y < 0.0f) {
            glm::vec2 C = glm::vec2(pWorld) - glm::vec2(-rearExtent, -rightExtent);
            blend_val = overlap(C,
                                cameraParams.rear_right_start_rad, cameraParams.rear_right_end_rad,
                                debug_role_color(CAMERA_REAR), debug_role_color(CAMERA_RIGHT));
        } else if (pWorld.x < 0.0f && pWorld.y > 0.0f) {
            glm::vec2 C = glm::vec2(pWorld) - glm::vec2(-rearExtent, leftExtent);
            blend_val = overlap(C,
                                cameraParams.left_rear_start_rad, cameraParams.left_rear_end_rad,
                                debug_role_color(CAMERA_LEFT), debug_role_color(CAMERA_REAR));
        }
    } else if (cameraParams.debug_mode == DEBUG_MODE_COVERAGE_MASK) {
        blend_val = debug_coverage_color(visibleCameraCount);
    } else if (cameraParams.debug_mode == DEBUG_MODE_REPROJECTION_GRID) {
        const uint32_t cameraIndex = cameraParams.debug_camera_index;
        if (cameraIndex < 4 && camera_visible[cameraIndex]) {
            blend_val = apply_reprojection_grid(cam_samples[cameraIndex], cam_uvs[cameraIndex]);
        } else {
            blend_val = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
        }
    }

    fb::store(framebuffer, x,  y, blend_val);
}

void project_async(Projector* ctx,
                   int width,
                   int height,
                   ViewParams virtCamera,
                   RigParams params,
                   cudarf::Framebuffer dev_framebuffer,
                   cudaStream_t stream)
{
    dim3 threadDim(16, 16, 1);
    dim3 blockDim(width / threadDim.x, height / threadDim.y, 1);

    TexturePack texObj = {
        ctx->cuda_textures[ctx->active_texture_set][0],
        ctx->cuda_textures[ctx->active_texture_set][1],
        ctx->cuda_textures[ctx->active_texture_set][2],
        ctx->cuda_textures[ctx->active_texture_set][3],
    };

    render_mesh_gpu <<<blockDim, threadDim, 0, stream>>> (
        width, height, texObj,
        ctx->tex_width, ctx->tex_height, virtCamera, params, dev_framebuffer);

    CUDA_CHK_ERROR("surround_view::project_async kernel");
}

void project_async(std::array<cudaTextureObject_t, SURROUND_VIEW_MAX_CAMERAS> cuda_textures,
                   int width,
                   int height,
                   int texture_width,
                   int texture_height,
                   ViewParams virtCamera,
                   RigParams params,
                   cudarf::Framebuffer dev_framebuffer,
                   cudaStream_t stream)
{
    dim3 threadDim(16, 16, 1);
    dim3 blockDim(width / threadDim.x, height / threadDim.y, 1);

    TexturePack texObj = {
        cuda_textures[0],
        cuda_textures[1],
        cuda_textures[2],
        cuda_textures[3],
    };

    render_mesh_gpu <<<blockDim, threadDim, 0, stream>>> (
        width, height, texObj,
        texture_width, texture_height, virtCamera, params, dev_framebuffer);

    CUDA_CHK_ERROR("surround_view::project_async kernel");
}

} // namespace rf::surround_view
