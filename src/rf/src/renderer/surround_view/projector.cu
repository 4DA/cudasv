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

InputFrames Projector::load_rgb(
    std::array<uint8_t *, SURROUND_VIEW_MAX_CAMERAS> rgb,
    unsigned int frame_set,
    int w,
    int h,
    int byte_stride)
{
    unsigned int srcPitchBytes = byte_stride;
    uint8_t alpha = 255;

    for (int i = 0; i < 4; i++) {
        // pad data to RGBA
        std::vector<uchar4> rgba((size_t)w*h);

        for (int y=0; y<h; ++y) {
            const uint8_t* s = rgb[i] + y*srcPitchBytes;
            uchar4* d = rgba.data() + (size_t)y*w;
            for (int x=0; x<w; ++x) {
                const uint8_t r = s[3*x+0];
                const uint8_t g = s[3*x+1];
                const uint8_t b = s[3*x+2];
                d[x] = make_uchar4(r,g,b,alpha);
            }
        }

        if (cuda_arrays[frame_set][i] == 0)
        {
            auto desc = cudaCreateChannelDesc<uchar4>();
            CUDA_CHK(cudaMallocArray(&cuda_arrays[frame_set][i], &desc, w, h));
            fflush(stdout);
            assert(cuda_arrays[frame_set][i] != 0);
        }

        CUDA_CHK(cudaMemcpy2DToArray(
                     cuda_arrays[frame_set][i], 0, 0,
                     rgba.data(),     /* src ptr */
                     (size_t)w * 4,   /* src pitch (bytes). use your real stride if different */
                     (size_t)w * 4,   /* width in bytes */
                     h,               /* rows */
                     cudaMemcpyHostToDevice));

        // Build texture object
        cudaResourceDesc res{};
        res.resType = cudaResourceTypeArray;
        res.res.array.array = cuda_arrays[0][i];

        cudaTextureDesc tex{};
        tex.addressMode[0]   = cudaAddressModeClamp;
        tex.addressMode[1]   = cudaAddressModeClamp;
        tex.filterMode       = cudaFilterModeLinear;         // bilinear
        tex.readMode         = cudaReadModeNormalizedFloat;  // u8 -> [0..1]
        tex.normalizedCoords = 1;                            // use [0,1] UVs

        CUDA_CHK(cudaCreateTextureObject(&cuda_textures[frame_set][i],
                                         &res,
                                         &tex,
                                         nullptr));
    }

    tex_width = w;
    tex_height = h;

    return cuda_textures[0];
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

__global__ void copy_to_surface(cudaSurfaceObject_t outputSurfObj, int w, int h, uint8_t* yuv420_plane)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < w && y < h) {
        uchar4 value;

        uint8_t* y_plane = yuv420_plane;
        uint8_t* u_plane = yuv420_plane + w * h;
        uint8_t* v_plane = u_plane + (w * h / 4);

        unsigned char yy = y_plane[y * w + x];
        uint16_t *uv_plane = (uint16_t *)(yuv420_plane + w * h);
        uint16_t uuvv = uv_plane[(y / 2) * (w / 2) + (x / 2)];
        unsigned char vv = uuvv >> 8;
        unsigned char uu = uuvv & 0xFF;

        // yuv_2_rgba
        int R = (float)yy + 1.402f   * ((float)vv - 128);
        int G = (float)yy - 0.34414f * ((float)uu - 128) - 0.71414f * ((float)vv - 128);
        int B = (float)yy + 1.772f   * ((float)uu - 128);

        if(R < 0) { R = 0; }
        if(G < 0) { G = 0; }
        if(B < 0) { B = 0; }

        if(R > 255) { R = 255; }
        if(G > 255) { G = 255; }
        if(B > 255) { B = 255; }

        value.x = R;
        value.y = G;
        value.z = B;
        value.w = 0xFF;

        surf2Dwrite(value, outputSurfObj, x * 4, y);
    }
}

__global__ void rgba_to_yuv_kernel(int w, int h, uint8_t* rgba_buffer, uint8_t* yuv420_plane)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < w && y < h) {
        int ofs = (y * w + x) * 4;

        int r = rgba_buffer[ofs + 0];
        int g = rgba_buffer[ofs + 1];
        int b = rgba_buffer[ofs + 2];

        // yuv_2_rgba
        //	int R = yy + 1.402f   * (vv - 128);
        //	int G = yy - 0.34414f * (uu - 128) - 0.71414f * (vv - 128);
        //	int B = yy + 1.772f   * (uu - 128);

        int yy = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        int uu = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        int vv = ((112 * r - 94 * g - 18 * b + 128) >> 8 ) + 128;
        if (yy < 0) { yy = 0; }
        if (uu < 0) { uu = 0; }
        if (vv < 0) { vv = 0; }

        yuv420_plane[y * w + x] = yy;
        uint16_t *uv_plane = (uint16_t *)(yuv420_plane + w * h);
        uint16_t uuvv = (vv << 8) | (uu);
        uv_plane[(y / 2) * (w / 2) + (x / 2)] = uuvv;
    }
}

extern "C" void copy_surface(cudaSurfaceObject_t surf, int w, int h, uint8_t* yuv420_plane)
{
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((w - 1) / blockSize2d.x + 1, (h - 1) / blockSize2d.y + 1);

    copy_to_surface<<<blockCount2d, blockSize2d>>>(surf, w, h, yuv420_plane);
}

extern "C" void rasterize_convert_rgba_to_yuv(uint32_t w, uint32_t h,
                                uint8_t* rgba_buffer,
                                uint8_t* yuv420_plane)
{
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((w - 1) / blockSize2d.x + 1, (h - 1) / blockSize2d.y + 1);

    rgba_to_yuv_kernel<<<blockCount2d, blockSize2d>>>(w, h, rgba_buffer, yuv420_plane);
//    convert_rgba_to_yuv(tex_width, tex_height, rgba_buffer, yuv420_plane);
}


//
__global__ void copy_padded_yuv_to_surface_rgba(cudaSurfaceObject_t outputSurfObj, int w, int h, uint8_t* yuv420_plane)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    // TODO: these should come from intrinsics, right now we have hardcoded 2048x1280 source frame and 1936x1036 in config
    const uint yuv_w = 2048;
    const uint yuv_h = 1280;

    if (x < w && y < h) {
        uchar4 value;

        uint8_t* y_plane = yuv420_plane;
        uint8_t* u_plane = yuv420_plane + yuv_w * yuv_h;
        uint8_t* v_plane = u_plane + (yuv_w * yuv_h / 4);

        unsigned char yy = y_plane[y * yuv_w + x];
        uint16_t *uv_plane = (uint16_t *)(yuv420_plane + yuv_w * yuv_h);
        uint16_t uuvv = uv_plane[(y / 2) * (yuv_w / 2) + (x / 2)];
        unsigned char vv = uuvv >> 8;
        unsigned char uu = uuvv & 0xFF;

        // yuv_2_rgba
        int R = (float)yy + 1.402f   * ((float)vv - 128);
        int G = (float)yy - 0.34414f * ((float)uu - 128) - 0.71414f * ((float)vv - 128);
        int B = (float)yy + 1.772f   * ((float)uu - 128);

        if(R < 0) { R = 0; }
        if(G < 0) { G = 0; }
        if(B < 0) { B = 0; }

        if(R > 255) { R = 255; }
        if(G > 255) { G = 255; }
        if(B > 255) { B = 255; }

        value.x = R;
        value.y = G;
        value.z = B;
        value.w = 0xFF;

        surf2Dwrite(value, outputSurfObj, x * 4, y);
    }
}

__global__ void copy_padded_yuv_to_buffer_rgba(uchar4* buffer, int w, int h, uint8_t* yuv420_plane)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    // TODO: these should come from intrinsics, right now we have hardcoded 2048x1280 source frame and 1936x1036 in config
    const uint yuv_w = 2048;
    const uint yuv_h = 1280;

    if (x < w && y < h) {
        uchar4 value;

        uint8_t* y_plane = yuv420_plane;
        uint8_t* u_plane = yuv420_plane + yuv_w * yuv_h;
        uint8_t* v_plane = u_plane + (yuv_w * yuv_h / 4);

        unsigned char yy = y_plane[y * yuv_w + x];
        uint16_t *uv_plane = (uint16_t *)(yuv420_plane + yuv_w * yuv_h);
        uint16_t uuvv = uv_plane[(y / 2) * (yuv_w / 2) + (x / 2)];
        unsigned char vv = uuvv >> 8;
        unsigned char uu = uuvv & 0xFF;

        // yuv_2_rgba
	int R = (float)yy + 1.402f   * ((float)vv - 128);
	int G = (float)yy - 0.34414f * ((float)uu - 128) - 0.71414f * ((float)vv - 128);
	int B = (float)yy + 1.772f   * ((float)uu - 128);

	if(R < 0) { R = 0; }
	if(G < 0) { G = 0; }
	if(B < 0) { B = 0; }

	if(R > 255) { R = 255; }
	if(G > 255) { G = 255; }
	if(B > 255) { B = 255; }

        value.x = R;
        value.y = G;
        value.z = B;
        value.w = 0xFF;

        buffer[y * w + x] = value; // surf2Dwrite(value, outputSurfObj, x * 4, y);
    }
}

__global__ void copy_padded_yuv_to_buffer_rgba_pitched(uchar4* buffer, int pitch, int w, int h, uint8_t* yuv420_plane)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    // TODO: these should come from intrinsics, right now we have hardcoded 2048x1280 source frame and 1936x1036 in config
    const uint yuv_w = 2048;
    const uint yuv_h = 1280;

    if (x < w && y < h) {
        uchar4 value;

        uint8_t* y_plane = yuv420_plane;
        uint8_t* u_plane = yuv420_plane + yuv_w * yuv_h;
        uint8_t* v_plane = u_plane + (yuv_w * yuv_h / 4);

        unsigned char yy = y_plane[y * yuv_w + x];
        uint16_t *uv_plane = (uint16_t *)(yuv420_plane + yuv_w * yuv_h);
        uint16_t uuvv = uv_plane[(y / 2) * (yuv_w / 2) + (x / 2)];
        unsigned char vv = uuvv >> 8;
        unsigned char uu = uuvv & 0xFF;

        // yuv_2_rgba
        int R = (float)yy + 1.402f   * ((float)vv - 128);
        int G = (float)yy - 0.34414f * ((float)uu - 128) - 0.71414f * ((float)vv - 128);
        int B = (float)yy + 1.772f   * ((float)uu - 128);

        if(R < 0) { R = 0; }
        if(G < 0) { G = 0; }
        if(B < 0) { B = 0; }

        if(R > 255) { R = 255; }
        if(G > 255) { G = 255; }
        if(B > 255) { B = 255; }

        value.x = R;
        value.y = G;
        value.z = B;
        value.w = 0xFF;

        buffer[y * pitch + x] = value;
    }
}



// FMA-based lerp function
template <typename T> __host__ __device__ inline T lerpf(T v0, T v1, T t) { return fma(t, v1, fma(-t, v0, v0)); }


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

    for (int i = 0; i < 4; i++) {
        const auto& projection = cameraParams.projections[i];

        glm::vec3 pImage = project_world_point_to_fisheye_plane(
            pWorld, projection.world_to_camera, projection.distortion_coeffs);

        glm::vec3 uv = project_to_normalized_uv(
            pImage, projection.intrinsics, tex_w, tex_h);

        float4 rgba = tex2D<float4>(texPack.texObj[i], uv.x, uv.y);

        if (uv.z > 0) {
            cam_samples[i] = rgba;
        }
    }

    float4 blend_val = {0.0f};

    float VL2 = cameraParams.vehicle_length / 2.0f;
    float VW2 = cameraParams.vehicle_width / 2.0f;

    // front camera
    if (pWorld.x > 0.0f && pWorld.y <= VW2 && pWorld.y >= -VW2) {
        blend_val = cam_samples[CAMERA_FRONT];
    } // front-left overlap
    else if (pWorld.x > 0.0f && pWorld.y > 0.0f) {

        glm::vec2 C = glm::vec2(pWorld) - glm::vec2(VL2, VW2);

        blend_val = overlap(C,
                            glm::radians(35.0f), glm::radians(55.0f),
                            cam_samples[CAMERA_FRONT], cam_samples[CAMERA_LEFT]);
    } // right-front overlap
    else if (pWorld.x > 0.0f && pWorld.y < 0.0f) {
        glm::vec2 C = glm::vec2(pWorld) - glm::vec2(VL2, -VW2);

        blend_val = overlap(C,
                            glm::radians(305 - 10.0f), glm::radians(305 + 10.0f),
                            cam_samples[CAMERA_RIGHT], cam_samples[CAMERA_FRONT]);
    } // rear-right overlap
    else if (pWorld.x < 0.0f && pWorld.y < 0.0f) {
        glm::vec2 C = glm::vec2(pWorld) - glm::vec2(-VL2, -VW2);

        blend_val = overlap(C,
                            glm::radians(225 - 10.0f), glm::radians(225 + 10.0f),
                            cam_samples[CAMERA_REAR], cam_samples[CAMERA_RIGHT]);
    }
    // left-rear overlap
    else if (pWorld.x < 0.0f && pWorld.y > 0.0f) {
        glm::vec2 C = glm::vec2(pWorld) - glm::vec2(-VL2, VW2);

        blend_val = overlap(C,
                            glm::radians(135 - 10.0f), glm::radians(135 + 10.0f),
                            cam_samples[CAMERA_LEFT], cam_samples[CAMERA_REAR]);
    }

    // DEBUG: draw single camera
    // blend_val = cam_samples[CAMERA_FRONT];

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
