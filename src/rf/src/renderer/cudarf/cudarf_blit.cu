// cudarf_blit.cu — framebuffer clear, copy, resample, and compose operations.
// These functions operate on Framebuffer / FBTexture / DepthValue directly and
// have no dependency on the rasterization pipeline internals.

#include <cuda_runtime.h>

#include <rf/renderer/cudarf/cudarf.hpp>
#include "types.hpp"
#include "helpers_cudavec.inl"
#include "framebuffer.inl"
#include "color_space.inl"

// Minimal local helpers — avoids pulling in the full raster/material stack.
static __device__ __inline__ cudarf::Color make_color(const cudarf::ColorRGB &rgb, float a)
{
    return make_float4(rgb.x, rgb.y, rgb.z, a);
}
static __device__ __inline__ cudarf::Color make_color(float r, float g, float b, float a)
{
    return make_float4(r, g, b, a);
}
static __device__ __inline__ cudarf::ColorRGB to_rgb(const cudarf::Vec4f &src)
{
    return make_float3(src.x, src.y, src.z);
}

using namespace cudarf;
using namespace cudarf::rast;

// ---------------------------------------------------------------------------
// Clear kernels
// ---------------------------------------------------------------------------

__global__
static void init_depth(DepthValue *image, int w, int h)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x < w && y < h) {
        image[x + y * w] = 1.0f;
    }
}

__global__
static void init_framebuffer(cudarf::Framebuffer fb, int w, int h, cudarf::ColorN color)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x < w && y < h) {
        fb::store(fb, x, y, color);
    }
}

__global__
static void init_framebuffer_checkers(cudarf::Framebuffer fb, int w, int h)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x < w && y < h) {
        if (x % 30 < 15 || y % 30 < 15) {
            fb::store(fb, x, y, make_uchar4(50, 50, 50, 255));
        } else {
            fb::store(fb, x, y, make_uchar4(0, 80, 155, 255));
        }
    }
}

// ---------------------------------------------------------------------------
// Compose kernels
// ---------------------------------------------------------------------------

__global__
static void compose(cudarf::Framebuffer lower,
                    cudarf::Framebuffer upper,
                    cudarf::Framebuffer overlay,
                    float exposure,
                    int w, int h,
                    float fadeMinY, float fadeMaxY,
                    uchar4 *out)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x >= w || y >= h) { return; }

    cudarf::Color lower_col, upper_col, overlay_col;
    fb::load(lower, x, y, lower_col);

    if (fadeMinY >= 0.0f && fadeMaxY >= 0.0f) {
        float v = (y + 0.5f) / h;
        if (v >= fadeMinY) {
            float fade = 1.0f - clamp((v - fadeMinY) / (fadeMaxY - fadeMinY), 0.0f, 1.0f);
            lower_col.x *= fade;
            lower_col.y *= fade;
            lower_col.z *= fade;
        }
    }

    fb::load(upper, x, y, upper_col);
    upper_col = cudarf::shading::tone_map(upper_col, exposure);

    fb::load(overlay, x, y, overlay_col);
    overlay_col = cudarf::shading::tone_map(overlay_col, exposure);

    float4 base = make_color(
        upper_col.w * to_rgb(upper_col) + (1.0f - upper_col.w) * to_rgb(lower_col), 1.0f);
    float4 color = make_color(
        to_rgb(overlay_col) + (1.0f - overlay_col.w) * to_rgb(base), 1.0f);

    out[x + (h - 1 - y) * w] = fb::to_rgba_norm(color);

}

__global__
static void compose_no_tonemap(cudarf::Framebuffer lower,
                                cudarf::Framebuffer upper,
                                cudarf::Framebuffer overlay,
                                int w, int h,
                                float fadeMinY, float fadeMaxY,
                                uchar4 *out)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x >= w || y >= h) { return; }

    cudarf::Color lower_col, upper_col, overlay_col;
    fb::load(lower, x, y, lower_col);

    if (fadeMinY >= 0.0f && fadeMaxY >= 0.0f) {
        float v = (y + 0.5f) / h;
        if (v >= fadeMinY) {
            float fade = 1.0f - clamp((v - fadeMinY) / (fadeMaxY - fadeMinY), 0.0f, 1.0f);
            lower_col.x *= fade;
            lower_col.y *= fade;
            lower_col.z *= fade;
        }
    }

    fb::load(upper, x, y, upper_col);
    fb::load(overlay, x, y, overlay_col);

    float4 base = make_color(
        upper_col.w * to_rgb(upper_col) + (1.0f - upper_col.w) * to_rgb(lower_col), 1.0f);
    float4 color = make_color(
        to_rgb(overlay_col) + (1.0f - overlay_col.w) * to_rgb(base), 1.0f);

    out[x + (h - 1 - y) * w] = fb::to_rgba_norm(color);

}

__global__
static void compose_single(cudarf::Framebuffer fb, int w, int h, uchar4 *out)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x >= w || y >= h) { return; }

    cudarf::ColorN color;
    fb::load(fb, x, y, color);
    out[x + (h - 1 - y) * w] = color;
}

// ---------------------------------------------------------------------------
// Copy / resample kernels
// ---------------------------------------------------------------------------

__global__
static void copy_depth_to_pbo(DepthValue *image, int w, int h, uchar4 *pbo)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + y * w;
    if (x < w && y < h) {
        float depth = image[index];
        pbo[index].w = 0;
        pbo[index].x = static_cast<int>(-depth * 255.0);
        pbo[index].y = static_cast<int>(-depth * 255.0);
        pbo[index].z = static_cast<int>(-depth * 255.0);
        if (depth <= -1) {
            pbo[index].x = 0;
            pbo[index].y = 255;
            pbo[index].z = 0;
        }
    }
}

__global__
static void copy_to_pbo(cudarf::Framebuffer fb, int w, int h, uchar4 *pbo)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x < w && y < h) {
        uchar4 pix;
        fb::load(fb, x, y, pix);
        pbo[x + y * w] = pix;
    }
}

__global__
static void copy_framebuffer_kernel(cudarf::Framebuffer src,
                                    cudarf::Framebuffer dst,
                                    int w,
                                    int h)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x < w && y < h) {
        cudarf::ColorN pix;
        fb::load(src, x, y, pix);
        fb::store(dst, x, y, pix);
    }
}

__global__
void resample_bilinear(cudarf::FBTexture tex, int width, int height, uchar4 *out)
{
    assert(tex);
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) { return; }

    float u = (x + 0.5f) / (float)width;
    float v = (y + 0.5f) / (float)height;

    float4 texval = fb::tex_sample_4f32(tex, u, v);
    out[y * width + x] = fb::to_rgba_norm(texval);

}

// ---------------------------------------------------------------------------
// Host launchers — clear
// ---------------------------------------------------------------------------

void cudarf::pipe::clear_framebuffer(cudarf::pipe::Ctx *desc, cudarf::Framebuffer fb,
                                     cudarf::ColorN color, cudaStream_t cuStream)
{
    dim3 blockSize2d(8, 8);
    dim3 blockCount2d((desc->width  - 1) / blockSize2d.x + 1,
                      (desc->height - 1) / blockSize2d.y + 1);
    init_framebuffer<<<blockCount2d, blockSize2d, 0, cuStream>>>(fb, desc->width, desc->height, color);
    CUDA_CHK_ERROR("clear_framebuffer");
}

void cudarf::pipe::copy_framebuffer(cudarf::pipe::Ctx *desc,
                                    cudarf::Framebuffer src,
                                    cudarf::Framebuffer dst,
                                    cudaStream_t cuStream)
{
    dim3 blockSize2d(8, 8);
    dim3 blockCount2d((desc->width - 1) / blockSize2d.x + 1,
                      (desc->height - 1) / blockSize2d.y + 1);
    copy_framebuffer_kernel<<<blockCount2d, blockSize2d, 0, cuStream>>>(
        src, dst, desc->width, desc->height);
    CUDA_CHK_ERROR("copy_framebuffer");
}

void cudarf::pipe::generate_checkers(cudarf::pipe::Ctx *desc, cudarf::Framebuffer fb,
                                     cudaStream_t cuStream)
{
    dim3 blockSize2d(8, 8);
    dim3 blockCount2d((desc->width  - 1) / blockSize2d.x + 1,
                      (desc->height - 1) / blockSize2d.y + 1);
    init_framebuffer_checkers<<<blockCount2d, blockSize2d, 0, cuStream>>>(fb, desc->width, desc->height);
    CUDA_CHK_ERROR("generate_checkers");
}

void cudarf::pipe::clear_depth(cudarf::pipe::Ctx *desc, cudaStream_t stream)
{
    dim3 blockSize2d(8, 8);
    dim3 blockCount2d((desc->width  - 1) / blockSize2d.x + 1,
                      (desc->height - 1) / blockSize2d.y + 1);
    init_depth<<<blockCount2d, blockSize2d, 0, stream>>>(desc->dev_depthbuffer.get(),
                                                         desc->width, desc->height);
    CUDA_CHK_ERROR("clear_depth");
}

// ---------------------------------------------------------------------------
// Host launchers — copy / resample
// ---------------------------------------------------------------------------

void cudarf::resample(cudarf::FBTexture src, int outputWidth, int outputHeight, uchar4 *pbo)
{
    assert(src);
    dim3 blockSize2d(8, 8);
    dim3 blockCount2d((outputWidth  - 1) / blockSize2d.x + 1,
                      (outputHeight - 1) / blockSize2d.y + 1);
    resample_bilinear<<<blockCount2d, blockSize2d>>>(src, outputWidth, outputHeight, pbo);
    CUDA_CHK_ERROR("resample");
}

void cudarf::pipe::copy_to_pbo(cudarf::pipe::Ctx *desc, cudarf::Framebuffer src, uchar4 *pbo)
{
    dim3 blockSize2d(8, 8);
    dim3 blockCount2d((desc->width  - 1) / blockSize2d.x + 1,
                      (desc->height - 1) / blockSize2d.y + 1);
    copy_to_pbo<<<blockCount2d, blockSize2d>>>(src, desc->width, desc->height, pbo);
    CUDA_CHK_ERROR("copy_to_pbo");
}

void cudarf::pipe::copy_depth_to_pbo(cudarf::pipe::Ctx *desc, uchar4 *pbo)
{
    dim3 blockSize2d(8, 8);
    dim3 blockCount2d((desc->width  - 1) / blockSize2d.x + 1,
                      (desc->height - 1) / blockSize2d.y + 1);
    copy_depth_to_pbo<<<blockCount2d, blockSize2d>>>(desc->dev_depthbuffer.get(),
                                                     desc->width, desc->height, pbo);
    CUDA_CHK_ERROR("copy_depth_to_pbo");
}

// ---------------------------------------------------------------------------
// Host launchers — compose
// ---------------------------------------------------------------------------

void cudarf::compose(cudarf::Framebuffer lower,
                     cudarf::Framebuffer upper,
                     cudarf::Framebuffer overlay,
                     float exposure,
                     unsigned int width,
                     unsigned int height,
                     float fadeMinY, float fadeMaxY,
                     uchar4 *dev_out, cudaStream_t cuStream)
{
    assert(lower);
    assert(upper);
    assert(overlay);
    assert(dev_out);
    dim3 blockSize2d(8, 8);
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
                      (height - 1) / blockSize2d.y + 1);
    compose<<<blockCount2d, blockSize2d, 0, cuStream>>>(
        lower, upper, overlay, exposure, width, height, fadeMinY, fadeMaxY, dev_out);
    CUDA_CHK_ERROR("compose");
}

void cudarf::compose(cudarf::Framebuffer lower,
                     cudarf::Framebuffer upper,
                     cudarf::Framebuffer overlay,
                     unsigned int width,
                     unsigned int height,
                     float fadeMinY, float fadeMaxY,
                     uchar4 *dev_out, cudaStream_t cuStream)
{
    assert(lower);
    assert(upper);
    assert(overlay);
    assert(dev_out);
    dim3 blockSize2d(8, 8);
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
                      (height - 1) / blockSize2d.y + 1);
    compose_no_tonemap<<<blockCount2d, blockSize2d, 0, cuStream>>>(
        lower, upper, overlay, width, height, fadeMinY, fadeMaxY, dev_out);
    CUDA_CHK_ERROR("compose");
}

void cudarf::compose(cudarf::Framebuffer fb,
                     unsigned int width,
                     unsigned int height,
                     uchar4 *dev_out, cudaStream_t cuStream)
{
    assert(fb);
    assert(dev_out);
    dim3 blockSize2d(8, 8);
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
                      (height - 1) / blockSize2d.y + 1);
    compose_single<<<blockCount2d, blockSize2d, 0, cuStream>>>(fb, width, height, dev_out);
    CUDA_CHK_ERROR("compose");
}
