// -*- mode: c++ -*-

#ifndef _KERNELS_CU_
#define _KERNELS_CU_

#include <algorithm>

#include <cuda_runtime.h>

//#include "cuda_utils.cu"

#include <rf/renderer/glm_common.hpp>
#include <rf/renderer/cudarf/cudarf.hpp>
#include "cuda_helpers.hpp"


#include <ctype.h>
#include <stdio.h>

using namespace cudarf;

__global__
void init_depth_kernel(DepthValue *image, int w, int h) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
        image[index] = -1.0f;
    }
}

__global__
void clear_pbo_vec4(glm::vec4 *image, int w, int h, glm::vec4 color) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < w && y < h) {
        int index = x + (y * w);
        image[index] = color;
    }
}

__global__
void clear_pbo_rgba8(cudarf::ColorN *image, int w, int h, cudarf::ColorN color) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < w && y < h) {
        int index = x + (y * w);
        image[index] = color;
    }
}

__global__
void copy_to_pbo(Color *image, int w, int h, uchar4 *pbo) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
        glm::vec4 color;
        color.x = glm::clamp(image[index].x, 0.0f, 1.0f) * 255.0;
        color.y = glm::clamp(image[index].y, 0.0f, 1.0f) * 255.0;
        color.z = glm::clamp(image[index].z, 0.0f, 1.0f) * 255.0;
        color.w = glm::clamp(image[index].w, 0.0f, 1.0f) * 255.0;

        pbo[index].x = static_cast<unsigned char>(color.x);
        pbo[index].y = static_cast<unsigned char>(color.y);
        pbo[index].z = static_cast<unsigned char>(color.z);
        pbo[index].w = static_cast<unsigned char>(color.w);
    }
}

__global__
void copy_to_pbo__surface(Color *image, int w, int h, cudaSurfaceObject_t outputSurface) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
        glm::vec4 color;

        color.x = glm::clamp(image[index].x, 0.0f, 1.0f) * 255.0;
        color.y = glm::clamp(image[index].y, 0.0f, 1.0f) * 255.0;
        color.z = glm::clamp(image[index].z, 0.0f, 1.0f) * 255.0;
        color.w = glm::clamp(image[index].w, 0.0f, 1.0f) * 255.0;

        uchar4 value;
        value.x = color.x;
        value.y = color.y;
        value.z = color.z;
        value.w = color.w;
        surf2Dwrite(value, outputSurface, x * 4, h - y - 1);
    }
}

__global__
void copy_to_pbo__buffer(Color *image, int w, int h, uchar4* buffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
        glm::vec4 color;

        color.x = glm::clamp(image[index].x, 0.0f, 1.0f) * 255.0;
        color.y = glm::clamp(image[index].y, 0.0f, 1.0f) * 255.0;
        color.z = glm::clamp(image[index].z, 0.0f, 1.0f) * 255.0;
        color.w = glm::clamp(image[index].w, 0.0f, 1.0f) * 255.0;

        uchar4 value;
        value.x = color.x;
        value.y = color.y;
        value.z = color.z;
        value.w = color.w;
        buffer[(h - 1 - y) * ((w + 0x3F) & ~(0x3F)) + x] = value;
//        surf2Dwrite(value, outputSurface, x * 4, h - y - 1);
    }
}

__global__
void copy_to_pbo__buffer(ColorN *image, int w, int h, uchar4* buffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
        buffer[(h - 1 - y) * ((w + 0x3F) & ~(0x3F)) + x] = image[index];
    }
}

extern "C"
void rasterize_copy_output_rgba_surface__async(Color* image, int w, int h, cudaSurfaceObject_t outputSurface, cudaStream_t stream)
{
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((w - 1) / blockSize2d.x + 1,
                      (h - 1) / blockSize2d.y + 1);

    copy_to_pbo__surface<<<blockCount2d, blockSize2d, 0, stream>>>(image, w, h, outputSurface);
    CUDA_CHK_KERNEL(stream, "copy_to_pbo__surface");
}

extern "C" void rasterize_copy_output_rgba_buffer_async(cudarf::ColorN* image, int w, int h, void* buffer, cudaStream_t stream)
{
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((w - 1) / blockSize2d.x + 1,
                      (h - 1) / blockSize2d.y + 1);

    copy_to_pbo__buffer<<<blockCount2d, blockSize2d, 0, stream>>>(image, w, h, (uchar4 *)buffer);
    CUDA_CHK_KERNEL(stream, "copy_to_pbo__buffer");
}

extern "C"
void rasterize_clear_output_rgba__async(cudarf::ColorN *image, int w, int h, float* color, cudaStream_t stream)
{
    auto clearColor = make_uchar4(0, 0, 0, 255);

    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((w - 1) / blockSize2d.x + 1,
                      (h - 1) / blockSize2d.y + 1);

    clear_pbo_rgba8<<<blockCount2d, blockSize2d, 0, stream>>>(image, w, h, clearColor);
    CUDA_CHK_KERNEL(stream, "clear_pbo_vec4");
}

void clear_depth__async(cudarf::pipe::Ctx *desc, cudaStream_t stream)
{
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((desc->width  - 1) / blockSize2d.x + 1,
                      (desc->height - 1) / blockSize2d.y + 1);

    init_depth_kernel<<<blockCount2d, blockSize2d,0, stream>>>(
                        desc->dev_depthbuffer.get(),
                        desc->width, desc->height);

    CUDA_CHK_KERNEL(stream, "init_depth");
}

#endif
