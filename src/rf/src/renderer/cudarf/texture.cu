#include <cassert>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/printf.h>

#include <rf/renderer/image.hpp>
#include <rf/renderer/cudarf/texture.hpp>
#include <rf/renderer/cudarf/cudarf.hpp>

#include "helpers.hpp"
#include "helpers_cudavec.inl"

__device__ uchar4 float4_to_uchar4(float4 v)
{
    v.x = fminf(fmaxf(v.x, 0.0f), 1.0f);
    v.y = fminf(fmaxf(v.y, 0.0f), 1.0f);
    v.z = fminf(fmaxf(v.z, 0.0f), 1.0f);
    v.w = fminf(fmaxf(v.w, 0.0f), 1.0f);

    return make_uchar4((unsigned char)lrintf(v.x * 255.0f),
                       (unsigned char)lrintf(v.y * 255.0f),
                       (unsigned char)lrintf(v.z * 255.0f),
                       (unsigned char)lrintf(v.w * 255.0f));
}

__global__ void mip_downsample2x(cudaTextureObject_t srcTex,
                                 cudaSurfaceObject_t output,
                                 unsigned int width,
                                 unsigned int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {return;}

    float tX = 2.0f * x + 0.5f;
    float tY = 2.0f * y + 0.5f;

    float4 v0 = tex2D<float4>(srcTex, tX,        tY);
    float4 v1 = tex2D<float4>(srcTex, tX + 1.0f, tY);
    float4 v2 = tex2D<float4>(srcTex, tX,        tY + 1.0f);
    float4 v3 = tex2D<float4>(srcTex, tX + 1.0f, tY + 1.0f);

    surf2Dwrite(float4_to_uchar4((v0 + v1 + v2 + v3) * 0.25f),
                output,
                x * sizeof(uchar4),
                y);
}

namespace
{
void generate_texture_mips(cudaMipmappedArray *dev_mipmapArray,
                           unsigned int width,
                           unsigned int height,
                           unsigned int channels,
                           unsigned int compSz,
                           unsigned int mipCount,
                           cudaStream_t cuStream)
{
    cudaArray_t dev_mipLevelArray0;
    cudaArray_t dev_mipLevelArray1;

    assert(channels == 4);
    assert(compSz == 1);

    for (unsigned int lvl = 1; lvl < mipCount; lvl++) {
        width  = std::max(width  / 2, 1u);
        height = std::max(height / 2, 1u);

        CUDA_CHK(cudaGetMipmappedArrayLevel(&dev_mipLevelArray0, dev_mipmapArray, lvl-1));
        CUDA_CHK(cudaGetMipmappedArrayLevel(&dev_mipLevelArray1, dev_mipmapArray, lvl));

        cudaTextureObject_t srcTex;
        cudaSurfaceObject_t dstSurf;

        cudarf::create_array_texture(srcTex,
                                     dev_mipLevelArray0,
                                     cudaAddressModeClamp,
                                     true);

        cudarf::create_array_surface(dstSurf, dev_mipLevelArray1);

        dim3 blockSize(32, 32);
        dim3 blockCount((width - 1)  / blockSize.x + 1,
                        (height - 1) / blockSize.y + 1);

        mip_downsample2x<<<blockCount, blockSize, 0, cuStream>>>(srcTex, dstSurf, width, height);

        CUDA_CHK_ERROR("mip_downsample2x");

        cudaDestroySurfaceObject(dstSurf);
        cudaDestroyTextureObject(srcTex);
    }
}
}

std::optional<cudarf::Texture> cudarf::create_cuda_texture(rf::Image image,
                                                           cudaTextureAddressMode addressMode,
                                                           unsigned int mipLevels,
                                                           cudaStream_t cuStream)
{
    assert(mipLevels > 0);
    if (mipLevels == 0) {return std::nullopt;}

    cudaTextureObject_t tex;

    unsigned int compSz;
    unsigned bitsR, bitsG, bitsB, bitsA;

    switch (image.pixel_type) {
    case rf::PixelChannelType::U8:
        bitsR = bitsG = bitsB = bitsA = 8;
        compSz = 1;
        break;
    case rf::PixelChannelType::U16:
        bitsR = bitsG = bitsB = bitsA = 16;
        compSz = 2;
        break;
    default:
        SPDLOG_ERROR("unsupported pixelformat: {}",
                     static_cast<int>(image.pixel_type));
        assert(false);
        return std::nullopt;
    }

    switch(image.channels) {
    case 1:
        bitsG = 0; bitsB = 0; bitsA = 0;
        break;
    case 2:
        bitsB = 0; bitsA = 0;
        break;
    case 3:
        SPDLOG_ERROR("unsupported channels count: {}", image.channels);
        bitsA = 0;
    case 4:
        break;
    default:
        SPDLOG_ERROR("{}", fmt::sprintf("unsupported channel count: %d", image.channels));
        return std::nullopt;
    }

    // Allocate array and copy image data
    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(bitsR, bitsG, bitsB, bitsA, cudaChannelFormatKindUnsigned);


    cudaArray *cuArray = nullptr;
    cudaMipmappedArray *dev_mipmapArray = nullptr;
    cudaResourceDesc texRes;
    cudaTextureDesc texDescr;

    memset(&texRes, 0, sizeof(cudaResourceDesc));
    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    if (mipLevels == 1) {
        CUDA_CHK(cudaMallocArray(&cuArray, &channelDesc, image.w, image.h));
        CUDA_CHK(cudaMemcpy2DToArrayAsync(cuArray,                           // source
                                          0, 0,                              // offsets
                                          image.data,                        // source ptr
                                          image.w * image.channels * compSz, // spitch
                                          image.w * image.channels * compSz, // width
                                          image.h,                           // height
                                          cudaMemcpyHostToDevice,            // destination
                                          cuStream));

        texRes.resType            = cudaResourceTypeArray;
        texRes.res.array.array    = cuArray;

        texDescr.normalizedCoords = true;
        texDescr.filterMode       = cudaFilterModeLinear;
        texDescr.addressMode[0]   = addressMode;
        texDescr.addressMode[1]   = addressMode;
        texDescr.readMode         = cudaReadModeNormalizedFloat;
    } else {
        assert(image.pixel_type == rf::PixelChannelType::U8);
        assert(image.channels == 4);

        CUDA_CHK(cudaMallocMipmappedArray(&dev_mipmapArray,
                                          &channelDesc,
                                          make_cudaExtent(image.w, image.h, 0),
                                          mipLevels,
                                          cudaArraySurfaceLoadStore));

        cudaArray_t dev_mipLevelArray;

        CUDA_CHK(cudaGetMipmappedArrayLevel(&dev_mipLevelArray, dev_mipmapArray, 0));

        CUDA_CHK(cudaMemcpy2DToArrayAsync(dev_mipLevelArray,                 // source
                                          0, 0,                              // offsets
                                          image.data,                        // source ptr
                                          image.w * image.channels * compSz, // spitch
                                          image.w * image.channels * compSz, // width
                                          image.h,                           // height
                                          cudaMemcpyHostToDevice,            // destination
                                          cuStream));

        CUDA_CHK(cudaStreamSynchronize(cuStream));

        generate_texture_mips(dev_mipmapArray, image.w, image.h, image.channels, compSz, mipLevels, cuStream);

        texRes.resType               = cudaResourceTypeMipmappedArray;
        texRes.res.mipmap.mipmap     = dev_mipmapArray;

        texDescr.normalizedCoords    = true;
        texDescr.filterMode          = cudaFilterModeLinear;
        texDescr.mipmapFilterMode    = cudaFilterModeLinear;
        texDescr.addressMode[0]      = addressMode;
        texDescr.addressMode[1]      = addressMode;
        texDescr.addressMode[2]      = cudaAddressModeClamp;
        texDescr.readMode            = cudaReadModeNormalizedFloat;
        texDescr.maxMipmapLevelClamp = float(mipLevels - 1);
    }
    CUDA_CHK(cudaCreateTextureObject(&tex, &texRes, &texDescr, NULL));

    cudarf::Texture result(tex,
                           false,
                           glm::mat3(1.0f),
                           image.channels,
                           image.w,
                           image.h);

    result.dev_array = cuArray;
    result.dev_mipmapArray = dev_mipmapArray;
    result.mipLevels = mipLevels;

    return result;
}
