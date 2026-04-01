#include <cstdio>
#include <cstring>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/printf.h>

#include <rf/renderer/glm_common.hpp>
#include "renderer/cudarf/helpers.hpp"

#include "utils.hpp"

using namespace rf;

void rf::log_duration(const std::string &message, const Duration& duration, unsigned int indent)
{
    const unsigned int PAD_MAX = 64;
    char padding[PAD_MAX];

    if (indent < PAD_MAX-1) {
        memset(padding, ' ', indent);
        padding[indent] = '\0';
    }

    SPDLOG_INFO("{}", fmt::sprintf("CPU TIME(ms) %s%s: %.3lf", padding, message.c_str(), duration.count()));
}

cudaTextureObject_t rf::create_cuda_texture(rf::Image image,
                                            cudaTextureAddressMode addressMode,
                                            cudaStream_t cuStream)
{
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
        return 0;
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
        return 0;
    }

    // Allocate array and copy image data
    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(bitsR, bitsG, bitsB, bitsA, cudaChannelFormatKindUnsigned);

    cudaArray *cuArray;
    CUDA_CHK(cudaMallocArray(&cuArray, &channelDesc, image.w, image.h));
    CUDA_CHK(cudaMemcpy2DToArrayAsync(cuArray,                           // source
                                      0, 0,                              // offsets
                                      image.data,                        // source ptr
                                      image.w * image.channels * compSz, // spitch
                                      image.w * image.channels * compSz, // width
                                      image.h,                           // height
                                      cudaMemcpyHostToDevice,            // destination
                                      cuStream));

    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = cuArray;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = true;
    texDescr.filterMode       = cudaFilterModeLinear;
    texDescr.addressMode[0]   = addressMode;
    texDescr.addressMode[1]   = addressMode;

    texDescr.readMode = cudaReadModeNormalizedFloat;

    CUDA_CHK(cudaCreateTextureObject(&tex, &texRes, &texDescr, NULL));

    return tex;
}
