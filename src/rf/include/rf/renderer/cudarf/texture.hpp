#ifndef CUDARF_TEXTURE
#define CUDARF_TEXTURE

#include <cuda_runtime.h>
#include <rf/renderer/glm_common.hpp>
#include <rf/renderer/image.hpp>

namespace cudarf
{
struct Texture {
    cudaTextureObject_t textureObject;
    unsigned int channels;
    unsigned int width;
    unsigned int height;

    int mipLevels;
    cudaArray *dev_array;
    cudaMipmappedArray *dev_mipmapArray;

    bool hasUVTransform;
    glm::mat3 uvTransform;

    __device__ __host__ Texture(cudaTextureObject_t textureObject,
                                bool hasUVTransform,
                                glm::mat3 uvTransform,
                                unsigned int channels,
                                unsigned int width,
                                unsigned int height):
        textureObject(textureObject),
        channels(channels),
        width(width),
        height(height),
        mipLevels(1),
        dev_array(nullptr),
        dev_mipmapArray(nullptr),
        hasUVTransform(hasUVTransform),
        uvTransform(uvTransform)
        {}

    __device__ __host__ Texture():
        textureObject(0),
        channels(0),
        width(0),
        height(0),
        mipLevels(0),
        dev_array(nullptr),
        dev_mipmapArray(nullptr),
        hasUVTransform(false),
        uvTransform(1.0f)
        {}
};

std::optional<Texture> create_cuda_texture(rf::Image image,
                                           cudaTextureAddressMode addressMode,
                                           unsigned int mipLevels,
                                           cudaStream_t cuStream);
}
#endif
