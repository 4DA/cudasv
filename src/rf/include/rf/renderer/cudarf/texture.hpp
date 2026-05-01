#ifndef CUDARF_TEXTURE
#define CUDARF_TEXTURE

#include <cuda_runtime.h>
#include <rf/renderer/glm_common.hpp>
#include <rf/renderer/image.hpp>

namespace cudarf
{
struct Texture {
    cudaTextureObject_t textureObject;
    bool hasUVTransform;
    glm::mat3 uvTransform;
    unsigned int channels;

    int mipLevels;
    cudaArray *dev_array;
    cudaMipmappedArray *dev_mipmappedArray;

    __device__ __host__ Texture(cudaTextureObject_t textureObject,
                                bool hasUVTransform,
                                glm::mat3 uvTransform,
                                unsigned int channels):
        textureObject(textureObject),
        hasUVTransform(hasUVTransform),
        uvTransform(uvTransform),
        channels(channels)
        {}

    __device__ __host__ Texture():
        textureObject(0),
        hasUVTransform(false),
        uvTransform(1.0f),
        channels(0)
        {}
};

cudaTextureObject_t create_cuda_texture(rf::Image image,
                                        cudaTextureAddressMode addressMode,
                                        int mipLevels,
                                        cudaStream_t cuStream);
}
#endif
