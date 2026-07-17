#include <cstring>
#include <utility>

#include <rf/renderer/cudarf/array_surface.hpp>

#include <rf/renderer/cuda_helpers.hpp>

namespace cudarf
{
namespace memory
{

ArraySurface::ArraySurface(unsigned int width,
                           unsigned int height,
                           cudaChannelFormatDesc channelDesc) noexcept
{
    CUDA_CHK(cudaMallocArray(&_array, &channelDesc, width, height, cudaArraySurfaceLoadStore));

    cudaResourceDesc surfRes;
    std::memset(&surfRes, 0, sizeof(cudaResourceDesc));
    surfRes.resType = cudaResourceTypeArray;
    surfRes.res.array.array = _array;

    CUDA_CHK(cudaCreateSurfaceObject(&_surface, &surfRes));
}

ArraySurface::ArraySurface(ArraySurface &&other) noexcept
{
    operator=(std::move(other));
}

void ArraySurface::destroy()
{
    if (_surface) {
        CUDA_CHK(cudaDestroySurfaceObject(std::exchange(_surface, 0)));
    }

    if (_array) {
        CUDA_CHK(cudaFreeArray(std::exchange(_array, nullptr)));
    }
}


ArraySurface & ArraySurface::operator=(ArraySurface &&other) noexcept
{
    if (&other != this) {
        destroy();
        _array = std::exchange(other._array, nullptr);
        _surface = std::exchange(other._surface, 0);
    }

    return *this;
}

TextureObject::TextureObject(cudaArray_t array,
                             cudaTextureAddressMode addressMode,
                             cudaTextureReadMode textureReadMode,
                             TextureSampling sampling)
{
    cudaResourceDesc texRes;
    std::memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = array;

    cudaTextureDesc texDescr;
    std::memset(&texDescr, 0, sizeof(cudaTextureDesc));
    switch (sampling) {
    case TextureSampling::NormalizedLinear:
        texDescr.normalizedCoords = 1;
        texDescr.filterMode = cudaFilterModeLinear;
        break;
    case TextureSampling::UnnormalizedPoint:
        texDescr.normalizedCoords = 0;
        texDescr.filterMode = cudaFilterModePoint;
        break;
    }
    texDescr.addressMode[0] = addressMode;
    texDescr.addressMode[1] = addressMode;
    texDescr.addressMode[2] = cudaAddressModeClamp;
    texDescr.readMode = textureReadMode;

    CUDA_CHK(cudaCreateTextureObject(&_object, &texRes, &texDescr, NULL));
}

TextureObject::TextureObject(TextureObject &&other) noexcept
{
    operator=(std::move(other));
}


TextureObject & TextureObject::operator=(TextureObject &&other) noexcept
{
    if (&other != this) {
        destroy();
        _object = std::exchange(other._object, 0);
    }

    return *this;
}

TextureObject::~TextureObject()
{
    destroy();
}

void TextureObject::destroy()
{
    if (_object) {
        CUDA_CHK(cudaDestroyTextureObject(_object));
        _object = 0;
    }
}

ArraySurfaceTexture &ArraySurfaceTexture::operator=(
    ArraySurfaceTexture &&other) noexcept
{
    if (this != &other) {
        _texture = TextureObject{};
        _arraySurface = std::move(other._arraySurface);
        _texture = std::move(other._texture);
    }

    return *this;
}

}
}
