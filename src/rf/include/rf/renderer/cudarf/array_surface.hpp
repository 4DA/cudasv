#ifndef CUDARF_ARRAY_SURFACE_HPP
#define CUDARF_ARRAY_SURFACE_HPP

#include <cuda_runtime.h>

namespace cudarf
{
namespace memory
{

enum class TextureSampling {
    NormalizedLinear,
    UnnormalizedPoint,
};

class ArraySurface {
public:
    ArraySurface(unsigned int width,
                 unsigned int height,
                 cudaChannelFormatDesc channelDesc) noexcept;

    ArraySurface(ArraySurface &&) noexcept;
    ArraySurface &operator=(ArraySurface &&) noexcept;
    ~ArraySurface() {destroy();}

    ArraySurface(const ArraySurface &) = delete;
    ArraySurface &operator=(const ArraySurface &) = delete;

    cudaSurfaceObject_t surface() const {return _surface;}
    cudaArray_t array() const {return _array;};

private:
    void destroy();
    cudaArray_t _array = nullptr;
    cudaSurfaceObject_t _surface = 0;
};

class TextureObject {
public:
    TextureObject() = default;

    // note: TextureObject depends on supplied array, which must be valid
    // during lifetime of TextureObject
    TextureObject(cudaArray_t array,
                  cudaTextureAddressMode addressMode,
                  cudaTextureReadMode textureReadMode,
                  TextureSampling sampling);

    TextureObject(TextureObject&&) noexcept;
    TextureObject& operator=(TextureObject&&) noexcept;

    TextureObject (const TextureObject&) = delete;
    TextureObject& operator=(const TextureObject&) = delete;

    ~TextureObject();

    cudaTextureObject_t get() const { return _object; }

private:
    void destroy();
    cudaTextureObject_t _object = 0;
};

class ArraySurfaceTexture {
  public:
      ArraySurfaceTexture(unsigned int width,
                          unsigned int height,
                          cudaChannelFormatDesc channelDesc,
                          cudaTextureAddressMode addressMode,
                          cudaTextureReadMode textureReadMode,
                          TextureSampling sampling)
          : _arraySurface(width, height, channelDesc),
            _texture(_arraySurface.array(), addressMode,
                     textureReadMode, sampling) {}

      ArraySurfaceTexture(ArraySurfaceTexture&&) noexcept = default;
      ArraySurfaceTexture&
      operator=(ArraySurfaceTexture&&) noexcept;

      ArraySurfaceTexture(const ArraySurfaceTexture&) = delete;
      ArraySurfaceTexture&
      operator=(const ArraySurfaceTexture&) = delete;

      cudaArray_t array() const {
          return _arraySurface.array();
      }

      cudaSurfaceObject_t surface() const {
          return _arraySurface.surface();
      }

      cudaTextureObject_t texture() const {
          return _texture.get();
      }

  private:
      ArraySurface _arraySurface;
      TextureObject _texture;
};


}
}

#endif
