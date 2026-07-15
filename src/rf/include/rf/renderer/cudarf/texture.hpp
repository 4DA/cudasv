#ifndef CUDARF_TEXTURE
#define CUDARF_TEXTURE

#include <type_traits>
#include <optional>

#include <cuda_runtime.h>

#include <rf/renderer/glm_common.hpp>
#include <rf/renderer/image.hpp>

namespace cudarf
{
struct Texture {
    cudaTextureObject_t textureObject = 0;
    unsigned int channels = 0;
    unsigned int width = 0;
    unsigned int height = 0;
    int mipLevels = 0;
    bool hasUVTransform = false;
    glm::mat3 uvTransform = glm::mat3(1.0f);
};

static_assert(std::is_trivially_copyable_v<Texture>);

class TextureResource {
public:
    TextureResource() = default;

    TextureResource(Texture view,
                    cudaArray_t array,
                    cudaMipmappedArray_t mipmappedArray);

    TextureResource(TextureResource&&) noexcept;
    TextureResource& operator=(TextureResource&&) noexcept;
    ~TextureResource();

    TextureResource(const TextureResource&) = delete;
    TextureResource& operator=(const TextureResource&) = delete;

    Texture view() const noexcept {return _view;}

private:
    void reset();
    cudaArray_t _array = nullptr;
    cudaMipmappedArray_t _mipmappedArray = nullptr;
    Texture _view = Texture{};
};

std::optional<TextureResource>
create_cuda_texture(rf::Image image,
                    cudaTextureAddressMode addressMode,
                    unsigned int mipLevels,
                    std::optional<glm::mat3> uvTransform,
                    cudaStream_t cuStream);
}
#endif
