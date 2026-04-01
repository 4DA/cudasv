#ifndef RF_TEXTURE_HPP
#define RF_TEXTURE_HPP

#include <array>
#include <functional>
#include <optional>
#include <vector>

namespace rf
{

enum class PixelChannelType
{
    FLOAT16,
    FLOAT32,
    U8,
    U16
};

struct Image {
    int w;                       // width in pixels
    int h;                       // height in pixels
    int channels;                // number of channels in each pixel
    PixelChannelType pixel_type; // pixel format
    const void *data;            // data pointer
    std::string name;            // name of image

    Image(): w(0), h(0), channels(0), data(nullptr) {}
};


using CubemapDescription = std::array<Image, 6>;

Image load_image_hdr(const std::string &file);

// When data is sourced from GLB, the UV coordinate system aligns with the
// image layout in memory, meaning that (0,0) corresponds to the lower-left
// pixel.
// Therefore, it is irrelevant that OpenGL's UV space begins at the
// upper-left pixel. However, when UV coordinates are generated (as in the case
// of the GGX lookup table), the image must be flipped before it is uploaded to
// the GPU to ensure proper alignment.

Image load_image(
    const std::string &file,
    bool try_16bit,
    bool flip_on_load = false,
    std::optional<unsigned int> channelCount = std::nullopt);

} // namespace rf

#endif
