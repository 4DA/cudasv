#include <cassert>
#include <array>
#include <string>

#include <spdlog/spdlog.h>

#include <tinygltf/stb_image.h>

#include "glcommon.hpp"
#include "texture.hpp"


using namespace rf;

Image rf::load_image_hdr(const std::string &file) {
    Image image;

    int x, y, comp;
    stbi_info (file.c_str(), &x, &y, &comp);

    image.data = stbi_loadf(file.c_str(), &image.w, &image.h, &image.channels, 4);
    if (!image.data) {
        SPDLOG_ERROR("Can not load HDR image: {}", file.c_str());
        return image;
    }

    image.channels = 4;

    SPDLOG_INFO("loaded image '{}': {} x {} @ {}", file.c_str(), image.w, image.h, image.channels);

    image.pixel_type = PixelChannelType::FLOAT32;

    return image;
}

Image rf::load_image(
    const std::string &file,
    bool try_16bit,
    bool flip_on_load,
    std::optional<unsigned int> channelCount)
{
    if (flip_on_load) {
        stbi_set_flip_vertically_on_load(1);
    }

    Image image;

    image.channels = 4;

    unsigned int req_channels;

    if (channelCount) {
        req_channels = *channelCount;
    } else {
        req_channels = 0;
    }

    if (try_16bit) {
        image.data = stbi_load_16(file.c_str(), &image.w, &image.h, &image.channels, req_channels);
        image.pixel_type = PixelChannelType::U16;
    }
    else {
        image.data = stbi_load(file.c_str(), &image.w, &image.h, &image.channels, req_channels);
        image.pixel_type = PixelChannelType::U8;
    }

    if (!image.data) {
        SPDLOG_ERROR("Can not load image: {}", file.c_str());
        return image;
    }

    image.name = file;

    if (channelCount) {
        image.channels = *channelCount;
    }

    SPDLOG_INFO("Loaded image '{}': {} x {} @ {}", file.c_str(), image.w, image.h, image.channels);

    if (flip_on_load) {
        stbi_set_flip_vertically_on_load(0);
    }

    return image;
}

// }
