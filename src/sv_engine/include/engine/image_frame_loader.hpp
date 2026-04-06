#ifndef IMAGE_FRAME_LOADER_HPP
#define IMAGE_FRAME_LOADER_HPP

#include <cstdint>
#include <memory>
#include <string>

namespace videoio
{

struct DecodedImageFrame
{
    std::shared_ptr<void> owner;
    uint8_t *data = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t stride = 0;
};

bool load_rgb_image(const std::string &path,
                    DecodedImageFrame &frame,
                    std::string *errorMessage = nullptr);

} // namespace videoio

#endif // IMAGE_FRAME_LOADER_HPP
