#ifndef FRAME_PACKET_HPP
#define FRAME_PACKET_HPP

#include <array>
#include <cstdint>

#include <engine/camera_config.hpp>
#include <engine/video_source.hpp>

namespace videoio
{

struct FramePacket
{
    FrameSet<camera::CAMERAS_TOTAL> frames;
    uint64_t frame_id = 0;
    uint64_t source_timestamp_ns = 0;
    bool has_source_timestamp = false;
    std::array<bool, camera::CAMERAS_TOTAL> valid_cameras = {true, true, true, true};
};

}

#endif // FRAME_PACKET_HPP
