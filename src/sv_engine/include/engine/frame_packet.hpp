#ifndef FRAME_PACKET_HPP
#define FRAME_PACKET_HPP

#include <array>
#include <cstdint>
#include <string>

#include <engine/camera_config.hpp>
#include <engine/video_source.hpp>

namespace videoio
{

struct FramePacketMetadata
{
    uint64_t frame_id = 0;  // current demo path uses this as a stable sample index
    std::string sample_id;
    bool has_sample_id = false;
    bool synchronized_cameras = true;
    uint64_t source_timestamp_ns = 0;
    bool has_source_timestamp = false;
    std::array<uint64_t, camera::CAMERAS_TOTAL> camera_timestamps_ns = {0, 0, 0, 0};
    std::array<bool, camera::CAMERAS_TOTAL> has_camera_timestamps = {false, false, false, false};
};

struct FramePacket
{
    FramePacketMetadata metadata;
    FrameSet<camera::CAMERAS_TOTAL> frames;
    std::array<bool, camera::CAMERAS_TOTAL> valid_cameras = {true, true, true, true};
};

struct RuntimeFramePacket4Cam
{
    FramePacketMetadata metadata;
    FrameSet<camera::CAMERAS_TOTAL> frames;
    std::array<bool, camera::CAMERAS_TOTAL> valid_cameras = {true, true, true, true};
};

}

#endif // FRAME_PACKET_HPP
