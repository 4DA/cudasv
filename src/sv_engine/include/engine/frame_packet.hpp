#ifndef FRAME_PACKET_HPP
#define FRAME_PACKET_HPP

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <engine/camera_config.hpp>
#include <engine/camera_rig.hpp>
#include <engine/video_source.hpp>

namespace videoio
{

struct FramePacketMetadata
{
    uint64_t frame_id = 0;  // current demo path uses this as a stable sample index
    uint64_t source_frame_sequence = 0;
    std::string sample_id;
    bool has_sample_id = false;
    bool synchronized_cameras = true;
    uint64_t source_timestamp_ns = 0;
    bool has_source_timestamp = false;
};

struct SourceCameraFrame
{
    camera::CameraRole role = camera::CameraRole::Unknown;
    uint8_t *data = nullptr;
    void *userdata = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t stride = 0;
    uint64_t timestamp_ns = 0;
    bool has_timestamp = false;
    bool valid = true;
};

struct FramePacket
{
    FramePacketMetadata metadata;
    std::vector<SourceCameraFrame> cameras;
};

struct RuntimeFramePacket4Cam
{
    FramePacketMetadata metadata;
    FrameSet<camera::CAMERAS_TOTAL> frames;
    std::array<bool, camera::CAMERAS_TOTAL> valid_cameras = {true, true, true, true};
};

}

#endif // FRAME_PACKET_HPP
