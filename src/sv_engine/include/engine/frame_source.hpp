#ifndef FRAME_SOURCE_HPP
#define FRAME_SOURCE_HPP

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <engine/camera_rig.hpp>
#include <engine/frame_packet.hpp>

namespace videoio
{

enum class SourceKind
{
    Unknown,
    FileSequence,
    NuScenes,
};

struct SourceInfo
{
    struct Contract {
        bool synchronized_samples = true;
        bool provides_sample_identity = false;
        bool provides_source_timestamp = false;
        bool provides_per_camera_timestamps = false;
        bool provides_ego_pose = false;
    };

    SourceKind kind = SourceKind::Unknown;
    std::string source_name;
    std::string dataset_root;
    std::string sequence_id;
    uint64_t sequence_frame_count = 0;
    bool has_sequence_frame_count = false;
    Contract contract;
    std::vector<camera::CameraRole> source_roles;
    std::array<camera::CameraRole, camera::CAMERAS_TOTAL> render_roles = {
        camera::CameraRole::Right,
        camera::CameraRole::Left,
        camera::CameraRole::Front,
        camera::CameraRole::Rear,
    };
};

struct FrameSource
{
    virtual ~FrameSource() = default;

    virtual bool open() = 0;
    virtual const camera::CameraRig& rig() const = 0;
    virtual const SourceInfo& info() const = 0;
    virtual bool get_next_frame(FramePacket &packet) = 0;
    virtual bool release_frame(const FramePacket &packet) = 0;
};

}

#endif // FRAME_SOURCE_HPP
