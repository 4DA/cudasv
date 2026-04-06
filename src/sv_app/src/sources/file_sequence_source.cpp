#include "sources/file_sequence_source.hpp"

#include "sources/render_bridge_4cam.hpp"
#include "sources/rigged_png_source_support.hpp"

namespace
{

static videoio::FrameSet<camera::CAMERAS_TOTAL> frame_set_from_source_packet(
    const videoio::FramePacket &packet)
{
    videoio::FrameSet<camera::CAMERAS_TOTAL> frames = {};

    for (std::size_t index = 0; index < packet.cameras.size() && index < camera::CAMERAS_TOTAL; ++index) {
        frames.data[index] = packet.cameras[index].data;
        frames.userdata[index] = packet.cameras[index].userdata;
        frames.width = packet.cameras[index].width;
        frames.height = packet.cameras[index].height;
        frames.stride = packet.cameras[index].stride;
        frames.timestamp = packet.cameras[index].timestamp_ns;
    }

    return frames;
}

} // namespace

namespace svapp
{

FileSequenceSource::FileSequenceSource(FileSequenceSourceConfig config):
    _config(std::move(config))
{
    _info.kind = videoio::SourceKind::FileSequence;
    _info.source_name = "file_sequence";
    _info.dataset_root = _config.dataset_root;
    _info.sequence_id = _config.sequence_id;
    _info.sequence_frame_count = 1;
    _info.has_sequence_frame_count = true;
    _info.contract.synchronized_samples = true;
    _info.contract.provides_sample_identity = true;
    _info.contract.provides_source_timestamp = false;
    _info.contract.provides_per_camera_timestamps = false;
    _info.contract.provides_ego_pose = false;
    _info.source_roles = {
        camera::CameraRole::Right,
        camera::CameraRole::Left,
        camera::CameraRole::Front,
        camera::CameraRole::Rear,
    };
    _info.render_roles = kRenderBridge4CameraRoles;
}

bool FileSequenceSource::open()
{
    return open_rigged_png_source_for_render_bridge_4cam(
        _config.rig_path,
        _config.frame_paths,
        _rig,
        _info,
        _png_source);
}

const camera::CameraRig& FileSequenceSource::rig() const
{
    return _rig;
}

const videoio::SourceInfo& FileSequenceSource::info() const
{
    return _info;
}

bool FileSequenceSource::get_next_frame(videoio::FramePacket &packet)
{
    videoio::FrameSet<camera::CAMERAS_TOTAL> frames;
    if (!_png_source.get_next_frame(frames)) {
        return false;
    }

    fill_static_frame_packet_metadata(_frame_id++, _info, packet);
    packet.cameras.clear();
    packet.cameras.reserve(camera::CAMERAS_TOTAL);

    for (std::size_t index = 0; index < camera::CAMERAS_TOTAL; ++index) {
        videoio::SourceCameraFrame cameraFrame;
        cameraFrame.role = _info.source_roles[index];
        cameraFrame.data = frames.data[index];
        cameraFrame.userdata = frames.userdata[index];
        cameraFrame.width = frames.width;
        cameraFrame.height = frames.height;
        cameraFrame.stride = frames.stride;
        cameraFrame.timestamp_ns = frames.timestamp;
        cameraFrame.has_timestamp = false;
        cameraFrame.valid = true;
        packet.cameras.push_back(cameraFrame);
    }

    return true;
}

bool FileSequenceSource::release_frame(const videoio::FramePacket &packet)
{
    return _png_source.release_frame(frame_set_from_source_packet(packet));
}

} // namespace svapp
