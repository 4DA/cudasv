#include "sources/file_sequence_source.hpp"

#include "sources/render_bridge_4cam.hpp"
#include "sources/rigged_png_source_support.hpp"

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
    if (!_png_source.get_next_frame(packet.frames)) {
        return false;
    }

    fill_static_frame_packet_metadata(_frame_id++, _info, packet);
    return true;
}

bool FileSequenceSource::release_frame(const videoio::FramePacket &packet)
{
    return _png_source.release_frame(packet.frames);
}

} // namespace svapp
