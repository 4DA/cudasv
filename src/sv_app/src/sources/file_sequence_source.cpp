#include "sources/file_sequence_source.hpp"

#include <fstream>

#include "sources/render_bridge_4cam.hpp"
#include "sources/rigged_png_source_support.hpp"

namespace
{

std::string resolve_rig_path(const std::string &rig_path)
{
    if (std::ifstream(rig_path).good()) {
        return rig_path;
    }

    return rig_path;
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
    _info.render_roles = kRenderBridge4CameraRoles;
}

bool FileSequenceSource::open()
{
    const std::string rig_path = resolve_rig_path(_config.rig_path);
    return open_rigged_png_source_for_render_bridge_4cam(
        rig_path,
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

    fill_static_frame_packet_metadata(_frame_id++, packet);
    return true;
}

bool FileSequenceSource::release_frame(const videoio::FramePacket &packet)
{
    return _png_source.release_frame(packet.frames);
}

} // namespace svapp
