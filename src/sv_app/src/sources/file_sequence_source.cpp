#include "sources/file_sequence_source.hpp"

#include <algorithm>
#include <fstream>
#include <set>

#include <spdlog/spdlog.h>

#include "config/canonical_rig.hpp"

namespace
{

bool all_paths_have_png_extension(const std::array<std::string, camera::CAMERAS_TOTAL> &paths)
{
    return std::all_of(paths.begin(), paths.end(), [](const std::string &path) {
        const auto dot = path.find_last_of('.');
        return dot != std::string::npos && path.substr(dot) == ".png";
    });
}

std::string resolve_rig_path(const std::string &rig_path)
{
    if (std::ifstream(rig_path).good()) {
        return rig_path;
    }

    return rig_path;
}

bool validate_render_bridge_compatibility(const camera::CameraRig &rig)
{
    if (rig.cameras.size() != camera::CAMERAS_COUNT) {
        SPDLOG_ERROR("Current render bridge requires exactly {} cameras in the rig",
                     static_cast<int>(camera::CAMERAS_COUNT));
        return false;
    }

    std::set<camera::CameraRole> seen_roles;
    std::array<uint32_t, 2> reference_size = {0, 0};

    for (const auto &camera_desc: rig.cameras) {
        if (!seen_roles.insert(camera_desc.role).second) {
            SPDLOG_ERROR("Rig contains duplicate camera role");
            return false;
        }

        switch (camera_desc.role) {
        case camera::CameraRole::Front:
        case camera::CameraRole::Rear:
        case camera::CameraRole::Left:
        case camera::CameraRole::Right:
            break;
        default:
            SPDLOG_ERROR("Rig contains role that is not supported by the 4-camera renderer");
            return false;
        }

        if (reference_size[0] == 0 && reference_size[1] == 0) {
            reference_size = camera_desc.image_size;
        } else if (camera_desc.image_size != reference_size) {
            SPDLOG_ERROR("Current PNG source expects all render cameras to have the same image size");
            return false;
        }
    }

    return seen_roles.count(camera::CameraRole::Front) == 1 &&
           seen_roles.count(camera::CameraRole::Rear) == 1 &&
           seen_roles.count(camera::CameraRole::Left) == 1 &&
           seen_roles.count(camera::CameraRole::Right) == 1;
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
}

bool FileSequenceSource::open()
{
    if (!all_paths_have_png_extension(_config.frame_paths)) {
        SPDLOG_ERROR("All --frames inputs must use the .png extension");
        return false;
    }

    const std::string rig_path = resolve_rig_path(_config.rig_path);
    if (load_canonical_rig(&_rig, rig_path)) {
        SPDLOG_ERROR("Failed to load canonical rig from {}", rig_path);
        return false;
    }

    if (!validate_render_bridge_compatibility(_rig)) {
        SPDLOG_ERROR("Rig is not compatible with the current 4-camera renderer");
        return false;
    }

    if (_png_source.start(_config.frame_paths) != true) {
        SPDLOG_ERROR("Failed to start PNG frame source");
        return false;
    }

    return true;
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

    packet.frame_id = _frame_id++;
    packet.source_timestamp_ns = 0;
    packet.has_source_timestamp = false;
    packet.valid_cameras = {true, true, true, true};
    return true;
}

bool FileSequenceSource::release_frame(const videoio::FramePacket &packet)
{
    return _png_source.release_frame(packet.frames);
}

} // namespace svapp
