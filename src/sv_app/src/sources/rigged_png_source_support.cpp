#include <algorithm>
#include <string>

#include <spdlog/spdlog.h>

#include "sources/rigged_png_source_support.hpp"
#include "config/canonical_rig.hpp"
#include "sources/render_bridge_4cam.hpp"

namespace
{

bool all_paths_have_png_extension(const std::array<std::string, camera::CAMERAS_TOTAL> &paths)
{
    return std::all_of(paths.begin(), paths.end(), [](const std::string &path) {
        const auto dot = path.find_last_of('.');
        return dot != std::string::npos && path.substr(dot) == ".png";
    });
}

} // namespace

namespace svapp
{

bool open_rigged_png_source_for_render_bridge_4cam(
    const std::string &rigPath,
    const std::array<std::string, camera::CAMERAS_TOTAL> &framePaths,
    camera::CameraRig &rig,
    videoio::SourceInfo &info,
    videoio::PNGSource &pngSource)
{
    if (!all_paths_have_png_extension(framePaths)) {
        SPDLOG_ERROR("All render-bridge frame inputs must use the .png extension");
        return false;
    }

    if (load_canonical_rig(&rig, rigPath)) {
        SPDLOG_ERROR("Failed to load canonical rig from {}", rigPath);
        return false;
    }

    if (!resolve_render_bridge_4cam(rig, info.render_roles)) {
        SPDLOG_ERROR("Rig is not compatible with the current 4-camera renderer");
        return false;
    }

    if (!pngSource.start(framePaths)) {
        SPDLOG_ERROR("Failed to start PNG frame source");
        return false;
    }

    return true;
}
void fill_static_frame_packet_metadata(uint64_t frameId,
                                       const videoio::SourceInfo &sourceInfo,
                                       videoio::FramePacket &packet)
{
    packet.frame_id = frameId;
    packet.sample_id = sourceInfo.source_name + ":" + std::to_string(frameId);
    packet.has_sample_id = sourceInfo.contract.provides_sample_identity;
    packet.synchronized_cameras = sourceInfo.contract.synchronized_samples;
    packet.source_timestamp_ns = 0;
    packet.has_source_timestamp = sourceInfo.contract.provides_source_timestamp;
    packet.camera_timestamps_ns = {0, 0, 0, 0};
    packet.has_camera_timestamps = {
        sourceInfo.contract.provides_per_camera_timestamps,
        sourceInfo.contract.provides_per_camera_timestamps,
        sourceInfo.contract.provides_per_camera_timestamps,
        sourceInfo.contract.provides_per_camera_timestamps,
    };
    packet.valid_cameras = {true, true, true, true};
}

} // namespace svapp
