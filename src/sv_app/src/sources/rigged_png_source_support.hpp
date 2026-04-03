#ifndef RIGGED_PNG_SOURCE_SUPPORT_HPP
#define RIGGED_PNG_SOURCE_SUPPORT_HPP

#include <array>
#include <string>

#include <engine/frame_source.hpp>
#include <engine/png_source.hpp>

namespace svapp
{

bool open_rigged_png_source_for_render_bridge_4cam(
    const std::string &rigPath,
    const std::array<std::string, camera::CAMERAS_TOTAL> &framePaths,
    camera::CameraRig &rig,
    videoio::SourceInfo &info,
    videoio::PNGSource &pngSource);

void fill_static_frame_packet_metadata(uint64_t frameId, videoio::FramePacket &packet);

} // namespace svapp

#endif // RIGGED_PNG_SOURCE_SUPPORT_HPP
