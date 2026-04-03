#ifndef RENDER_BRIDGE_4CAM_HPP
#define RENDER_BRIDGE_4CAM_HPP

#include <array>

#include <engine/frame_packet.hpp>
#include <engine/frame_source.hpp>

#include "compat/render_bridge_4cam_contract.hpp"

namespace svapp
{

bool resolve_render_bridge_4cam(
    const camera::CameraRig &rig,
    std::array<camera::CameraRole, camera::CAMERAS_TOTAL> &renderRoles,
    std::array<uint32_t, 2> *uniformImageSize = nullptr);

bool remap_source_frame_packet_to_render_bridge_4cam(
    videoio::FramePacket &packet,
    const videoio::SourceInfo &sourceInfo,
    const videoio::SourceInfo &renderBridgeInfo);

} // namespace svapp

#endif // RENDER_BRIDGE_4CAM_HPP
