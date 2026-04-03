#ifndef RENDER_BRIDGE_4CAM_HPP
#define RENDER_BRIDGE_4CAM_HPP

#include <array>

#include <engine/frame_packet.hpp>
#include <engine/frame_source.hpp>

#include "compat/render_bridge_4cam_contract.hpp"

namespace svapp
{

struct RuntimeRenderBridge4CamContext
{
    const videoio::SourceInfo *sourceInfo = nullptr;
    videoio::SourceInfo runtimeSourceInfo;
};

bool resolve_render_bridge_4cam(
    const camera::CameraRig &rig,
    std::array<camera::CameraRole, camera::CAMERAS_TOTAL> &renderRoles,
    std::array<uint32_t, 2> *uniformImageSize = nullptr);

bool prepare_runtime_render_bridge_4cam_context(
    const videoio::SourceInfo &sourceInfo,
    RuntimeRenderBridge4CamContext &context);

bool adapt_frame_packet_for_runtime_render_bridge_4cam(
    videoio::FramePacket &packet,
    const RuntimeRenderBridge4CamContext &context);

} // namespace svapp

#endif // RENDER_BRIDGE_4CAM_HPP
