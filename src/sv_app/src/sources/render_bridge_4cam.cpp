#include "sources/render_bridge_4cam.hpp"

#include <array>
#include <set>

#include <spdlog/spdlog.h>

namespace svapp
{

bool resolve_render_bridge_4cam(
    const camera::CameraRig &rig,
    std::array<camera::CameraRole, camera::CAMERAS_TOTAL> &renderRoles,
    std::array<uint32_t, 2> *uniformImageSize)
{
    if (rig.cameras.size() != camera::CAMERAS_COUNT) {
        SPDLOG_ERROR("Current render bridge requires exactly {} cameras in the rig",
                     static_cast<int>(camera::CAMERAS_COUNT));
        return false;
    }

    std::set<camera::CameraRole> seenRoles;
    std::array<uint32_t, 2> referenceSize = {0, 0};

    for (const auto &cameraDesc : rig.cameras) {
        camera::CameraID renderSlot;
        if (!render_bridge_4cam_slot_for_role(cameraDesc.role, &renderSlot)) {
            SPDLOG_ERROR("Rig contains role that is not supported by the 4-camera renderer");
            return false;
        }
        (void)renderSlot;

        if (!seenRoles.insert(cameraDesc.role).second) {
            SPDLOG_ERROR("Rig contains duplicate camera role");
            return false;
        }

        if (referenceSize[0] == 0 && referenceSize[1] == 0) {
            referenceSize = cameraDesc.image_size;
        } else if (cameraDesc.image_size != referenceSize) {
            SPDLOG_ERROR("Current 4-camera render bridge expects all render cameras to have the same image size");
            return false;
        }
    }

    for (const auto expectedRole : kRenderBridge4CameraRoles) {
        if (!seenRoles.count(expectedRole)) {
            SPDLOG_ERROR("Rig is missing one of the required 4-camera render roles");
            return false;
        }
    }

    renderRoles = kRenderBridge4CameraRoles;
    if (uniformImageSize != nullptr) {
        *uniformImageSize = referenceSize;
    }

    return true;
}

bool prepare_runtime_render_bridge_4cam_context(
    const videoio::SourceInfo &sourceInfo,
    RuntimeRenderBridge4CamContext &context)
{
    context.sourceInfo = &sourceInfo;
    context.runtimeSourceInfo = sourceInfo;
    context.runtimeSourceInfo.render_roles = kRenderBridge4CameraRoles;
    return true;
}

bool adapt_frame_packet_for_runtime_render_bridge_4cam(
    const videoio::FramePacket &sourcePacket,
    videoio::RuntimeFramePacket4Cam &runtimePacket,
    const RuntimeRenderBridge4CamContext &context)
{
    const videoio::SourceInfo &sourceInfo = *context.sourceInfo;
    const videoio::SourceInfo &renderBridgeInfo = context.runtimeSourceInfo;
    std::array<int, camera::CAMERAS_TOTAL> sourceIndexByRenderSlot = {-1, -1, -1, -1};

    for (std::size_t sourceIndex = 0; sourceIndex < sourcePacket.cameras.size(); ++sourceIndex) {
        camera::CameraID renderSlot;
        if (!render_bridge_4cam_slot_for_role(sourcePacket.cameras[sourceIndex].role, &renderSlot)) {
            continue;
        }

        if (sourceIndexByRenderSlot[renderSlot] >= 0) {
            SPDLOG_ERROR("Source '{}' frame packet contains duplicate camera role for the current 4-camera bridge",
                         sourceInfo.source_name);
            return false;
        }

        sourceIndexByRenderSlot[renderSlot] = static_cast<int>(sourceIndex);
    }

    for (std::size_t renderSlot = 0; renderSlot < renderBridgeInfo.render_roles.size(); ++renderSlot) {
        if (renderBridgeInfo.render_roles[renderSlot] != kRenderBridge4CameraRoles[renderSlot]) {
            SPDLOG_ERROR("Render-bridge role description does not match the expected 4-camera slot contract");
            return false;
        }

        const int sourceIndex = sourceIndexByRenderSlot[renderSlot];
        if (sourceIndex < 0) {
            SPDLOG_ERROR("Source '{}' does not provide a complete 4-camera render-role mapping",
                         sourceInfo.source_name);
            return false;
        }

        runtimePacket.frames.data[renderSlot] = sourcePacket.cameras[sourceIndex].data;
        runtimePacket.frames.userdata[renderSlot] = sourcePacket.cameras[sourceIndex].userdata;
        runtimePacket.valid_cameras[renderSlot] = sourcePacket.cameras[sourceIndex].valid;
    }

    runtimePacket.frames.width = sourcePacket.cameras[0].width;
    runtimePacket.frames.height = sourcePacket.cameras[0].height;
    runtimePacket.frames.stride = sourcePacket.cameras[0].stride;
    runtimePacket.frames.timestamp = sourcePacket.cameras[0].timestamp_ns;
    runtimePacket.frames.frameseq = sourcePacket.metadata.source_frame_sequence;
    runtimePacket.metadata = sourcePacket.metadata;

    return true;
}

} // namespace svapp
