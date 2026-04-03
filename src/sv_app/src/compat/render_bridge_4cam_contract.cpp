#include "compat/render_bridge_4cam_contract.hpp"

#include <spdlog/spdlog.h>

namespace svapp
{

bool render_bridge_4cam_slot_for_role(camera::CameraRole role,
                                      camera::CameraID *slot)
{
    if (slot == nullptr) {
        SPDLOG_ERROR("Render-bridge slot output pointer is null");
        return false;
    }

    switch (role)
    {
    case camera::CameraRole::Right:
        *slot = camera::CAMERA_RIGHT;
        return true;
    case camera::CameraRole::Left:
        *slot = camera::CAMERA_LEFT;
        return true;
    case camera::CameraRole::Front:
        *slot = camera::CAMERA_FRONT;
        return true;
    case camera::CameraRole::Rear:
        *slot = camera::CAMERA_REAR;
        return true;
    default:
        return false;
    }
}

bool resolve_render_bridge_4cam_source_slot_indices(
    const std::array<camera::CameraRole, camera::CAMERAS_TOTAL> &sourceRoles,
    const std::string &sourceName,
    std::array<int, camera::CAMERAS_TOTAL> &sourceIndexByRenderSlot)
{
    sourceIndexByRenderSlot = {-1, -1, -1, -1};

    for (std::size_t sourceIndex = 0; sourceIndex < sourceRoles.size(); ++sourceIndex) {
        camera::CameraID slot;
        if (!render_bridge_4cam_slot_for_role(sourceRoles[sourceIndex], &slot)) {
            SPDLOG_ERROR("Source '{}' exposes an unsupported role in the current 4-camera render bridge",
                         sourceName);
            return false;
        }

        if (sourceIndexByRenderSlot[slot] >= 0) {
            SPDLOG_ERROR("Source '{}' maps multiple cameras into the same 4-camera render slot",
                         sourceName);
            return false;
        }

        sourceIndexByRenderSlot[slot] = static_cast<int>(sourceIndex);
    }

    return true;
}

} // namespace svapp
