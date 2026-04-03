#ifndef RENDER_BRIDGE_4CAM_CONTRACT_HPP
#define RENDER_BRIDGE_4CAM_CONTRACT_HPP

#include <array>
#include <string>

#include <engine/camera_config.hpp>
#include <engine/camera_rig.hpp>

namespace svapp
{

inline constexpr std::array<camera::CameraRole, camera::CAMERAS_TOTAL> kRenderBridge4CameraRoles = {
    camera::CameraRole::Right,
    camera::CameraRole::Left,
    camera::CameraRole::Front,
    camera::CameraRole::Rear,
};

bool render_bridge_4cam_slot_for_role(camera::CameraRole role,
                                      camera::CameraID *slot);

bool resolve_render_bridge_4cam_source_slot_indices(
    const std::array<camera::CameraRole, camera::CAMERAS_TOTAL> &sourceRoles,
    const std::string &sourceName,
    std::array<int, camera::CAMERAS_TOTAL> &sourceIndexByRenderSlot);

} // namespace svapp

#endif // RENDER_BRIDGE_4CAM_CONTRACT_HPP
