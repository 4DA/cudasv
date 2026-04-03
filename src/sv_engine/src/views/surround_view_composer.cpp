#include "views/surround_view_composer.hpp"

#include <cstring>

#include <world.hpp>

#include <engine/camera_config.hpp>
#include <engine/engine.hpp>

#include <rf/renderer/virtual_camera.hpp>
#include <rf/renderer/surround_view/projector.hpp>

namespace
{

uint32_t map_debug_mode(int mode)
{
    switch (mode) {
    case SURROUND_VIEW_DEBUG_NORMAL:
        return rf::surround_view::DEBUG_MODE_NORMAL;
    case SURROUND_VIEW_DEBUG_CAMERA_ROLES:
        return rf::surround_view::DEBUG_MODE_CAMERA_ROLES;
    case SURROUND_VIEW_DEBUG_COVERAGE_MASK:
        return rf::surround_view::DEBUG_MODE_COVERAGE_MASK;
    case SURROUND_VIEW_DEBUG_REPROJECTION_GRID:
        return rf::surround_view::DEBUG_MODE_REPROJECTION_GRID;
    default:
        return rf::surround_view::DEBUG_MODE_NORMAL;
    }
}

uint32_t map_debug_camera_role(const std::string &cameraRole)
{
    if (cameraRole == "right") {
        return camera::CAMERA_RIGHT;
    }
    if (cameraRole == "left") {
        return camera::CAMERA_LEFT;
    }
    if (cameraRole == "front") {
        return camera::CAMERA_FRONT;
    }
    if (cameraRole == "rear") {
        return camera::CAMERA_REAR;
    }

    return camera::CAMERA_FRONT;
}

rf::surround_view::CameraProjectionParams make_projection_params(const camera::CameraConfig &cameraConfig)
{
    rf::surround_view::CameraProjectionParams result {};
    glm::vec3 cameraTranslation = glm::make_vec3(cameraConfig.extrinsics.T.data());
    glm::mat3 cameraRotation = glm::make_mat3(cameraConfig.extrinsics.R.data());

    // Extrinsics are stored as camera->vehicle. Projection uses vehicle->camera.
    result.world_to_camera = glm::transpose(glm::mat4(cameraRotation));
    result.world_to_camera = glm::translate(result.world_to_camera, -cameraTranslation);

    result.intrinsics = glm::make_mat3(cameraConfig.intrinsics.camera.data());
    std::memcpy(result.distortion_coeffs,
                cameraConfig.intrinsics.distortion_coeffs.data(),
                sizeof(result.distortion_coeffs));

    return result;
}

rf::surround_view::RigParams make_camera_params(const engine::Config &config)
{
    rf::surround_view::RigParams result {};

    for (unsigned int cameraIndex = 0; cameraIndex < camera::CAMERAS_TOTAL; ++cameraIndex) {
        result.projections[cameraIndex] =
            make_projection_params(config.calibration_config.camera_cfg[cameraIndex]);
    }

    const auto &blendConfig = config.overlays_config.surround_view_blend_config;
    result.front_extent = config.vehicle_config.length / 2.0f + blendConfig.front_anchor_offset_mm;
    result.rear_extent = config.vehicle_config.length / 2.0f + blendConfig.rear_anchor_offset_mm;
    result.left_extent = config.vehicle_config.width / 2.0f + blendConfig.left_anchor_offset_mm;
    result.right_extent = config.vehicle_config.width / 2.0f + blendConfig.right_anchor_offset_mm;
    result.front_left_start_rad = glm::radians(blendConfig.front_left_start_deg);
    result.front_left_end_rad = glm::radians(blendConfig.front_left_end_deg);
    result.right_front_start_rad = glm::radians(blendConfig.right_front_start_deg);
    result.right_front_end_rad = glm::radians(blendConfig.right_front_end_deg);
    result.rear_right_start_rad = glm::radians(blendConfig.rear_right_start_deg);
    result.rear_right_end_rad = glm::radians(blendConfig.rear_right_end_deg);
    result.left_rear_start_rad = glm::radians(blendConfig.left_rear_start_deg);
    result.left_rear_end_rad = glm::radians(blendConfig.left_rear_end_deg);
    result.debug_mode = map_debug_mode(config.overlays_config.surround_view_debug_config.mode);
    result.debug_camera_index =
        map_debug_camera_role(config.overlays_config.surround_view_debug_config.camera_role);

    return result;
}

rf::surround_view::ViewParams make_virtual_camera_params(const rf::VirtualCamera &virtualCamera)
{
    rf::surround_view::ViewParams result {};
    result.origin = virtualCamera.transform.translation;

    virtualCamera.get_virtual_image_plane_params(result.horizontal,
                                                 result.vertical,
                                                 result.lower_left);

    return result;
}

} // namespace

void engine::view::SurroundViewComposer::compose(
    const Config &config,
    const World &world,
    const rf::VirtualCamera &virtualCamera,
    unsigned int width,
    unsigned int height,
    cudarf::Framebuffer meshGPUOutput,
    cudarf::CudaStreams cudaStreams) const
{
    rf::surround_view::project_async(world.camera_textures(),
                                     width,
                                     height,
                                     world.frame_projector().tex_width,
                                     world.frame_projector().tex_height,
                                     make_virtual_camera_params(virtualCamera),
                                     make_camera_params(config),
                                     meshGPUOutput,
                                     cudaStreams.rendering);
}
