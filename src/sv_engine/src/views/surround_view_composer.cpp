#include "views/surround_view_composer.hpp"

#include <cstring>

#include <world.hpp>

#include <engine/camera_config.hpp>
#include <engine/engine.hpp>

#include <rf/renderer/virtual_camera.hpp>
#include <rf/renderer/surround_view/projector.hpp>

namespace
{

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

    result.vehicle_width = config.vehicle_config.width;
    result.vehicle_length = config.vehicle_config.length;

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
