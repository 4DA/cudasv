#include "views/viewpoint_registry.hpp"

namespace engine::view
{

std::optional<rf::Viewpoint> ViewpointRegistry::create_viewpoint(const ViewConfig3D &view_config,
                                                                 const Viewpoint3D &preset,
                                                                 unsigned int width,
                                                                 unsigned int height)
{
    rf::Projection projection(
        glm::radians(preset.camera.vfov),
        static_cast<float>(width) / height,
        preset.camera.z_near,
        preset.camera.z_far);

    if (preset.poseMode == POSE_MODE_ORBITAL) {
        rf::EllipticRotator rotator(
            view_config.viewpoint.rotator.X,
            view_config.viewpoint.rotator.Y,
            view_config.viewpoint.rotator.Z);

        return rf::Viewpoint(
            rf::SphericalCoord(
                glm::radians(preset.spherical.polar),
                glm::radians(preset.spherical.azimuthal)),
            rotator,
            projection);
    }

    return rf::Viewpoint(
        glm::vec3(preset.camera.position[0],
                  preset.camera.position[1],
                  preset.camera.position[2]),
        glm::vec3(preset.camera.look_at[0],
                  preset.camera.look_at[1],
                  preset.camera.look_at[2]),
        glm::vec3(preset.camera.up[0],
                  preset.camera.up[1],
                  preset.camera.up[2]),
        projection);
}

int ViewpointRegistry::init(const ViewConfig3D &view_config, unsigned int width, unsigned int height)
{
    _viewpoints.clear();
    _viewpoints.reserve(view_config.viewpoints_count);

    for (uint32_t i = 0; i < view_config.viewpoints_count; ++i) {
        auto viewpoint = create_viewpoint(view_config, view_config.viewpoint_presets[i], width, height);
        if (!viewpoint) {
            return -1;
        }

        _viewpoints.push_back(*viewpoint);
    }

    return 0;
}

} // namespace engine::view
