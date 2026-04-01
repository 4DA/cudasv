#include "views/view_navigation_state.hpp"

namespace engine::view
{

int ViewNavigationState::init(const ViewConfig3D &view_config)
{
    _rotator = std::make_unique<rf::EllipticRotator>(view_config.viewpoint.rotator.X,
                                                     view_config.viewpoint.rotator.Y,
                                                     view_config.viewpoint.rotator.Z);

    _sphericalBoundary = rf::SphericalBoundary(
        glm::radians(view_config.viewpoint.boundary.angle_min.polar),
        glm::radians(view_config.viewpoint.boundary.angle_max.polar),
        glm::radians(view_config.viewpoint.boundary.angle_min.azimuthal),
        glm::radians(view_config.viewpoint.boundary.angle_max.azimuthal));

    _rectangularBoundary = rf::RectangularBoundary(
        view_config.topview_limits.top,
        view_config.topview_limits.bottom,
        view_config.topview_limits.left,
        view_config.topview_limits.right);

    _cameraSphericalAngle = rf::SphericalCoord(
        glm::radians(view_config.viewpoint.spherical.polar),
        glm::radians(view_config.viewpoint.spherical.azimuthal));

    _orbitController = std::make_unique<rf::OrbitCameraController>(
        _cameraSphericalAngle,
        _sphericalBoundary,
        view_config.viewpoint.rotator.X,
        view_config.min_rotator_scale,
        view_config.max_rotator_scale);

    _topviewController = std::make_unique<rf::TopviewCameraController>(_rectangularBoundary);

    set_active_controller(view_config.viewpoint.navigationMode);

    return 0;
}

void ViewNavigationState::set_active_controller(NavigationMode navigation_mode)
{
    switch (navigation_mode)
    {
    case NAVIGATION_MODE_TOPVIEW:
        _activeController = _topviewController.get();
        break;
    case NAVIGATION_MODE_ORBITAL:
        _activeController = _orbitController.get();
        break;
    default:
        _activeController = nullptr;
        break;
    }
}

} // namespace engine::view
