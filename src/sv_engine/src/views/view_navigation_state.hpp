#ifndef SV_VIEW_NAVIGATION_STATE_HPP
#define SV_VIEW_NAVIGATION_STATE_HPP

#include <memory>

#include <engine/views_config.hpp>

#include <rf/camera_control/camera_control.hpp>
#include <rf/renderer/primitive_component.hpp>

namespace engine::view
{

class ViewNavigationState
{
public:
    int init(const ViewConfig3D &view_config);

    rf::EllipticRotator &rotator() { return *_rotator; }
    const rf::EllipticRotator &rotator() const { return *_rotator; }

    rf::SphericalCoord &camera_spherical_angle() { return _cameraSphericalAngle; }
    const rf::SphericalCoord &camera_spherical_angle() const { return _cameraSphericalAngle; }

    rf::ICameraController *active_controller() const { return _activeController; }
    rf::OrbitCameraController *orbit_controller() const { return _orbitController.get(); }
    rf::TopviewCameraController *topview_controller() const { return _topviewController.get(); }

    void set_active_controller(NavigationMode navigation_mode);
    void clear_active_controller() { _activeController = nullptr; }

    int current_viewpoint() const { return _currentViewpoint; }
    void set_current_viewpoint(int viewpoint) { _currentViewpoint = viewpoint; }

    rf::PrimitiveComponent *current_viewpoint_component() const { return _currentViewpointCompo; }
    void set_current_viewpoint_component(rf::PrimitiveComponent *component)
    {
        _currentViewpointCompo = component;
    }

private:
    std::unique_ptr<rf::EllipticRotator> _rotator;

    rf::ICameraController *_activeController = nullptr;
    std::unique_ptr<rf::OrbitCameraController> _orbitController;
    rf::SphericalBoundary _sphericalBoundary;
    std::unique_ptr<rf::TopviewCameraController> _topviewController;
    rf::RectangularBoundary _rectangularBoundary;

    rf::SphericalCoord _cameraSphericalAngle;
    int _currentViewpoint = 0;
    rf::PrimitiveComponent *_currentViewpointCompo = nullptr;
};

} // namespace engine::view

#endif
