#ifndef RF_CAMERA_CONTROL_HPP
#define RF_CAMERA_CONTROL_HPP

#include <chrono>
#include <memory>
#include <functional>

#include <rf/renderer/trs_transform.hpp>
#include <rf/renderer/glm_common.hpp>
#include <rf/renderer/virtual_camera.hpp>
#include "rotator.hpp"

namespace rf
{


// Abstract class for controlling camera via user input
struct ICameraController
{
public:
    using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock, std::chrono::nanoseconds>;

    // Handle mouse cursor/touchscreen change
    virtual void handle_cursor(float xdiff,
                               float ydiff,
                               VirtualCamera &camera,
                               EllipticRotator &rotator,
                               SphericalCoord &camera_pos) = 0;

    // Handle touchscreen pinch-zoom or mouse scroll change
    virtual void handle_scroll(float scroll,
                               const SphericalCoord &camera_pos,
                               VirtualCamera &camera,
                               EllipticRotator &rotator) = 0;

    // returns true if update need to be called
    virtual bool update_needed() const = 0;


// Update camera when no user input is happening and camera is moving freely
    virtual void update(VirtualCamera &camera,
                        SphericalCoord &camera_pos,
                        EllipticRotator &rotator) = 0;
};

// Camera controller that moves camera in XY plane
struct TopviewCameraController: public ICameraController
{
    TimePoint cursor_timestamp;

    glm::vec3 source = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 target = glm::vec3(0.0f, 0.0f, 0.0f);

    const RectangularBoundary &bounds;

    // Handle mouse cursor/touchscreen change
    virtual void handle_cursor(float xdiff,
                               float ydiff,
                               VirtualCamera &camera,
                               EllipticRotator &rotator,
                               SphericalCoord &camera_pos) override;


    // Change zoom level in topview position
    void handle_scroll(float scroll,
                       const SphericalCoord &camera_pos,
                       VirtualCamera &camera,
                       EllipticRotator &rotator) override;

    // returns true if update need to be called
    virtual bool update_needed() const override;


   // Update camera when no user input is happening and camera is moving freely
    virtual void update(VirtualCamera &camera,
                        SphericalCoord &camera_pos,
                        EllipticRotator &rotator) override;

    TopviewCameraController(const RectangularBoundary &bounds);

};

// Camera controller that moves camera along ellipsoid arc
struct OrbitCameraController: public ICameraController
{
    SphericalCoord sourcePos;
    SphericalCoord targetPos;
    const SphericalBoundary &bounds;
    float defaultRotatorScale;
    float minRotatorScale;
    float maxRotatorScale;

    TimePoint cursorTimestamp;
    TimePoint scrollTimestamp;

    EllipticRotator sourceRotator = EllipticRotator(0.0f, 0.0f, 0.0f);
    EllipticRotator desiredRotator = EllipticRotator(0.0f, 0.0f, 0.0f);


    // change zoom level in topview position
    OrbitCameraController(const SphericalCoord &camera_pos,
                          const SphericalBoundary &bounds,
                          float defaultEllipsoidScale,
                          float minEllipsoidScale,
                          float maxEllipsoidScale);


    // rotate camera over ellipsoid arc
    void handle_cursor(float xdiff,
                       float ydiff,
                       VirtualCamera &camera,
                       EllipticRotator &rotator,
                       SphericalCoord &camera_pos) override;


    // scale ellipsoid uniformly
    void handle_scroll(float scroll,
                       const SphericalCoord &cameraPos,
                       VirtualCamera &camera,
                       EllipticRotator &rotator) override;

    bool update_needed() const override;

    // Update camera during free flight
    void update(VirtualCamera &camera,
                SphericalCoord &camera_pos,
                EllipticRotator &rotator) override;
};

} //namespace rf

#endif
