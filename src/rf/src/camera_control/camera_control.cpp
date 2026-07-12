#include <chrono>
#include <glm/common.hpp>
#include <spdlog/spdlog.h>

#include "camera_control.hpp"

using namespace std::chrono;

const float INERTIA_TIME = 0.5f;
const float SCROLL_POW_COEF = 4.0f;

rf::OrbitCameraController::OrbitCameraController(const SphericalCoord &camera_pos,
                                                 const SphericalBoundary &bounds,
                                                 float default_ellipsoid_scale,
                                                 float min_rotator_scale,
                                                 float max_rotator_scale):
    sourcePos(camera_pos),
    targetPos(camera_pos),
    bounds(bounds),
    defaultRotatorScale(default_ellipsoid_scale),
    minRotatorScale(min_rotator_scale),
    maxRotatorScale(max_rotator_scale)
{}

void rf::OrbitCameraController::handle_cursor(float xpos,
                                              float ypos,
                                              VirtualCamera &camera,
                                              EllipticRotator &rotator,
                                              SphericalCoord &camera_pos)
{
    const unsigned int RATIO = 10;

    SphericalCoord delta (glm::clamp(xpos * 100 / 180.0f * glm::pi<float>(), -0.025f, 0.025f),
                          glm::clamp(ypos * 100 / 180.0f * glm::pi<float>(), -0.025f, 0.025f));

    if (bounds.point_inside(camera_pos + delta)) {
        camera_pos = camera_pos + delta;
        sourcePos = camera_pos;
        targetPos = camera_pos + delta * RATIO;
        camera.transform = rotator.get_orientation(camera_pos);
    }

    cursorTimestamp = std::chrono::high_resolution_clock::now();
}

void rf::OrbitCameraController::handle_scroll(float scroll,
                                              const SphericalCoord &,
                                              VirtualCamera &,
                                              EllipticRotator &rotator)
{
    float current_scale = rotator.A / defaultRotatorScale;

    if (current_scale * scroll < minRotatorScale ||
        current_scale * scroll > maxRotatorScale)
    {
        return;
    }

    sourceRotator = rotator;
    desiredRotator = rotator;
    desiredRotator.scale(std::pow(scroll, SCROLL_POW_COEF));
    scrollTimestamp = std::chrono::high_resolution_clock::now();
}

bool rf::OrbitCameraController::update_needed() const
{
    auto now = std::chrono::high_resolution_clock::now();

    float cursor_time = std::chrono::duration_cast<std::chrono::microseconds>
        (now - cursorTimestamp).count() * 1e-6f;

    float scroll_time = std::chrono::duration_cast<std::chrono::microseconds>
        (now - scrollTimestamp).count() * 1e-6f;

    return (cursor_time < INERTIA_TIME) || (scroll_time < INERTIA_TIME);
}

void rf::OrbitCameraController::update(VirtualCamera &camera,
                                       SphericalCoord &camera_pos,
                                       EllipticRotator &rotator)
{
    auto now = std::chrono::high_resolution_clock::now();

    float release_time = std::chrono::duration_cast<std::chrono::microseconds>
        (now - cursorTimestamp).count() * 1e-6f;

    if (release_time < INERTIA_TIME) {
        float t = release_time / INERTIA_TIME;

        SphericalCoord target = SphericalCoord::interpolate(sourcePos, targetPos, t);

        if (bounds.point_inside(target)) {
            camera_pos = target;
        }
    }

    release_time = std::chrono::duration_cast<std::chrono::microseconds>
        (now - scrollTimestamp).count() * 1e-6f;

    if (release_time < INERTIA_TIME) {
        float t = release_time / INERTIA_TIME;

        t = glm::smoothstep(0.0f, 1.0f, t);

        float new_scale = glm::mix(sourceRotator.A, desiredRotator.A, t) / defaultRotatorScale;

        if (!(new_scale < minRotatorScale || new_scale > maxRotatorScale)) {
            rotator.A = glm::mix(sourceRotator.A, desiredRotator.A, t);
            rotator.B = glm::mix(sourceRotator.B, desiredRotator.B, t);
            rotator.C = glm::mix(sourceRotator.C, desiredRotator.C, t);
        }
    }

    camera.transform = rotator.get_orientation(camera_pos);
}

rf::TopviewCameraController::TopviewCameraController(const RectangularBoundary &bounds)
    : bounds(bounds) {}

void
rf::TopviewCameraController::handle_cursor(float xdiff,
                                           float ydiff,
                                           VirtualCamera &camera,
                                           EllipticRotator &,
                                           SphericalCoord &)
{
    glm::vec3 up = glm::rotate(camera.transform.rotation,
                               VirtualCamera::DEFAULT_CAMERA_UP);

    glm::vec3 dir(1000.0f * glm::clamp(-xdiff, -0.01f, 0.01f),
                  1000.0f * glm::clamp(-ydiff, -0.01f, 0.01f),
                  0.0f);

    if (std::abs(up.x + 1.0f) <= 2.0f * std::numeric_limits<float>::epsilon()) {
        dir *= -1.0f;
    }

    glm::vec3 new_target = camera.transform.translation + dir * 2.5f;

    if (bounds.is_inside(new_target)) {
        camera.transform.translation = new_target;
    }

    cursor_timestamp = std::chrono::high_resolution_clock::now();
}

void rf::TopviewCameraController::handle_scroll(float scroll,
                                                const SphericalCoord &,
                                                VirtualCamera &camera,
                                                EllipticRotator &)
{
    float start_scale = 300.0f;
    scroll = (scroll > 1.0f ? scroll : -scroll);

    source = camera.transform.translation;
    source.z += start_scale * scroll;
    camera.transform.translation = source;
    cursor_timestamp = std::chrono::high_resolution_clock::now();
}

bool rf::TopviewCameraController::update_needed() const
{
    return false;
}

void rf::TopviewCameraController::update(VirtualCamera &,
                                         SphericalCoord &,
                                         EllipticRotator &)
{
}

