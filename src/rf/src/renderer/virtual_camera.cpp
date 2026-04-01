#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/printf.h>

#include "virtual_camera.hpp"

using namespace rf;

// Default camera orientation
const glm::vec3 VirtualCamera::DEFAULT_CAMERA_FORWARD = glm::vec3(0.0f, 0.0f, -1.0f);
const glm::vec3 VirtualCamera::DEFAULT_CAMERA_UP      = glm::vec3(0.0f, 1.0f, 0.0f);

TRSTransform rf::get_look_at_trs(const glm::vec3 &pos,
                                 const glm::vec3 &lookAt,
                                 const glm::vec3 &worldUp)
{
    return get_look_at_trs(pos, lookAt, worldUp,
                           VirtualCamera::DEFAULT_CAMERA_FORWARD,
                           VirtualCamera::DEFAULT_CAMERA_UP);
}

TRSTransform rf::get_look_at_trs(const glm::vec3 &pos,
                                 const glm::vec3 &lookAt,
                                 const glm::vec3 &worldUp,
                                 const glm::vec3 &originalForward,
                                 const glm::vec3 &originalUp)
{
    glm::vec3 direction = glm::normalize(lookAt - pos);
    glm::vec3 right = glm::normalize(glm::cross(direction, worldUp));
    glm::vec3 desired_up = glm::normalize(glm::cross(right, direction));

    // Rotation from orignal camera direction to requested direction
    glm::quat orig_cam_to_world = glm::rotation(originalForward, direction);

    // Up vector resulting from orig_cam_to_world rotation
    glm::vec3 new_up = glm::rotate(orig_cam_to_world, originalUp);

    // Corrected rotation to bring newUp to desiredUp
    glm::quat corrected_up;

    // Special case when newUp and desiredUp vectors are in opposite
    // directions, so any rotation axis can be used, but problem is direction
    // might be changed, which is what we don't want. Axis that preserves
    // forward vector is ``direction'' itself.
    if (glm::dot(new_up, desired_up) < -1.0f + std::numeric_limits<float>::epsilon())
    {
        corrected_up = glm::angleAxis(glm::pi<float>(), direction);
    }
    else
    {
        corrected_up = glm::normalize(glm::rotation(new_up, desired_up));
    }

    return TRSTransform(pos, corrected_up * orig_cam_to_world);
}

glm::mat4 rf::get_camera_matrix(const TRSTransform transform)
{
    // get quaternion corresponding to *reverse* of opengl -> world rotation
    // and build matrix from it
    glm::mat4 reverse = glm::mat4_cast(glm::conjugate(transform.rotation));

    // build world -> opengl camera matrix by adding *reverse* of camera position
    return glm::translate(reverse, -transform.translation);
}

glm::mat4 OrtographicProjection::get_matrix() const {
    return glm::ortho(left, right, bottom, top, zNear, zFar);
}

glm::mat4 PerspectiveProjection::get_matrix() const {
    return glm::perspective(fov_y, aspect, near, far);
}

glm::mat4 Projection::get_matrix() const {
    if (is_perspective) {
        return perspective.get_matrix();
    } else {
        return orto.get_matrix();
    }
}

Projection Projection::mix(const Projection &p1, const Projection &p2, float t)
{
    if (p1.is_perspective && p2.is_perspective) {
        return Projection(
            glm::mix(p1.perspective.fov_y, p2.perspective.fov_y, t),
            glm::mix(p1.perspective.aspect, p2.perspective.aspect, t),
            glm::mix(p1.perspective.near, p2.perspective.near, t),
            glm::mix(p1.perspective.far, p2.perspective.far, t));
    }
    else if (!p1.is_perspective && !p2.is_perspective) {
        return Projection(
            glm::mix(p1.orto.left, p2.orto.left, t),
            glm::mix(p1.orto.right, p2.orto.right, t),
            glm::mix(p1.orto.bottom, p2.orto.bottom, t),
            glm::mix(p1.orto.top, p2.orto.top, t),
            glm::mix(p1.orto.zNear, p2.orto.zNear, t),
            glm::mix(p1.orto.zFar, p2.orto.zFar, t));
    }
    else {
        assert(false /*not implemented*/);
        // TODO: create special projection subclass to store only matrix?
        // glm::mat4 a = p1.get_matrix();
        // glm::mat4 b = p2.get_matrix();
        // return  a + (b - a) * t;
        return p1;
    }
}

VirtualCamera::VirtualCamera(const Projection &projection, TRSTransform transform, float exposure):
    m_projection(projection),
    transform(transform),
    projection_matrix(projection.get_matrix()),
    exposure(exposure)
{
}

void VirtualCamera::set_projection(const Projection &new_projection)
{
    this->m_projection = new_projection;
    projection_matrix = new_projection.get_matrix();
}

void VirtualCamera::get_virtual_image_plane_params(glm::vec3 &horizontal,
                                                   glm::vec3 &vertical,
                                                   glm::vec3 &lower_left) const
{
    assert(m_projection.is_perspective);

    // we shoot ray towards virtual image plane
    float ipd = 100.0f;    // virtual image plane distance, could be unitary,
                           // but it's harder to debug

    float half_height = ipd * std::tan(m_projection.perspective.fov_y / 2.0f);

    float plane_height = 2.0f * half_height;
    float plane_width = plane_height * m_projection.perspective.aspect;

    glm::vec3 origin = this->transform.translation;
    glm::vec3 lookAt = glm::rotate(this->transform.rotation, glm::vec3(0.0f, 0.0f, -1.0f));

    // compute camera basis: w - to image plane, u - right, v - up
    // camera looks towards -w
    glm::vec3 w = glm::normalize(-lookAt);
    glm::vec3 v = glm::normalize(glm::rotate(this->transform.rotation,
                                             glm::vec3(0.0f, 1.0f, 0.0f)));
    glm::vec3 u = glm::normalize(glm::cross(v, w));

    horizontal = plane_width * u;
    vertical = plane_height * v;

    lower_left = origin - horizontal / 2.0f - vertical / 2.0f - ipd * w;

    SPDLOG_TRACE("{}", fmt::sprintf("cam basis: w[%f, %f, %f], u[%f, %f, %f], v[%f, %f, %f], ll[%f, %f, %f]",
           w.x, w.y, w.z,
           u.x, u.y, u.z,
           v.x, v.y, v.z,
           lower_left.x,
           lower_left.y,
           lower_left.z
        ));
}

glm::vec3 VirtualCamera::get_ray(glm::vec2 screen_ray) const
{
    glm::vec3 origin = this->transform.translation;

    glm::vec3 horizontal;
    glm::vec3 vertical;
    glm::vec3 lower_left;

    get_virtual_image_plane_params(horizontal, vertical, lower_left);

    return lower_left + screen_ray.x * horizontal + screen_ray.y * vertical - origin;
}
