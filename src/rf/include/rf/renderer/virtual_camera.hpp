#ifndef RF_VIRTUAL_CAMERA_HPP
#define RF_VIRTUAL_CAMERA_HPP

#include <rf/renderer/glm_common.hpp>
#include <rf/renderer/trs_transform.hpp>

namespace rf
{

// Ortopgrahic projection parameters
//
struct OrtographicProjection {
 	float  	left;   //left plane
    float  	right;  // right palne
    float  	bottom; // bottom plane
    float  	top;    // top plane
    float  	zNear;  // near plane
    float  	zFar;   // far plane

    OrtographicProjection() = default;
    OrtographicProjection(float left, float right, float bottom, float top, float zNear, float zFar):
        left(left), right(right), bottom(bottom), top(top), zNear(zNear), zFar(zFar) {}

    // @retuns 4x4 matrix projection matrix
    //
    glm::mat4 get_matrix() const;
};

// Perspective projection parameters
//
struct PerspectiveProjection {
    float fov_y;  // field of view in vertical direction
    float aspect; // fov_x / fov_y factor
    float near;   // near clipping plane
    float far;    // far clipping plane

    PerspectiveProjection() = default;
    PerspectiveProjection(float fov_y, float aspect, float near, float far):
        fov_y(fov_y), aspect(aspect), near(near), far(far) {}

    // @retuns 4x4 matrix projection matrix
    //
    glm::mat4 get_matrix() const;
};

// wrapper class for interpolation between projections of same type
//
struct Projection {
    // TODO: use std::any here
    PerspectiveProjection perspective;
    OrtographicProjection orto;
    bool is_perspective;

    Projection(float fov_y, float aspect, float near, float far):
        perspective(fov_y, aspect, near, far), is_perspective(true) {}

    Projection(float left, float right, float bottom, float top, float zNear, float zFar):
        orto(left, right, bottom, top, zNear, zFar),
        is_perspective(false) {}

    glm::mat4 get_matrix() const;

    static Projection mix(const Projection &p1, const Projection &p2, float t);
};

// virtual camera stores all data necessary for supplying projection and
// camera matrices
struct VirtualCamera {
    static const glm::vec3 DEFAULT_CAMERA_UP;
    static const glm::vec3 DEFAULT_CAMERA_FORWARD;

    // projection params
    //
    Projection m_projection;

    // opengl->world transform
    //
    TRSTransform transform;

    mutable glm::mat4 projection_matrix;
    mutable glm::mat4 camera_matrix;

    float exposure;

    VirtualCamera(const Projection &projection,
                  TRSTransform transform,
                  float exposure);

    const Projection & get_projection() {return m_projection;}
    void set_projection(const Projection &new_projection);

    glm::vec3 get_up_vector() const {return glm::rotate(transform.rotation, DEFAULT_CAMERA_UP);}


    static glm::vec3 get_up_vector(glm::quat rotation) {return glm::rotate(rotation, DEFAULT_CAMERA_UP);}

    void get_virtual_image_plane_params(glm::vec3 &horizontal,
                                        glm::vec3 &vertical,
                                        glm::vec3 &lower_left) const;

    // get ray in world space
    glm::vec3 get_ray(glm::vec2 screen_ray) const;
};

// compute transform that takes opengl camera to world
//
TRSTransform get_look_at_trs(const glm::vec3 &pos,
                             const glm::vec3 &lookAt,
                             const glm::vec3 &worldUp);

// compute transform from from one camera view direction to another
//
TRSTransform get_look_at_trs(const glm::vec3 &pos,
                             const glm::vec3 &lookAt,
                             const glm::vec3 &worldUp,
                             const glm::vec3 &originalForward,
                             const glm::vec3 &originalUp);

// get camera 4x4 matrix from transform
glm::mat4 get_camera_matrix(const TRSTransform transform);

} // namespace rf

#endif
