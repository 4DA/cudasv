#include <spdlog/spdlog.h>

#include "primitive_component.hpp"
#include "virtual_camera.hpp"

using namespace rf;

namespace
{

const glm::vec3 FRONT_FACE_DEFAULT_NORMAL(1.0f, 0.0f, 0.0f);
const glm::vec3 FRONT_FACE_DEFAULT_UP(0.0f, 1.0f, 0.0f);

static BoundingBox get_bounding_box(const std::vector<std::shared_ptr<Primitive>> primitives)
{
    BoundingBox result;

    for (const auto &primitive: primitives) {
        result = result.union_with(primitive->bounds);
    }

    return result;
}

// remove the part of a vector that points along a plane normal, leaving only
// the part that lies inside the plane
static glm::vec3 project_to_plane(glm::vec3 value, const glm::vec3 &planeNormal)
{
    return glm::normalize(value - planeNormal * glm::dot(value, planeNormal));
}

} // namespace

BoundingBox Primitive::get_bounds(const TRSTransform &to_world) const {
    return bounds.transform(to_world);
}

Primitive * PrimitiveComponent::add_primitive(MeshInfo geometry,
                                              glm::vec3 vertexMin,
                                              glm::vec3 vertexMax,
                                              unsigned int drawPacketId,
                                              std::shared_ptr<cudarf::Material> material)
{
    _primitiveGroup._primitives.push_back(
        std::make_shared<Primitive>(geometry, vertexMin, vertexMax,
                                    material, drawPacketId));

    return _primitiveGroup._primitives.back().get();
}

glm::quat PrimitiveComponent::get_front_face_rotation(const VirtualCamera camera) const
{
    const glm::quat originalRotation = glm::normalize(toWorld.rotation);
    const glm::quat cameraRotation = glm::normalize(camera.transform.rotation);

    glm::vec3 currentFront = glm::rotate(originalRotation, FRONT_FACE_DEFAULT_NORMAL);
    glm::vec3 targetFront = -glm::rotate(cameraRotation, VirtualCamera::DEFAULT_CAMERA_FORWARD);

    glm::quat faceCamera = glm::rotation(currentFront, targetFront);
    glm::quat result = glm::normalize(faceCamera * originalRotation);

    glm::vec3 currentUp = project_to_plane(glm::rotate(result, FRONT_FACE_DEFAULT_UP), targetFront);
    glm::vec3 targetUp = project_to_plane(glm::rotate(cameraRotation, VirtualCamera::DEFAULT_CAMERA_UP), targetFront);
    glm::quat restoreUp = glm::rotation(currentUp, targetUp);

    return glm::normalize(restoreUp * result);
}

BoundingBox PrimitiveComponent::compute_bounding_box()
{
    bounds = get_bounding_box(_primitiveGroup._primitives).transform(toWorld);

    for (const auto &child: children) {
        bounds = bounds.union_with(child->compute_bounding_box());
    }

    return bounds;
}

BoundingBox PrimitiveComponent::compute_bounding_box(const rf::VirtualCamera *camera)
{
    assert(camera);

    TRSTransform worldTransform;

    if (_frontFacing) {
        worldTransform.rotation = get_front_face_rotation(*camera);
    } else {
        worldTransform = toWorld;
    }

    bounds = get_bounding_box(_primitiveGroup._primitives).transform(worldTransform);

    for (const auto &child: children) {
        bounds = bounds.union_with(child->compute_bounding_box(camera));
    }

    return bounds;
}

VertexTransform PrimitiveComponent::get_vertex_transform(const rf::VirtualCamera *camera) const
{
    assert(camera);

    glm::mat4 M;

    if (_frontFacing) {
        TRSTransform ffToWorld = this->toWorld;
        ffToWorld.rotation = get_front_face_rotation(*camera);
        M = to_matrix(ffToWorld);
    } else {
        M = to_matrix(toWorld);
    }

    return VertexTransform {
        M,
        get_camera_matrix(camera->transform),
        camera->projection_matrix,
        glm::mat4(glm::transpose(glm::inverse(glm::mat3(M))))
    };
}
