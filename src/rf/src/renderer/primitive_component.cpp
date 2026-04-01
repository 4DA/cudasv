#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/printf.h>

#include "primitive_component.hpp"
#include "virtual_camera.hpp"

using namespace rf;

const glm::vec4 FRONT_FACE_DEFAULT_NORMAL(1.0, 0.0, 0.0, 0.0);

static BoundingBox get_bounding_box(const std::vector<std::shared_ptr<Primitive>> primitives)
{
    BoundingBox result;

    for (const auto &primitive: primitives) {
        result = result.union_with(primitive->bounds);
    }

    return result;
}

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
    glm::vec3 N = glm::vec3(FRONT_FACE_DEFAULT_NORMAL) * toLocal.rotation;
    glm::vec3 v2c = glm::normalize((camera.transform.translation - toLocal.translation));
    float angle = glm::angle(N, v2c);
    glm::vec3 axis = glm::normalize(glm::cross(N, v2c));
    return glm::angleAxis(angle, axis);
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

    bounds = get_bounding_box(_primitiveGroup._primitives).transform(toWorld);

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
