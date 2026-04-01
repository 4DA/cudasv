#ifndef RF_PRIMITIVE_COMPONENT_HPP
#define RF_PRIMITIVE_COMPONENT_HPP

#include <memory>
#include <cstddef>
#include <type_traits>
#include <map>
#include <ranges>

#include "scene_component.hpp"
#include "bounding_box.hpp"
#include "mesh_geometry.hpp"

#include <rf/renderer/cudarf/cudarf.hpp>

namespace rf
{

// primitive is the smallest (in terms of abstraction) renderable entity
//
struct Primitive {
    const MeshInfo meshInfo;
    BoundingBox bounds;

    // cudarf state, TODO: move to opaque data structure
    const std::shared_ptr<cudarf::Material> cudarfMaterial;
    rf::NaiveMeshPtr cudarfMesh;
    unsigned int drawPacketId;
    // ..

    BoundingBox get_bounds(const TRSTransform &to_world) const;

    Primitive(MeshInfo meshInfo,
              glm::vec3 vertex_min,
              glm::vec3 vertex_max,
              const std::shared_ptr<cudarf::Material> &cudarfMaterial,
              unsigned int drawPacketId) :
        meshInfo(meshInfo),
        bounds(BoundingBox(vertex_min, vertex_max)),
        cudarfMaterial(cudarfMaterial),
        drawPacketId(drawPacketId)
        {}
};


struct VertexTransform {
    glm::mat4 M; // model matrix
    glm::mat4 V; // view matrix
    glm::mat4 P; // projection matrix
    glm::mat4 N; // normal matrix
};

class VirtualCamera;

// primitive component is scene component, that contains renderable data
struct PrimitiveComponent : public SceneComponent {
    // name    - Name of component. Must be unique in scene.
    // toLocal - Local space transform affecting geometry and children
    // parent  - Parent, null if this is root component
    // hitable - True if object accepts camera rays
    PrimitiveComponent(const std::string &name,
                       const TRSTransform &toLocal,
                       SceneComponent *parent,
                       bool hitable = false,
                       bool isFrontFacing = false)
        : SceneComponent(name, toLocal, parent),
          _hitable(hitable),
          _frontFacing(isFrontFacing){}

    // Add primitive to commponent. Must not be called after rendering
    // resources are initialized
    // TODO: rework draw packet ownership/batching.
    Primitive * add_primitive(MeshInfo geometry,
                              glm::vec3 vertexMin,
                              glm::vec3 vertexMax,
                              unsigned int drawPacketId,
                              std::shared_ptr<cudarf::Material> material);

    // shallow copy primitive group
    void setPrimitiveGroupFrom(const PrimitiveComponent &other) {
        _primitiveGroup = other._primitiveGroup;
    }

    // TODO: implement
    // std::unique_ptr<PrimitiveComponent> clone() const

    // return bounding box (with children) in world space
    virtual BoundingBox compute_bounding_box() override;

    // compute bounding box (with children) in world space
    virtual BoundingBox compute_bounding_box(const rf::VirtualCamera *camera) override;

    // return scene_info with necessary information, or default when camera
    // is null
    virtual VertexTransform get_vertex_transform(const rf::VirtualCamera *camera) const;

    auto get_primitives() const { return std::ranges::subrange(_primitiveGroup._primitives); }

    bool is_hitable() const {return _hitable; }

    bool is_front_facing() const { return _frontFacing; }

private:
    glm::quat get_front_face_rotation(const VirtualCamera camera) const;

    struct {
        std::vector<std::shared_ptr<Primitive>> _primitives;
    } _primitiveGroup;

    // if true, component interacts with raycasting
    bool _hitable;

    // if true, component should oriented towards camera
    bool _frontFacing;
};

}

#endif
