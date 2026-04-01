#ifndef RF_SCENE_COMPONENT_HPP
#define RF_SCENE_COMPONENT_HPP
#include <atomic>
#include <string>

#include <rf/renderer/trs_transform.hpp>
#include "bounding_box.hpp"
#include <rf/renderer/virtual_camera.hpp>

namespace rf
{

struct SceneComponent;


/// Scene Component is base class for entity that has position/orintation and can be added
/// to the scene
struct SceneComponent {
    const std::string name;

    SceneComponent *parent; /// null if this is root

    TRSTransform toLocal; /// local space transform for children and geometry

    TRSTransform toWorld; /// transform to world space

    std::vector<SceneComponent *> children;

    BoundingBox bounds; /// bounding box (including children)

    static std::atomic_size_t next_id;
    const std::size_t id;

    /// @param[in] name Name of component. Must be unique in scene.
    /// @param[in] toLocal Local space transform affecting geometry and children
    /// @param[in] parent Parent, null if this is root component
    SceneComponent(const std::string &name,
                      TRSTransform toLocal = TRSTransform(),
                      SceneComponent *parent = nullptr)
        : name(name), parent(parent),
          toLocal(toLocal),
          toWorld(compute_to_world(parent)),
          id(next_id++) {}

    virtual ~SceneComponent() = default;

    /// @returns world space transform for this component
    ///
    TRSTransform compute_to_world(SceneComponent *_parent) {
        return (_parent ? compose(this->toLocal, _parent->toWorld) : this->toLocal);
    }

    /// recompute global space transform for this component and its children
    ///
    void update_transforms(SceneComponent *parent);

    /// @returns bounding box with children in world space
    ///
    virtual BoundingBox compute_bounding_box();

    virtual BoundingBox compute_bounding_box(const rf::VirtualCamera *camera);

};

} // namespace rf

#endif
