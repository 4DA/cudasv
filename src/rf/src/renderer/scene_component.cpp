#include <stack>
#include <spdlog/spdlog.h>

#include "scene_component.hpp"

using namespace rf;

std::atomic_size_t SceneComponent::next_id{0};

void SceneComponent::update_transforms(SceneComponent *parent) {
    this->toWorld = compute_to_world(parent);

    for (SceneComponent *child: children) {
        child->update_transforms(this);
    }
}

BoundingBox SceneComponent::compute_bounding_box(const rf::VirtualCamera *camera)
{
    (void)camera;
    return compute_bounding_box();
}

BoundingBox SceneComponent::compute_bounding_box()
{
    bounds = BoundingBox();

    for (const auto &child: children) {
        bounds = bounds.union_with(child->compute_bounding_box());
    }

    return bounds;
}

