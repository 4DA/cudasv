#ifndef SV_VIEW_INTERACTION_ROUTER_HPP
#define SV_VIEW_INTERACTION_ROUTER_HPP

#include <functional>
#include <vector>

#include <engine/engine.hpp>

namespace rf
{
class PrimitiveComponent;
class VirtualCamera;
struct RayIntersectionResult;
}

namespace engine
{
class World;
}

namespace engine::view
{

class ViewNavigationState;

class ViewInteractionRouter
{
public:
    void init(World *world,
              const VirtualControlConfig *controls_config,
              rf::VirtualCamera *virtual_camera,
              ViewNavigationState *navigation_state);

    int handle_event(const Output *output,
                     InputEvent *event,
                     const std::vector<rf::PrimitiveComponent *> &viewpoint_controls,
                     const std::function<int(int)> &on_viewpoint_selected);

private:
    int handle_ray_intersect(const std::vector<rf::PrimitiveComponent *> &viewpoint_controls,
                             const std::function<int(int)> &on_viewpoint_selected,
                             InputEvent *event);

    World *_world = nullptr;
    const VirtualControlConfig *_controlsConfig = nullptr;
    rf::VirtualCamera *_virtualCamera = nullptr;
    ViewNavigationState *_navigationState = nullptr;
};

} // namespace engine::view

#endif
