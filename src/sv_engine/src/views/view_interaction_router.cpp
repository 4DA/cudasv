#include "views/view_interaction_router.hpp"

#include <algorithm>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/printf.h>

#include <world.hpp>

#include <rf/renderer/primitive_component.hpp>
#include <rf/renderer/scene.hpp>
#include <rf/renderer/virtual_camera.hpp>

#include "views/view_navigation_state.hpp"

namespace engine::view
{

void ViewInteractionRouter::init(World *world,
                                 const VirtualControlConfig *controls_config,
                                 rf::VirtualCamera *virtual_camera,
                                 ViewNavigationState *navigation_state)
{
    _world = world;
    _controlsConfig = controls_config;
    _virtualCamera = virtual_camera;
    _navigationState = navigation_state;
}

int ViewInteractionRouter::handle_event(const Output *,
                                        InputEvent *event,
                                        const std::vector<rf::PrimitiveComponent *> &viewpoint_controls,
                                        const std::function<int(int)> &on_viewpoint_selected)
{
    switch (event->type) {
    case DOUBLE_TAP:
        return on_viewpoint_selected(_navigationState->current_viewpoint());
    case PAN:
    {
        auto *active_controller = _navigationState->active_controller();
        if (!active_controller) {
            break;
        }

        active_controller->handle_cursor(-event->y, event->x,
                                         *_virtualCamera,
                                         _navigationState->rotator(),
                                         _navigationState->camera_spherical_angle());
    }
    break;
    case TAP:
        return handle_ray_intersect(viewpoint_controls, on_viewpoint_selected, event);
    case PINCH_ZOOM:
    {
        auto *active_controller = _navigationState->active_controller();
        if (!active_controller) {
            break;
        }

        if (active_controller == _navigationState->topview_controller()) {
            active_controller->handle_scroll(event->scale,
                                             _navigationState->camera_spherical_angle(),
                                             *_virtualCamera,
                                             _navigationState->rotator());
            break;
        }

        _navigationState->orbit_controller()->handle_scroll(
            event->scale,
            _navigationState->camera_spherical_angle(),
            *_virtualCamera,
            _navigationState->rotator());
    }
    break;
    }

    return 0;
}

int ViewInteractionRouter::handle_ray_intersect(
    const std::vector<rf::PrimitiveComponent *> &viewpoint_controls,
    const std::function<int(int)> &on_viewpoint_selected,
    InputEvent *event)
{
    glm::vec2 screen_ray((event->x + 1.0) / 2.0f, (event->y + 1.0) / 2.0f);

    glm::vec3 ray = glm::normalize(_virtualCamera->get_ray(screen_ray));
    std::vector<rf::RayIntersectionResult> result =
        _world->scene().ray_intersect(_virtualCamera->transform.translation, ray, _virtualCamera);

    if (!result.empty() &&
        result[0].object == _navigationState->current_viewpoint_component()) {
        result.erase(result.begin());
    }

    SPDLOG_INFO("{}", fmt::sprintf("pix[%.2f, %.2f] -> query[%f, %f, %f] results, cnt = %zu",
                                   screen_ray.x, screen_ray.y,
                                   ray.x, ray.y, ray.z,
                                   result.size()));

    if (result.empty()) {
        return 0;
    }

    for (const rf::RayIntersectionResult &res: result) {
        SPDLOG_INFO("intersection: name = {} | t = {}",
                    res.object->name.c_str(),
                    res.t);
    }

    auto it = std::find(viewpoint_controls.begin(), viewpoint_controls.end(), result[0].object);

    if (it == viewpoint_controls.end()) {
        SPDLOG_ERROR("{}", fmt::sprintf("viewpoint control[id = %s] is not registered",
                                        result[0].object->name.c_str()));
        return -1;
    }

    unsigned int idx = std::distance(viewpoint_controls.begin(), it);

    if (idx < _controlsConfig->controls_count) {
        SPDLOG_TRACE("{}", fmt::sprintf("controls_config->controls[idx].viewpoint %d %d",
                                        idx,
                                        _controlsConfig->controls[idx].viewpoint));
        return on_viewpoint_selected(_controlsConfig->controls[idx].viewpoint);
    }

    return 1;
}

} // namespace engine::view
