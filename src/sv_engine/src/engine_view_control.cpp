#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/printf.h>

#include <engine/engine.hpp>

#include "engine_internal.hpp"

using namespace engine;

engine::Error Engine::input_event(const Output *output, InputEvent *event)
{
    for (int viewIndex = 0; viewIndex < output->views_count; ++viewIndex) {
        const int active_view = output->active_view_ids[viewIndex];

        switch (active_view) {
        case view::SV_VIEW_3D:
            if (_impl->view_3d) {
                _impl->view_3d->handle_event(output, event);
            }
            break;
        default:
            SPDLOG_ERROR("{}", fmt::sprintf("Unknown view type %d", active_view));
            return ERROR;
        }
    }

    return OK;
}

engine::Error Engine::view_animate(int view, int viewpoint)
{
    switch (view) {
    case view::SV_VIEW_3D:
        if (_impl->view_3d) {
            _impl->view_3d->handle_viewpoint(viewpoint);
        }
        break;
    default:
        SPDLOG_ERROR("{}", fmt::sprintf("Unknown view type %d", view));
        return ERROR;
    }

    return OK;
}

engine::Error Engine::get_viewpoint(int view, int *viewpoint)
{
    switch (view) {
    case view::SV_VIEW_3D:
        if (_impl->view_3d) {
            *viewpoint = _impl->view_3d->get_current_viewpoint();
        }
        break;
    default:
        SPDLOG_ERROR("{}", fmt::sprintf("Unknown view type %d", view));
        return ERROR;
    }

    return OK;
}

engine::Error Engine::get_animation_status(int view, int *is_active, unsigned int *timeleft_ms)
{
    *is_active = 0;

    switch (view) {
    case view::SV_VIEW_3D:
        if (_impl->view_3d && _impl->view_3d->is_animation_active(*timeleft_ms)) {
            *is_active = 1;
        }
        break;
    default:
        SPDLOG_ERROR("{}", fmt::sprintf("Unsupported view type %d", view));
        return ERROR;
    }

    return OK;
}
