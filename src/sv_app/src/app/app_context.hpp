#pragma once

#include <memory>
#include <pthread.h>

#include <engine/engine.hpp>

namespace svapp
{

struct AppContext
{
    bool running = true;
    bool hide_current_viewpoint = false;
    bool interactive_input_enabled = true;
    float cursor_x = 0.0f;
    float cursor_y = 0.0f;

    pthread_mutex_t access;
    std::unique_ptr<engine::Engine> engine;
};

} // namespace svapp
