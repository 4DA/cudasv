#include "app/glfw_host.hpp"

#include <cmath>
#include <cstdlib>

#include <GLES3/gl3.h>

#include <glm/glm.hpp>

#include <spdlog/spdlog.h>

namespace
{

void window_to_ndc(int x, int y, int width, int height, float &out_x, float &out_y)
{
    out_x = (static_cast<float>(x) / width) * 2.0f - 1.0f;
    out_y = (static_cast<float>(y) / height) * 2.0f - 1.0f;
    out_y = -out_y;
}

} // namespace

namespace svapp
{

void GLFWHost::error_callback(int, const char *description)
{
    SPDLOG_ERROR("Error: {}", description);
}

GLFWHost::GLFWHost(AppContext *app, engine::OutputSet *output_set, int outputs_number):
    _app(app),
    _output_set(output_set),
    _outputs_number(outputs_number)
{
    if (!glfwInit()) {
        SPDLOG_ERROR("glfwInit() failed");
        std::exit(EXIT_FAILURE);
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_SAMPLES, 0);

    for (int index = 0; index < _outputs_number; ++index) {
        if (!_output_set->outputs[index].active) {
            continue;
        }

        const std::string window_name = _output_set->outputs[index].name;

        window[index] = glfwCreateWindow(
            _output_set->outputs[index].config.display_width,
            _output_set->outputs[index].config.display_height,
            window_name.c_str(),
            nullptr,
            index > 0 ? window[0] : nullptr);

        if (!window[index]) {
            SPDLOG_ERROR("GLFW can't create window `{}`", window_name);
            glfwTerminate();
            std::exit(EXIT_FAILURE);
        }

        glfwSetWindowUserPointer(window[index], this);
        glfwSetKeyCallback(window[index],
                           [](GLFWwindow *w, int key, int scancode, int action, int mods) {
                               auto *self = static_cast<GLFWHost *>(glfwGetWindowUserPointer(w));
                               self->key_callback(w, key, scancode, action, mods);
                           });
        glfwSetScrollCallback(window[index],
                              [](GLFWwindow *w, double xoffset, double yoffset) {
                                  auto *self = static_cast<GLFWHost *>(glfwGetWindowUserPointer(w));
                                  self->scroll_callback(w, xoffset, yoffset);
                              });
        glfwSetMouseButtonCallback(window[index],
                                   [](GLFWwindow *w, int button, int action, int mods) {
                                       auto *self = static_cast<GLFWHost *>(glfwGetWindowUserPointer(w));
                                       self->mouse_button_callback(w, button, action, mods);
                                   });
        glfwSetCursorPosCallback(window[index],
                                 [](GLFWwindow *w, double xpos, double ypos) {
                                     auto *self = static_cast<GLFWHost *>(glfwGetWindowUserPointer(w));
                                     self->cursor_position_callback(w, xpos, ypos);
                                 });

        glfwMakeContextCurrent(window[index]);
        glfwSwapInterval(1);
        glfwSetErrorCallback(error_callback);
    }
}

GLFWHost::~GLFWHost()
{
    for (int index = 0; index < _outputs_number; ++index) {
        if (window[index]) {
            glfwDestroyWindow(window[index]);
        }
    }

    glfwTerminate();
}

void GLFWHost::make_current(int index)
{
    glfwMakeContextCurrent(window[index]);
}

void GLFWHost::swap_buffers(int index)
{
    glfwSwapBuffers(window[index]);
    glfwPollEvents();
}

bool GLFWHost::should_close_any() const
{
    for (int index = 0; index < _outputs_number; ++index) {
        if (window[index] && glfwWindowShouldClose(window[index])) {
            return true;
        }
    }

    return false;
}

void GLFWHost::key_callback(GLFWwindow *window, int key, int, int action, int)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
        _app->running = false;
    }
}

void GLFWHost::cursor_position_callback(GLFWwindow *window, double xpos, double ypos)
{
    if (!_app->interactive_input_enabled) {
        return;
    }

    int width = 0;
    int height = 0;
    glfwGetWindowSize(window, &width, &height);

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) != GLFW_PRESS) {
        return;
    }

    float new_xpos = 0.0f;
    float new_ypos = 0.0f;
    window_to_ndc(xpos, ypos, width, height, new_xpos, new_ypos);

    glm::vec2 delta(-new_xpos + _app->cursor_x, -new_ypos + _app->cursor_y);

    _app->cursor_x = new_xpos;
    _app->cursor_y = new_ypos;

    engine::InputEvent event;
    event.type = engine::PAN;
    event.x = delta.x;
    event.y = delta.y;
    event.scale = 0.0f;

    float angle = -glm::degrees(atanf(event.x / event.y));
    if (isnanf(angle)) {
        return;
    }

    if (event.y > 0) {
        angle += 180.0f;
    }

    if (fabsf(event.y) < 0.0001f) {
        angle = event.x < 0.0f ? -90.0f : 90.0f;
    }

    pthread_mutex_lock(&_app->access);

    int viewpoint = 0;
    if (_app->engine->get_viewpoint(engine::view::SV_VIEW_3D, &viewpoint)) {
        pthread_mutex_unlock(&_app->access);
        return;
    }

    if ((viewpoint > 3) && (viewpoint < 7) &&
        ((fabsf(event.x) > 0.1f) || (fabsf(event.y) > 0.1f))) {
        int is_active = 0;
        unsigned int timeleft = 0;

        if (_app->engine->get_animation_status(engine::view::SV_VIEW_3D, &is_active, &timeleft)) {
            pthread_mutex_unlock(&_app->access);
            return;
        }

        if (!is_active) {
            _app->engine->config.views_config.view_3d.viewpoint_presets[7] =
                _app->engine->config.views_config.view_3d.viewpoint_presets[0];
            _app->engine->config.views_config.view_3d.viewpoint_presets[7].spherical.polar = 45.0f;
            _app->engine->config.views_config.view_3d.viewpoint_presets[7].spherical.azimuthal =
                angle + 180.0f;

            _app->engine->view_animate(engine::view::SV_VIEW_3D, 7);
            _app->hide_current_viewpoint = false;
        }
    } else {
        _app->engine->input_event(&_output_set->outputs[0], &event);
        _app->hide_current_viewpoint = false;
    }

    pthread_mutex_unlock(&_app->access);
}

void GLFWHost::mouse_button_callback(GLFWwindow *window, int button, int action, int)
{
    if (!_app->interactive_input_enabled) {
        return;
    }

    double xpos = 0.0;
    double ypos = 0.0;
    int width = 0;
    int height = 0;

    glfwGetWindowSize(window, &width, &height);
    glfwGetCursorPos(window, &xpos, &ypos);

    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        window_to_ndc(xpos, ypos, width, height, _app->cursor_x, _app->cursor_y);
    }

    if (button != GLFW_MOUSE_BUTTON_RIGHT || action != GLFW_PRESS) {
        return;
    }

    engine::InputEvent event;
    event.type = engine::TAP;
    event.scale = 1.0f;
    window_to_ndc(xpos, ypos, width, height, event.x, event.y);

    int viewpoint_before = 0;
    int viewpoint_after = 0;

    _app->engine->get_viewpoint(engine::view::SV_VIEW_3D, &viewpoint_before);
    _app->engine->input_event(&_output_set->outputs[0], &event);
    _app->engine->get_viewpoint(engine::view::SV_VIEW_3D, &viewpoint_after);

    if (viewpoint_before != viewpoint_after) {
        _app->hide_current_viewpoint = true;
    }
}

void GLFWHost::scroll_callback(GLFWwindow *, double, double yoffset)
{
    if (!_app->interactive_input_enabled) {
        return;
    }

    engine::InputEvent event;
    event.type = engine::PINCH_ZOOM;
    event.scale = yoffset < 0.0f ? 1.05f : 1.0f / 1.05f;

    _app->engine->input_event(&_output_set->outputs[0], &event);
}

} // namespace svapp
