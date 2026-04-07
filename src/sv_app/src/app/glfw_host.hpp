#pragma once

#include <string>

#include <GLFW/glfw3.h>

#include <engine/engine.hpp>

#include "app/app_context.hpp"

namespace svapp
{

class GLFWHost
{
public:
    GLFWHost(AppContext *app, engine::OutputSet *output_set, int outputs_number);
    ~GLFWHost();

    void make_current(int index);
    void swap_buffers(int index);
    bool should_close_any() const;
    bool key_pressed(int index, int key) const;
    void set_window_title(int index, const std::string &title) const;

private:
    static void error_callback(int error, const char *description);

    void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods);
    void cursor_position_callback(GLFWwindow *window, double xpos, double ypos);
    void mouse_button_callback(GLFWwindow *window, int button, int action, int mods);
    void scroll_callback(GLFWwindow *window, double xoffset, double yoffset);

    AppContext *_app = nullptr;
    const engine::OutputSet *_output_set = nullptr;
    GLFWwindow *window[SV_MAX_OUTPUTS] = {};
    int _outputs_number = 0;
};

} // namespace svapp
