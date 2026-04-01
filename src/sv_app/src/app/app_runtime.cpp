#include "app/app_runtime.hpp"

#include <unistd.h>

#include <GLFW/glfw3.h>
#include <GLES3/gl3.h>

#include <spdlog/spdlog.h>

#include "app/demo_vehicle_signals.hpp"

namespace svapp
{

int run_application_loop(AppContext &app,
                         GLFWHost &glfw_host,
                         const CmdlineOpts &options,
                         engine::OutputSet output_set,
                         videoio::FrameSource &frame_source)
{
    videoio::FramePacket frame_packets[2];
    int frame_set_index = 0;
    vehicle::CANSignals demo_state;

    while (app.running && !glfw_host.should_close_any()) {
        update_demo_vehicle_signals(demo_state);
        app.engine->update_vehicle_state(&demo_state);

        for (std::size_t output_index = 0; output_index < SV_MAX_OUTPUTS; ++output_index) {
            if (output_set.outputs[output_index].active) {
                glfw_host.make_current(static_cast<int>(output_index));
                break;
            }
        }

        frame_source.get_next_frame(frame_packets[frame_set_index]);

        pthread_mutex_lock(&app.access);
        if (app.engine->pre_process(frame_packets[frame_set_index].frames)) {
            pthread_mutex_unlock(&app.access);
            SPDLOG_ERROR("engine->pre_process() failed");
            return EXIT_FAILURE;
        }
        pthread_mutex_unlock(&app.access);

        for (std::size_t output_index = 0; output_index < SV_MAX_OUTPUTS; ++output_index) {
            if (!output_set.outputs[output_index].active) {
                continue;
            }

            glfw_host.make_current(static_cast<int>(output_index));

            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glViewport(0, 0,
                       output_set.outputs[output_index].config.display_width,
                       output_set.outputs[output_index].config.display_height);
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            pthread_mutex_lock(&app.access);
            if (output_set.outputs[output_index].active) {
                float clear_color[4] = {0.0f, 0.0f, 0.0f, 1.0f};

                if (app.engine->process(frame_packets[frame_set_index].frames,
                                        nullptr,
                                        0,
                                        output_set.outputs[output_index].config.display_width,
                                        output_set.outputs[output_index].config.display_height,
                                        clear_color,
                                        output_set.outputs[output_index],
                                        static_cast<int>(output_index))) {
                    pthread_mutex_unlock(&app.access);
                    SPDLOG_ERROR("engine->process() failed");
                    return EXIT_FAILURE;
                }
            }
            pthread_mutex_unlock(&app.access);

            glfw_host.swap_buffers(output_index);
        }

        frame_source.release_frame(frame_packets[frame_set_index]);
        frame_set_index = 1 - frame_set_index;

        usleep(1000000 / options.fps);
    }

    return 0;
}

} // namespace svapp
