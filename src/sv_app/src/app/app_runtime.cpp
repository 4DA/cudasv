#include "app/app_runtime.hpp"

#include <unistd.h>

#include <GLFW/glfw3.h>
#include <GLES3/gl3.h>

#include <spdlog/spdlog.h>

#include "sources/render_bridge_4cam.hpp"

namespace svapp
{

int run_application_loop(AppContext &app,
                         GLFWHost &glfw_host,
                         const CmdlineOpts &options,
                         engine::OutputSet output_set,
                         videoio::FrameSource &frame_source,
                         VehicleSignalProvider &signal_provider)
{
    videoio::FramePacket source_packets[2];
    videoio::RuntimeFramePacket4Cam runtime_packets[2];
    int frame_set_index = 0;
    vehicle::CANSignals demo_state;
    const videoio::SourceInfo &source_info = frame_source.info();
    RuntimeRenderBridge4CamContext runtime_render_bridge;
    const bool dump_after_frame = !options.dump_frame_path.empty();

    if (!prepare_runtime_render_bridge_4cam_context(source_info, runtime_render_bridge)) {
        SPDLOG_ERROR("Failed to prepare the current 4-camera runtime compatibility bridge");
        return EXIT_FAILURE;
    }

    while (app.running && !glfw_host.should_close_any()) {
        if (!signal_provider.get_next_signals(demo_state)) {
            SPDLOG_ERROR("Failed to fetch next vehicle signal sample");
            return EXIT_FAILURE;
        }
        app.engine->update_vehicle_state(&demo_state);

        for (std::size_t output_index = 0; output_index < SV_MAX_OUTPUTS; ++output_index) {
            if (output_set.outputs[output_index].active) {
                glfw_host.make_current(static_cast<int>(output_index));
                break;
            }
        }

        if (!frame_source.get_next_frame(source_packets[frame_set_index])) {
            SPDLOG_ERROR("Failed to fetch next frame packet from source");
            return EXIT_FAILURE;
        }

        if (!adapt_frame_packet_for_runtime_render_bridge_4cam(source_packets[frame_set_index],
                                                               runtime_packets[frame_set_index],
                                                               runtime_render_bridge)) {
            SPDLOG_ERROR("Failed to adapt source frame packet through the current 4-camera runtime compatibility bridge");
            return EXIT_FAILURE;
        }

        pthread_mutex_lock(&app.access);
        if (app.engine->pre_process(runtime_packets[frame_set_index])) {
            pthread_mutex_unlock(&app.access);
            SPDLOG_ERROR("engine->pre_process() failed");
            return EXIT_FAILURE;
        }
        pthread_mutex_unlock(&app.access);

        bool frame_dumped = false;

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

                if (app.engine->process(runtime_packets[frame_set_index],
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

            if (dump_after_frame) {
                pthread_mutex_lock(&app.access);
                const engine::Error dump_status =
                    app.engine->dump_output_png(options.dump_frame_path,
                                                static_cast<int>(output_index));
                pthread_mutex_unlock(&app.access);

                if (dump_status != engine::OK) {
                    SPDLOG_ERROR("engine->dump_output_png() failed");
                    return EXIT_FAILURE;
                }

                frame_dumped = true;
                break;
            }

            glfw_host.swap_buffers(output_index);
        }

        if (!frame_source.release_frame(source_packets[frame_set_index])) {
            SPDLOG_ERROR("Failed to release frame packet back to source");
            return EXIT_FAILURE;
        }

        if (frame_dumped) {
            return EXIT_SUCCESS;
        }

        frame_set_index = 1 - frame_set_index;

        usleep(1000000 / options.fps);
    }

    return 0;
}

} // namespace svapp
