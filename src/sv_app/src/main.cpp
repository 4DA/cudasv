#include <memory>
#include <pthread.h>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <engine/engine.hpp>

#include "config/config.hpp"
#include "config/canonical_rig.hpp"
#include "config/runtime_calibration_bridge_4cam.hpp"
#include "app/app_context.hpp"
#include "app/app_runtime.hpp"
#include "app/cmdline.hpp"
#include "app/glfw_host.hpp"
#include "sources/source_factory.hpp"

int main(int argc, char* argv[])
{
    svapp::CmdlineOpts cmdline;
    svapp::AppContext app;
    pthread_mutex_init(&app.access, nullptr);

    // init spdlog
    // --
    using spdlog::level::level_enum;

    auto sink = std::make_shared<spdlog::sinks::ansicolor_stderr_sink_mt>();
    // customize colors per level
    sink->set_color(level_enum::trace,    sink->white);
    sink->set_color(level_enum::debug,    sink->cyan);
    sink->set_color(level_enum::info,     sink->green);
    sink->set_color(level_enum::warn,     sink->yellow);
    sink->set_color(level_enum::err,      sink->red);
    sink->set_color(level_enum::critical, sink->red);

    // build default logger with this sink
    auto logger = std::make_shared<spdlog::logger>("default", sink);
    logger->set_level(spdlog::level::info);

    // (A) color the WHOLE line by level:
    // logger->set_pattern("%^[%H:%M:%S.%e] [tid %t] [%s:%#] %v%$");

    // (B) color ONLY the level tag; rest stays normal:
    logger->set_pattern("[%M:%S.%e] [tid %t] [%^%l%$] [%s:%#] %v");

    spdlog::set_default_logger(logger);
    // --

    if (argc < 2) {
        SPDLOG_ERROR("Incorrect command line");
        return -1;
    }

    app.engine = std::make_unique<engine::Engine>();

    if (svapp::parse_cmdline(argc, argv, cmdline)) {
        return -1;
    }

    if (!svapp::has_frame_inputs(cmdline)) {
        SPDLOG_ERROR("--frames is mandatory and requires exactly {} PNG files",
                     static_cast<int>(camera::CAMERAS_TOTAL));
        return -1;
    }

    load_config(&app.engine->config, "configs.json");

    if (cmdline.width <= 0 || cmdline.height <= 0) {
        SPDLOG_ERROR("--width and --height are mandatory and must be positive");
        return -1;
    }

    engine::OutputSet output_set = svapp::make_single_output_set(cmdline.width, cmdline.height);

    int j = 0;

    for (size_t i = 0; i < SV_MAX_OUTPUTS; ++i) {
        if (output_set.outputs[i].active > 0) {
            app.engine->config.outputs[j++] = output_set.outputs[i].config;
        }
    }

    app.engine->config.outputs_number = j;

    svapp::GLFWHost glfw_host(&app, &output_set, app.engine->config.outputs_number);

    svapp::SourceFactoryConfig source_config;
    source_config.frame_paths = cmdline.files;
    source_config.rig_path = cmdline.rig_file;

    auto frame_source = svapp::create_source(source_config);
    if (!frame_source || !frame_source->open()) {
        SPDLOG_ERROR("Failed to create frame source");
        return -1;
    }

    if (load_camera_rig_into_runtime_calibration(&app.engine->config.calibration_config,
                                                 frame_source->rig())) {
        SPDLOG_ERROR("Failed to convert canonical rig into runtime calibration");
        return -1;
    }

    for (std::size_t i = 0; i < SV_MAX_OUTPUTS; ++i) {
        if (output_set.outputs[i].active) {
            glfw_host.make_current(static_cast<int>(i));
            break;
        }
    }

    app.engine->init();

    const int rc = svapp::run_application_loop(app,
                                               glfw_host,
                                               cmdline,
                                               output_set,
                                               *frame_source);

    pthread_mutex_destroy(&app.access);
    return rc;
}
