#include <memory>
#include <pthread.h>
#include <array>
#include <utility>

#include <spdlog/spdlog.h>
#include <spdlog/common.h>
#include <spdlog/details/null_mutex.h>
#include <spdlog/sinks/base_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <engine/engine.hpp>

#include "config/config.hpp"
#include "app/app_context.hpp"
#include "app/app_runtime.hpp"
#include "app/cmdline.hpp"
#include "app/demo_vehicle_signal_provider.hpp"
#include "app/glfw_host.hpp"
#include "app/nuscenes_inspector.hpp"
#include "app/test_scenario.hpp"
#include "compat/runtime_source_bridge_4cam.hpp"
#include "sources/source_factory.hpp"
#include "sources/source_validation.hpp"

namespace
{

template <typename Mutex>
class level_range_sink final : public spdlog::sinks::base_sink<Mutex>
{
public:
    level_range_sink(spdlog::sink_ptr inner,
                     spdlog::level::level_enum minLevel,
                     spdlog::level::level_enum maxLevel)
        : _inner(std::move(inner)),
          _minLevel(minLevel),
          _maxLevel(maxLevel)
    {
    }

protected:
    void sink_it_(const spdlog::details::log_msg &msg) override
    {
        if (msg.level < _minLevel || msg.level > _maxLevel) {
            return;
        }

        _inner->log(msg);
    }

    void flush_() override
    {
        _inner->flush();
    }

    void set_pattern_(const std::string &pattern) override
    {
        _inner->set_pattern(pattern);
    }

    void set_formatter_(std::unique_ptr<spdlog::formatter> sinkFormatter) override
    {
        _inner->set_formatter(std::move(sinkFormatter));
    }

private:
    spdlog::sink_ptr _inner;
    spdlog::level::level_enum _minLevel;
    spdlog::level::level_enum _maxLevel;
};

using level_range_sink_mt = level_range_sink<std::mutex>;

} // namespace

int main(int argc, char* argv[])
{
    svapp::CmdlineOpts cmdline;
    svapp::AppContext app;
    pthread_mutex_init(&app.access, nullptr);

    // init spdlog
    // --
    using spdlog::level::level_enum;

    auto stdout_sink = std::make_shared<spdlog::sinks::ansicolor_stdout_sink_mt>();
    auto stderr_sink = std::make_shared<spdlog::sinks::ansicolor_stderr_sink_mt>();

    stdout_sink->set_color(level_enum::trace,    stdout_sink->white);
    stdout_sink->set_color(level_enum::debug,    stdout_sink->cyan);
    stdout_sink->set_color(level_enum::info,     stdout_sink->green);
    stdout_sink->set_color(level_enum::warn,     stdout_sink->yellow);
    stdout_sink->set_color(level_enum::err,      stdout_sink->red);
    stdout_sink->set_color(level_enum::critical, stdout_sink->red);

    stderr_sink->set_color(level_enum::trace,    stderr_sink->white);
    stderr_sink->set_color(level_enum::debug,    stderr_sink->cyan);
    stderr_sink->set_color(level_enum::info,     stderr_sink->green);
    stderr_sink->set_color(level_enum::warn,     stderr_sink->yellow);
    stderr_sink->set_color(level_enum::err,      stderr_sink->red);
    stderr_sink->set_color(level_enum::critical, stderr_sink->red);

    auto stdout_range_sink = std::make_shared<level_range_sink_mt>(
        stdout_sink,
        level_enum::trace,
        level_enum::info);

    auto stderr_range_sink = std::make_shared<level_range_sink_mt>(
        stderr_sink,
        level_enum::warn,
        level_enum::critical);

    // build default logger with this sink
    auto logger = std::make_shared<spdlog::logger>(
        "default",
        spdlog::sinks_init_list{stdout_range_sink, stderr_range_sink});
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

    if (!svapp::validate_source_inputs(cmdline)) {
        return -1;
    }

    if (cmdline.source_kind != videoio::SourceKind::NuScenes) {
        load_config(&app.engine->config, "configs.json");
    }

    if (!cmdline.test_scenario_file.empty()) {
        if (!svapp::load_test_scenario_config(
                cmdline.test_scenario_file,
                app.engine->config.overlays_config.test_scenario_config))
        {
            return -1;
        }
    }

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
    source_config.source_kind = cmdline.source_kind;
    source_config.dataset_root = cmdline.dataset_root;
    source_config.sequence_id = cmdline.sequence_id;
    source_config.frame_paths = cmdline.files;
    source_config.rig_path = cmdline.rig_file;

    auto frame_source = svapp::create_source(source_config);
    if (!frame_source || !frame_source->open()) {
        SPDLOG_ERROR("Failed to create frame source");
        return -1;
    }

    const videoio::SourceInfo &source_info = frame_source->info();
    SPDLOG_INFO("Opened source '{}' [kind={}, dataset_root='{}', sequence_id='{}']",
                source_info.source_name,
                static_cast<int>(source_info.kind),
                source_info.dataset_root,
                source_info.sequence_id);

    if (source_info.kind == videoio::SourceKind::NuScenes) {
        svapp::report_source(source_info, frame_source->rig());
        return svapp::run_nuscenes_inspector_loop(app,
                                                  glfw_host,
                                                  cmdline,
                                                  output_set,
                                                  *frame_source);
    }

    if (!svapp::prepare_source_for_runtime_bridge_4cam(&app.engine->config.calibration_config,
                                                       source_info,
                                                       frame_source->rig())) {
        return -1;
    }

    for (std::size_t i = 0; i < SV_MAX_OUTPUTS; ++i) {
        if (output_set.outputs[i].active) {
            glfw_host.make_current(static_cast<int>(i));
            break;
        }
    }

    app.engine->init();

    svapp::DemoVehicleSignalProvider signal_provider;

    const int rc = svapp::run_application_loop(app,
                                               glfw_host,
                                               cmdline,
                                               output_set,
                                               *frame_source,
                                               signal_provider);

    pthread_mutex_destroy(&app.access);
    return rc;
}
