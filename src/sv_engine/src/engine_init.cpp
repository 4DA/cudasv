#include <spdlog/spdlog.h>

#include <engine/camera_config.hpp>
#include <engine/engine.hpp>

#include <rf/renderer/cuda_helpers.hpp>
#include <rf/renderer/gltf_loader.hpp>
#include <rf/renderer/trs_transform.hpp>

#include <virtual_controls/viewpoints/viewpoints.hpp>

#include "engine_internal.hpp"

using namespace engine;

namespace
{

glm::quat make_rotation_quat_xyz_deg(const float rotationDeg[3])
{
    const float x = glm::radians(rotationDeg[0]);
    const float y = glm::radians(rotationDeg[1]);
    const float z = glm::radians(rotationDeg[2]);

    const glm::quat qx = glm::angleAxis(x, glm::vec3(1.0f, 0.0f, 0.0f));
    const glm::quat qy = glm::angleAxis(y, glm::vec3(0.0f, 1.0f, 0.0f));
    const glm::quat qz = glm::angleAxis(z, glm::vec3(0.0f, 0.0f, 1.0f));

    return glm::normalize(qz * qy * qx);
}

rf::TRSTransform make_test_scenario_transform(const TestScenarioConfig &config)
{
    return rf::TRSTransform(glm::vec3(config.position[0], config.position[1], config.position[2]),
                            make_rotation_quat_xyz_deg(config.rotation),
                            glm::vec3(config.scale[0], config.scale[1], config.scale[2]));
}

} // namespace

engine::Error Engine::init()
{
    OverlaysConfig *overlays_config = &config.overlays_config;
    VehicleModelConfig *vehicle_model_config = &overlays_config->vehicle_config;
    TestScenarioConfig *test_scenario_config = &overlays_config->test_scenario_config;

    _impl->world = std::make_unique<World>();
    _impl->frameTimeDB = std::make_shared<cudarf::profiling::Events>("frameTime");

    for (int outputIndex = 0; outputIndex < config.outputs_number; ++outputIndex) {
        CUDA_CHK(cudaStreamCreateWithFlags(&_impl->cudaOutputStreams[outputIndex],
                                           cudaStreamDefault | cudaStreamNonBlocking));

        const int displayWidth = config.outputs[outputIndex].display_width;
        const int displayHeight = config.outputs[outputIndex].display_height;

        _impl->cudaOutput[outputIndex] =
            std::make_unique<cudarf::CudaOutput>(displayWidth, displayHeight);

        SPDLOG_INFO("CUDA Output[{}]/WH: {} / {}", outputIndex, displayWidth, displayHeight);

        _impl->cuda_rasterizers[outputIndex] = std::make_unique<cudarf::pipe::Ctx>(
                           displayWidth,
                           displayHeight,
                           DefaultTileQueueLimit,
                           _impl->cudaOutput[outputIndex]->SMPCount,
#ifdef WITH_TAA
                           true,
#else
                           false,
#endif
                           -1,
                           _impl->cudaOutputStreams[outputIndex]);

        cudarf::create_surface(_impl->mesh_gpu_outputs[outputIndex],
                               displayWidth,
                               displayHeight,
                               _impl->cudaOutputStreams[outputIndex]);

        if constexpr (CUDARF_ENABLE_CUDA_PROFILING) {
            const std::string timeDbName =
                std::string("Output [") + std::to_string(outputIndex) + std::string("]");
            _impl->outputRenderTimeDB[outputIndex] = _impl->frameTimeDB->add_child(timeDbName);
        }
    }

    SPDLOG_INFO("Start mesh loading");

    rf::TRSTransform vehicleTransform;
    vehicleTransform.scale = glm::vec3(VehicleScaleFactor, VehicleScaleFactor, VehicleScaleFactor);

    _impl->world->vehicle_component() =
        _impl->world->scene().add_scene_component(vehicleCompoName,
                                                  vehicleTransform,
                                                  _impl->world->scene().get_root());

    loader::load_gltf_model(_impl->cuda_rasterizers[0].get(),
                            vehicle_model_config->model_path,
                            _impl->world->scene(),
                            _impl->world->vehicle_component(),
                            "vehicle::",
                            _impl->world->animations(),
                            _impl->world->vehicle_model(),
                            nullptr,
                            _impl->cudaOutputStreams[0]);

    assert(_impl->world->scene().get_materials_count() > 0);

    SPDLOG_INFO("Done loading mesh. OK");

    if (test_scenario_config->enabled) {
        std::string scenarioName = test_scenario_config->name.empty()
                                       ? "test_scenario"
                                       : test_scenario_config->name;
        std::string scenarioPrefix = "test::" + scenarioName + "::";
        std::string scenarioRootName = "TestScenarioRoot::" + scenarioName;

        rf::SceneComponent *scenarioRoot =
            _impl->world->scene().add_scene_component(scenarioRootName,
                                                      make_test_scenario_transform(*test_scenario_config),
                                                      _impl->world->vehicle_component());

        _impl->testScenarioModels.emplace_back();
        _impl->testScenarioAnimations.emplace_back();

        if (!loader::load_gltf_model(_impl->cuda_rasterizers[0].get(),
                                     test_scenario_config->glb_path,
                                     _impl->world->scene(),
                                     scenarioRoot,
                                     scenarioPrefix,
                                     _impl->testScenarioAnimations.back(),
                                     _impl->testScenarioModels.back(),
                                     nullptr,
                                     _impl->cudaOutputStreams[0]))
        {
            SPDLOG_ERROR("Failed to load test scenario '{}'", scenarioName);
            return ERROR;
        }

        SPDLOG_INFO("Loaded test scenario '{}' under VehicleRoot", scenarioName);
    }

    if (_impl->world->init(_impl->cuda_rasterizers[0].get(),
                           &config,
                           _impl->cudaOutputStreams[0])) {
        SPDLOG_ERROR("world initialization failed");
        assert(false);
        return ERROR;
    }

    if (!config.views_config.view_3d.enabled) {
        return OK;
    }

    _impl->view_3d = std::make_unique<view::View3D>();

    const int displayWidth = config.outputs[0].display_width;
    const int displayHeight = config.outputs[0].display_height;

    if (_impl->view_3d->init(&config,
                             _impl->cuda_rasterizers[0].get(),
                             _impl->world.get(),
                             &_impl->vehicleState,
                             displayWidth,
                             displayHeight)) {
        SPDLOG_ERROR("3d view initialization failed");
        assert(false);
        return ERROR;
    }

    if (config.overlays_config.controls_config.enabled) {
        std::vector<rf::PrimitiveComponent *> &viewpointControls =
            _impl->view_3d->get_virtual_controls();

        if (sv_viewpoint_controls_init(*_impl->cuda_rasterizers[0],
                                       viewpointControls,
                                       &config.overlays_config.controls_config,
                                       *_impl->world,
                                       _impl->view_3d->get_viewpoints(),
                                       _impl->cudaOutputStreams[0])) {
            SPDLOG_ERROR("3d view virtual controls initialization failed");
            assert(false);
        }
    }

    cudarf::profiling::Events *eventDb = nullptr;

    if constexpr (CUDARF_ENABLE_CUDA_PROFILING) {
        eventDb = _impl->frameTimeDB.get();
    }

    _impl->view_3d->set_draw_list_renderer(
        std::make_unique<cudarf::DrawListRenderer>(_impl->world->scene(), eventDb));

    return OK;
}
