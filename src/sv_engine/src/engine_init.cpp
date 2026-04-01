#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/printf.h>

#include <engine/camera_config.hpp>
#include <engine/engine.hpp>

#include <rf/renderer/cuda_helpers.hpp>
#include <rf/renderer/gltf_loader.hpp>
#include <rf/renderer/trs_transform.hpp>

#include <virtual_controls/viewpoints/viewpoints.hpp>

#include "engine_internal.hpp"

using namespace engine;

engine::Error Engine::init()
{
    OverlaysConfig *overlays_config = &config.overlays_config;
    VehicleModelConfig *vehicle_model_config = &overlays_config->vehicle_config;

    _impl->world = std::make_unique<World>();
    _impl->frameTimeDB = std::make_shared<cudarf::profiling::Events>("frameTime");

    for (int outputIndex = 0; outputIndex < config.outputs_number; ++outputIndex) {
        CUDA_CHK(cudaStreamCreateWithFlags(&_impl->cudaOutputStreams[outputIndex],
                                           cudaStreamDefault | cudaStreamNonBlocking));

        const int displayWidth = config.outputs[outputIndex].display_width;
        const int displayHeight = config.outputs[outputIndex].display_height;

        _impl->cudaOutput[outputIndex] =
            std::make_unique<cudarf::CudaOutput>(displayWidth, displayHeight);

        SPDLOG_INFO("{}",
                    fmt::sprintf("CUDA Output[%d]/WH: %d / %d",
                                 outputIndex,
                                 displayWidth,
                                 displayHeight));

        _impl->cuda_rasterizers[outputIndex] = std::make_unique<cudarf::pipe::Ctx>();

        cudarf::pipe::init(_impl->cuda_rasterizers[outputIndex].get(),
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
                           0);

        cudarf::create_surface(_impl->mesh_gpu_outputs[outputIndex],
                               displayWidth,
                               displayHeight,
                               _impl->cudaOutputStreams[outputIndex]);

#if defined(DUMP_FRAME_TIMING)
        const std::string timeDbName =
            std::string("Output [") + std::to_string(outputIndex) + std::string("]");
        _impl->outputRenderTimeDB[outputIndex] = _impl->frameTimeDB->add_child(timeDbName);
#endif
    }

    SPDLOG_INFO("{}", fmt::sprintf("Start mesh loading"));

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

    SPDLOG_INFO("{}", fmt::sprintf("Done loading mesh. OK"));

    if (_impl->world->init(_impl->cuda_rasterizers[0].get(),
                           &config,
                           _impl->cudaOutputStreams[0])) {
        SPDLOG_ERROR("{}", fmt::sprintf("world initialization failed"));
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
        SPDLOG_ERROR("{}", fmt::sprintf("3d view initialization failed"));
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
            SPDLOG_ERROR("{}", fmt::sprintf("3d view virtual controls initialization failed"));
            assert(false);
        }
    }

    cudarf::profiling::Events *eventDb = nullptr;

#if defined(DUMP_FRAME_TIMING)
    eventDb = _impl->frameTimeDB.get();
#endif

    _impl->view_3d->set_draw_list_renderer(
        std::make_unique<cudarf::DrawListRenderer>(_impl->world->scene(), eventDb));

    return OK;
}
