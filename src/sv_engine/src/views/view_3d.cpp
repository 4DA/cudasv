#include <spdlog/spdlog.h>

#include <world.hpp>

#include <rf/camera_control/viewpoint_animation.hpp>
#include <rf/renderer/trs_transform.hpp>
#include <rf/renderer/virtual_camera.hpp>

#include <engine/engine.hpp>

#include "views/view_interaction_router.hpp"
#include "views/view_post_process_pipeline.hpp"
#include "views/view_navigation_state.hpp"
#include "views/scene_pass_builder.hpp"
#include "views/surround_view_composer.hpp"
#include "views/viewpoint_registry.hpp"
#include "view_3d.hpp"

using namespace engine;
using namespace view;

const std::vector<rf::Viewpoint> &View3D::get_viewpoints()
{
    return _viewpointRegistry->viewpoints();
}

int View3D::get_current_viewpoint()
{
    return _navigationState->current_viewpoint();
}

int View3D::handle_viewpoint(int viewpoint)
{
    const auto &viewpoints = _viewpointRegistry->viewpoints();
    if (viewpoint < 0 || viewpoint >= static_cast<int>(viewpoints.size())) {
        SPDLOG_DEBUG("ignoring non-selectable viewpoint id {}", viewpoint);
        return 0;
    }

    if (_viewpointAnimator) {
        unsigned int remainingTimeMs;

        if (_viewpointAnimator->is_finished(remainingTimeMs)) {
            SPDLOG_ERROR("viewpoint animation is active");
            return -1;
        }

        _navigationState->set_current_viewpoint(viewpoint);

        // disable camera controller during animation phase
        _navigationState->clear_active_controller();

        _viewpointAnimator->set_viewpoint(
            *_virtualCamera,
            _navigationState->camera_spherical_angle(),
            _navigationState->rotator(),
            viewpoints.at(viewpoint))
            .then([viewpoint, this]() {
                NavigationMode navigation_mode =
                    this->_viewConfig->viewpoint_presets[viewpoint].navigationMode;
                _navigationState->set_active_controller(navigation_mode);
            });

        if (_viewpointControls.size()) {
            if (viewpoint < static_cast<int>(_viewpointControls.size())) {
                _navigationState->set_current_viewpoint_component(_viewpointControls[viewpoint]);
            }
            else {
                _navigationState->set_current_viewpoint_component(nullptr);
            }
        }

        return 0;
    }

    return -1;
}

int View3D::handle_event(const Output *output, InputEvent *event)
{
    return _interactionRouter->handle_event(
        output,
        event,
        _viewpointControls,
        [this](int viewpoint) {
            return handle_viewpoint(viewpoint);
        });
}


bool View3D::is_animation_active(unsigned int &remainingTimeMs)
{
    if (_viewpointAnimator) {
        if (_viewpointAnimator->is_finished(remainingTimeMs)) {
            return true;
        }
    } else {
        remainingTimeMs = 0;
    }

    return false;
}

int View3D::init(const Config *config,
                 cudarf::pipe::Ctx *_rasterCtx,
                 World *world,
                 const engine::VehicleState *state,
                 unsigned int width,
                 unsigned int height)
{
    (void)state;

    this->_world = world;
    this->_config = config;
    this->_viewConfig = &config->views_config.view_3d;
    this->_rasterCtx = _rasterCtx;

    this->_width = width;
    this->_height = height;

    assert(_width > 0);
    assert(_height > 0);

    rf::TRSTransform transform;
    if (_viewConfig->viewpoint.poseMode == POSE_MODE_LOOK_AT) {
        transform = rf::get_look_at_trs(
            glm::vec3(_viewConfig->viewpoint.camera.position[0],
                      _viewConfig->viewpoint.camera.position[1],
                      _viewConfig->viewpoint.camera.position[2]),
            glm::vec3(_viewConfig->viewpoint.camera.look_at[0],
                      _viewConfig->viewpoint.camera.look_at[1],
                      _viewConfig->viewpoint.camera.look_at[2]),
            glm::vec3(_viewConfig->viewpoint.camera.up[0],
                      _viewConfig->viewpoint.camera.up[1],
                      _viewConfig->viewpoint.camera.up[2]));
    }

    rf::Projection initial_projection(
        glm::radians(_viewConfig->viewpoint.camera.vfov),
        static_cast<float>(_width) / _height,
        _viewConfig->viewpoint.camera.z_near,
        _viewConfig->viewpoint.camera.z_far);

    _virtualCamera = std::make_unique<rf::VirtualCamera>(
        initial_projection,
        transform,
        _viewConfig->exposure);

    _navigationState = std::make_unique<ViewNavigationState>();
    if (_navigationState->init(*_viewConfig)) {
        return -1;
    }

    SPDLOG_INFO("using perspective projection: vfov = {}, aspect = {}\n",
                _viewConfig->viewpoint.camera.vfov,
                initial_projection.perspective.aspect);

    assert(initial_projection.perspective.aspect > 0.0f);

    if (_viewConfig->viewpoint.poseMode == POSE_MODE_ORBITAL) {
        _virtualCamera->transform =
            _navigationState->rotator().get_orientation(_navigationState->camera_spherical_angle());
    }

    _viewpointRegistry = std::make_unique<ViewpointRegistry>();
    if (_viewpointRegistry->init(*_viewConfig, _width, _height)) {
        return -1;
    }

    if (_viewConfig->viewpoints_count > 0) {
        _viewpointAnimator.reset(
            new rf::ViewpointAnimator(_viewConfig->duration_ms));
    }

    _interactionRouter = std::make_unique<ViewInteractionRouter>();
    _interactionRouter->init(_world,
                             &config->overlays_config.controls_config,
                             _virtualCamera.get(),
                             _navigationState.get());

    _surroundViewComposer = std::make_unique<SurroundViewComposer>();
    _scenePassBuilder = std::make_unique<ScenePassBuilder>();
    _postProcessPipeline = std::make_unique<ViewPostProcessPipeline>();

    return 0;
}

void View3D::update_camera()
{
    assert(_virtualCamera);
    assert(_navigationState);
    assert(_viewConfig);

    if (_viewpointAnimator) {
        _viewpointAnimator->update(*_virtualCamera,
                                   _navigationState->camera_spherical_angle(),
                                   _navigationState->rotator());
    }

    auto *active_controller = _navigationState->active_controller();
    if (active_controller) {
        if (active_controller->update_needed()) {
            active_controller->update(*_virtualCamera,
                                      _navigationState->camera_spherical_angle(),
                                      _navigationState->rotator());

            SPDLOG_DEBUG("new camera position: phi = {}, theta = {}",
                         _navigationState->camera_spherical_angle().polar,
                         _navigationState->camera_spherical_angle().azimuthal);
        }
    }

    _virtualCamera->exposure = _viewConfig->exposure;
}

void View3D::compose(videoio::FrameSet<camera::CAMERAS_TOTAL> frames_set,
                     uchar4 *outputBuffer,
                     cudarf::Framebuffer meshGPUOutput,
                     unsigned int width,
                     unsigned int height,
                     float steering_angle,
                     cudarf::CudaStreams cudaStreams,
                     cudarf::profiling::Events &composeTime,
                     unsigned int frameCounter)
{
    (void)frames_set;
    assert(_width == width);
    assert(_height == height);
    assert(_rasterCtx);
    assert(_drawListRenderer);
    assert(_surroundViewComposer);
    assert(_scenePassBuilder);
    assert(_postProcessPipeline);
    (void)steering_angle;

    int viewTotalInterval = -1;
    if constexpr (CUDARF_ENABLE_CUDA_PROFILING) {
        viewTotalInterval = composeTime.start_interval("view_3d_total", cudaStreams.rendering);
    }

    int surroundProjectionInterval = -1;
    if constexpr (CUDARF_ENABLE_CUDA_PROFILING) {
        surroundProjectionInterval =
            composeTime.start_interval("surround_view_projection", cudaStreams.rendering);
    }

    _surroundViewComposer->compose(*_config,
                                   *_world,
                                   *_virtualCamera,
                                   width,
                                   height,
                                   meshGPUOutput,
                                   cudaStreams);

    if constexpr (CUDARF_ENABLE_CUDA_PROFILING) {
        composeTime.stop_interval(surroundProjectionInterval);
    }

    auto sceneWork = _scenePassBuilder->build(*_config,
                                              *_world,
                                              *_virtualCamera,
                                              *_drawListRenderer,
                                              width,
                                              height);

    _postProcessPipeline->begin_frame(sceneWork, frameCounter);

    cudarf::Framebuffer internalFB = cudarf::pipe::get_internal_fb(_rasterCtx, frameCounter);
    cudarf::Framebuffer uiFB = cudarf::pipe::get_ui_fb(_rasterCtx);

    int sceneRenderInterval = -1;
    if constexpr (CUDARF_ENABLE_CUDA_PROFILING) {
        sceneRenderInterval = composeTime.start_interval("scene_render", cudaStreams.rendering);
    }

    _scenePassBuilder->render(*_drawListRenderer,
                              _rasterCtx,
                              _world->scene(),
                              *_virtualCamera,
                              sceneWork,
                              _postProcessPipeline->history(),
                              _config->overlays_config.renderer_config.use_visibuf != 0,
                              internalFB,
                              uiFB,
                              frameCounter,
                              cudaStreams.rendering);

    if constexpr (CUDARF_ENABLE_CUDA_PROFILING) {
        composeTime.stop_interval(sceneRenderInterval);
    }

    _postProcessPipeline->run(_rasterCtx,
                              *_virtualCamera,
                              *_viewConfig,
                              meshGPUOutput,
                              uiFB,
                              outputBuffer,
                              cudaStreams,
                              composeTime,
                              frameCounter);

    if constexpr (CUDARF_ENABLE_CUDA_PROFILING) {
        composeTime.stop_interval(viewTotalInterval);
    }

    _postProcessPipeline->end_frame(sceneWork);
}
