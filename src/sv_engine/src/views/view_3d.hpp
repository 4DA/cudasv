#ifndef SV_VIEW_3D_HPP
#define SV_VIEW_3D_HPP

#include <memory>
#include <utility>
#include <vector>

#include <engine/engine.hpp>

#include <rf/camera_control/viewpoint_animation.hpp>
#include <rf/renderer/cudarf/cudarf.hpp>
#include <rf/renderer/cudarf/draw_list_renderer.hpp>

#include "views/view_interaction_router.hpp"
#include "views/view_post_process_pipeline.hpp"
#include "views/view_navigation_state.hpp"
#include "views/scene_pass_builder.hpp"
#include "views/surround_view_composer.hpp"
#include "views/viewpoint_registry.hpp"

namespace engine
{

struct VehicleState;
class World;

namespace view
{

class View3D {
public:
    int init(const engine::Config *config,
             cudarf::pipe::Ctx *rasterizationDesc,
             World *world,
             const engine::VehicleState *state,
             unsigned int width,
             unsigned int height);

    void compose(videoio::FrameSet<camera::CAMERAS_TOTAL> frames_set,
                 uchar4 *outputBuffer,
                 cudarf::Framebuffer meshGPUOutput,
                 unsigned int width,
                 unsigned int height,
                 float steering_angle,
                 cudarf::CudaStreams cudaStreams,
                 cudarf::profiling::Events &composeTime,
                 unsigned int frameCounter);

    int handle_event(const engine::Output *output, engine::InputEvent *event);

    int handle_viewpoint(int viewpoint);

    const std::vector<rf::Viewpoint> & get_viewpoints();

    std::vector<rf::PrimitiveComponent *> & get_virtual_controls()
        { return _viewpointControls; }

    int get_current_viewpoint();

    bool is_animation_active(unsigned int &timeleft_ms);

    void set_draw_list_renderer(std::unique_ptr<cudarf::DrawListRenderer> drawListRenderer) {
        _drawListRenderer = std::move(drawListRenderer);
    }

private:
    World                                         *_world;
    const ViewConfig3D                            *_viewConfig;
    const engine::Config                          *_config;

    cudarf::pipe::Ctx                            *_rasterCtx = nullptr;
    std::unique_ptr<cudarf::DrawListRenderer>    _drawListRenderer = nullptr;

    unsigned  int                                 _width;
    unsigned  int                                 _height;

    std::unique_ptr<rf::VirtualCamera>            _virtualCamera;
    std::unique_ptr<ViewpointRegistry>            _viewpointRegistry;
    std::unique_ptr<ViewNavigationState>          _navigationState;
    std::unique_ptr<ViewInteractionRouter>        _interactionRouter;

    std::vector<rf::PrimitiveComponent *>         _viewpointControls;
    std::unique_ptr<rf::ViewpointAnimator>        _viewpointAnimator;

    std::unique_ptr<SurroundViewComposer>         _surroundViewComposer;
    std::unique_ptr<ScenePassBuilder>             _scenePassBuilder;
    std::unique_ptr<ViewPostProcessPipeline>      _postProcessPipeline;
};

} // namespace view

} // namespace engine

#endif
