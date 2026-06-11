#include "views/scene_pass_builder.hpp"

#include <limits>
#include <set>

#include <spdlog/spdlog.h>

#include <world.hpp>

#include <engine/engine.hpp>

#include <overlays/underlay/underlay.hpp>

namespace
{

bool equal_margin(float left, float right, float margin)
{
    return std::abs(left - right) <= margin;
}

} // namespace

engine::view::ScenePassBuilder::WorkSet engine::view::ScenePassBuilder::build(
    const Config &config,
    const World &world,
    rf::VirtualCamera &virtualCamera,
    cudarf::DrawListRenderer &drawListRenderer,
    unsigned int width,
    unsigned int height) const
{
    WorkSet work;
    const float aspect = static_cast<float>(width) / height;

    if (!equal_margin(aspect,
                      virtualCamera.m_projection.perspective.aspect,
                      2.0f * std::numeric_limits<float>::epsilon()))
    {
        SPDLOG_ERROR("TAA: projection {} and output {} aspect ratios don't match. Output aspect is used for rendering",
                     virtualCamera.m_projection.perspective.aspect,
                     aspect);

        virtualCamera.m_projection.perspective.aspect = aspect;
        virtualCamera.set_projection(virtualCamera.m_projection);
    }

    drawListRenderer.add_work(work.opaque,
                              world.scene(),
                              vehicleCompoName,
                              virtualCamera,
                              {});

    drawListRenderer.add_work(work.translucent,
                              world.scene(),
                              vehicleCompoName,
                              virtualCamera,
                              {});

    drawListRenderer.add_work(work.translucent,
                              world.scene(),
                              underlayCompoName,
                              virtualCamera,
                              std::set<std::string>(),
                              [](const rf::PrimitiveComponent &) {
                                  return CUDARF_LAYER_LOWEST;
                              });

    if (config.overlays_config.controls_config.enabled) {
        drawListRenderer.add_work(work.opaque,
                                  world.scene(),
                                  controlCompoName,
                                  virtualCamera,
                                  std::set<std::string>());

        drawListRenderer.add_work(work.ui,
                                  world.scene(),
                                  controlCompoName,
                                  virtualCamera,
                                  std::set<std::string>(),
                                  [](const rf::PrimitiveComponent &component) {
                                      if (component.is_front_facing()) {
                                          return CUDARF_LAYER_LOWEST;
                                      }

                                      return CUDARF_LAYER_DEFAULT;
                                  });
    }

    work.translucent.flat.sort(virtualCamera);
    work.translucent.pbr.sort(virtualCamera);
    work.ui.flat.sort(virtualCamera);
    work.ui.pbr.sort(virtualCamera);

    return work;
}

void engine::view::ScenePassBuilder::render(
    cudarf::DrawListRenderer &drawListRenderer,
    cudarf::pipe::Ctx *rasterCtx,
    rf::Scene &scene,
    rf::VirtualCamera &virtualCamera,
    const WorkSet &work,
    const WorkSet &history,
    bool withOpaqueVisibuf,
    cudarf::Framebuffer output,
    cudarf::Framebuffer uiOutput,
    unsigned int frameCounter,
    cudaStream_t cudaStream) const
{
    drawListRenderer.render(rasterCtx,
                            scene,
                            virtualCamera,
                            work.opaque,
#ifdef WITH_TAA
                            history.opaque,
#endif
                            withOpaqueVisibuf,
                            output,
                            frameCounter,
                            cudaStream);

    drawListRenderer.render(rasterCtx,
                            scene,
                            virtualCamera,
                            work.translucent,
#ifdef WITH_TAA
                            history.translucent,
#endif
                            output,
                            cudarf::SHADER_TYPE_PBR,
                            frameCounter,
                            cudaStream);

    drawListRenderer.render(rasterCtx,
                            scene,
                            virtualCamera,
                            work.translucent,
#ifdef WITH_TAA
                            history.translucent,
#endif
                            output,
                            cudarf::SHADER_TYPE_UNLIT,
                            frameCounter,
                            cudaStream);

    drawListRenderer.render(rasterCtx,
                            scene,
                            virtualCamera,
                            work.ui,
#ifdef WITH_TAA
                            history.ui,
#endif
                            uiOutput,
                            cudarf::SHADER_TYPE_PBR,
                            frameCounter,
                            cudaStream);

    drawListRenderer.render(rasterCtx,
                            scene,
                            virtualCamera,
                            work.ui,
#ifdef WITH_TAA
                            history.ui,
#endif
                            uiOutput,
                            cudarf::SHADER_TYPE_UNLIT,
                            frameCounter,
                            cudaStream);
}
