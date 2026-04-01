#include "views/view_post_process_pipeline.hpp"

#include <overlays/2d/sky_fade.hpp>

#include <engine/views_config.hpp>

#include <rf/renderer/cudarf/cudarf_camera.hpp>
#include <rf/renderer/virtual_camera.hpp>

void engine::view::ViewPostProcessPipeline::begin_frame(
    const ScenePassBuilder::WorkSet &work,
    unsigned int frameCounter)
{
    if (frameCounter == 0) {
        _history = work;
    }
}

void engine::view::ViewPostProcessPipeline::run(
    cudarf::pipe::Ctx *rasterCtx,
    rf::VirtualCamera &virtualCamera,
    const engine::view::ViewConfig3D &viewConfig,
    cudarf::Framebuffer meshGPUOutput,
    uchar4 *outputBuffer,
    cudarf::CudaStreams cudaStreams,
    cudarf::profiling::Events &composeTime,
    unsigned int frameCounter)
{
    {
        auto projection = cudarf::ProjectionParams {
            virtualCamera.get_projection().perspective.near,
            virtualCamera.get_projection().perspective.far,
            virtualCamera.get_projection().perspective.fov_y,
        };

#ifdef WITH_TAA
        auto frameUniforms = cudarf::make_common(&virtualCamera);
        cudarf::pipe::TAA(rasterCtx,
                          frameUniforms,
                          _historyUniforms,
                          projection,
                          frameCounter,
                          cudaStreams.rendering);
        _historyUniforms = frameUniforms;
#endif
    }

    float skyFadeStartV = -1.0f;
    float skyFadeMaxV = -1.0f;

    if (viewConfig.sky_fade.enabled) {
        compute_sky_fade(&viewConfig.sky_fade,
                         virtualCamera,
                         skyFadeStartV,
                         skyFadeMaxV);
    }

#ifdef DUMP_FRAME_TIMING
    const auto composeInterval = composeTime.startInterval("compose", cudaStreams.rendering);
#endif

    cudarf::compose(meshGPUOutput,
                    cudarf::pipe::get_output_fb(rasterCtx, frameCounter),
                    virtualCamera.exposure,
                    rasterCtx->width,
                    rasterCtx->height,
                    skyFadeStartV,
                    skyFadeMaxV,
                    outputBuffer,
                    cudaStreams.rendering);

#ifdef DUMP_FRAME_TIMING
    composeTime.stopInterval(composeInterval);
#endif
}
