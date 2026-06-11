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
        _historyUniforms = {};
    }

}

void engine::view::ViewPostProcessPipeline::run(
    cudarf::pipe::Ctx *rasterCtx,
    rf::VirtualCamera &virtualCamera,
    const engine::view::ViewConfig3D &viewConfig,
    cudarf::Framebuffer meshGPUOutput,
    cudarf::Framebuffer uiOutput,
    uchar4 *outputBuffer,
    cudarf::CudaStreams cudaStreams,
    cudarf::profiling::Events &composeTime,
    unsigned int frameCounter)
{
    cudarf::Framebuffer sceneOutput = cudarf::pipe::get_output_fb(rasterCtx, frameCounter);

    {
        auto projection = cudarf::ProjectionParams {
            virtualCamera.get_projection().perspective.near,
            virtualCamera.get_projection().perspective.far,
            virtualCamera.get_projection().perspective.fov_y,
        };

#ifdef WITH_TAA
        auto frameUniforms = cudarf::make_common(&virtualCamera);
        if (rasterCtx->TAAEnabled) {
            if (frameCounter == 0) {
                _historyUniforms = frameUniforms;
                sceneOutput = cudarf::pipe::get_output_fb(rasterCtx, frameCounter);
                cudarf::pipe::copy_framebuffer(
                    rasterCtx,
                    cudarf::pipe::get_internal_fb(rasterCtx, frameCounter),
                    sceneOutput,
                    cudaStreams.rendering);
            } else {
                cudarf::pipe::TAA(rasterCtx,
                                  frameUniforms,
                                  _historyUniforms,
                                  projection,
                                  frameCounter,
                                  cudaStreams.rendering);
            }
        } else {
            sceneOutput = cudarf::pipe::get_internal_fb(rasterCtx, frameCounter);
        }
        _historyUniforms = frameUniforms;
#else
        (void)projection;
        sceneOutput = cudarf::pipe::get_internal_fb(rasterCtx, frameCounter);
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

    int composeInterval = -1;
    if constexpr (CUDARF_ENABLE_CUDA_PROFILING) {
        composeInterval = composeTime.start_interval("compose", cudaStreams.rendering);
    }

    cudarf::compose(meshGPUOutput,
                    sceneOutput,
                    uiOutput,
                    virtualCamera.exposure,
                    rasterCtx->width,
                    rasterCtx->height,
                    skyFadeStartV,
                    skyFadeMaxV,
                    outputBuffer,
                    cudaStreams.rendering);

    if constexpr (CUDARF_ENABLE_CUDA_PROFILING) {
        composeTime.stop_interval(composeInterval);
    }
}
