#pragma once

#include <rf/renderer/cudarf/cudarf.hpp>

#include "views/scene_pass_builder.hpp"

namespace rf
{
class VirtualCamera;
}

namespace engine
{
namespace view
{

struct ViewConfig3D;

class ViewPostProcessPipeline
{
public:
    void begin_frame(const ScenePassBuilder::WorkSet &work, unsigned int frameCounter);

    const ScenePassBuilder::WorkSet &history() const
    {
        return _history;
    }

    void run(cudarf::pipe::Ctx *rasterCtx,
             rf::VirtualCamera &virtualCamera,
             const ViewConfig3D &viewConfig,
             cudarf::Framebuffer meshGPUOutput,
             uchar4 *outputBuffer,
             cudarf::CudaStreams cudaStreams,
             cudarf::profiling::Events &composeTime,
             unsigned int frameCounter);

    void end_frame(const ScenePassBuilder::WorkSet &work)
    {
        _history = work;
    }

private:
    ScenePassBuilder::WorkSet _history;
    cudarf::CommonUniforms _historyUniforms;
};

} // namespace view

} // namespace engine
