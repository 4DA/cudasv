#pragma once

#include <rf/renderer/cudarf/draw_list_renderer.hpp>

namespace engine
{

struct Config;
class World;

namespace view
{

class ScenePassBuilder
{
public:
    using OpaqueWork = cudarf::DrawListRenderer::WorkDescription<rf::RENDER_PASS_OPAQUE>;
    using TranslucentWork = cudarf::DrawListRenderer::WorkDescription<rf::RENDER_PASS_TRANSLUCENT>;

    struct WorkSet
    {
        OpaqueWork opaque;
        TranslucentWork translucent;
    };

    WorkSet build(const Config &config,
                  const World &world,
                  rf::VirtualCamera &virtualCamera,
                  cudarf::DrawListRenderer &drawListRenderer,
                  unsigned int width,
                  unsigned int height) const;

    void render(cudarf::DrawListRenderer &drawListRenderer,
                cudarf::pipe::Ctx *rasterCtx,
                rf::Scene &scene,
                rf::VirtualCamera &virtualCamera,
                const WorkSet &work,
                const WorkSet &history,
                cudarf::Framebuffer output,
                unsigned int frameCounter,
                cudaStream_t cudaStream) const;
};

} // namespace view

} // namespace engine
