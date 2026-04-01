#pragma once

#include <rf/renderer/cudarf/cudarf.hpp>

namespace rf
{
class VirtualCamera;
}

namespace engine
{

struct Config;
class World;

namespace view
{

class SurroundViewComposer
{
public:
    void compose(const Config &config,
                 const World &world,
                 const rf::VirtualCamera &virtualCamera,
                 unsigned int width,
                 unsigned int height,
                 cudarf::Framebuffer meshGPUOutput,
                 cudarf::CudaStreams cudaStreams) const;
};

} // namespace view

} // namespace engine
