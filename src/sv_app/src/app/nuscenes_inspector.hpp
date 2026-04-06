#pragma once

#include <engine/frame_source.hpp>

#include "app/app_context.hpp"
#include "app/cmdline.hpp"
#include "app/glfw_host.hpp"

namespace engine
{
struct OutputSet;
}

namespace svapp
{

int run_nuscenes_inspector_loop(AppContext &app,
                                GLFWHost &glfwHost,
                                const CmdlineOpts &options,
                                const engine::OutputSet &outputSet,
                                videoio::FrameSource &frameSource);

} // namespace svapp
