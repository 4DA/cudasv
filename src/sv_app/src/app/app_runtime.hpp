#pragma once

#include <engine/frame_source.hpp>

#include "app/app_context.hpp"
#include "app/cmdline.hpp"
#include "app/glfw_host.hpp"

namespace svapp
{

int run_application_loop(AppContext &app,
                         GLFWHost &glfw_host,
                         const CmdlineOpts &options,
                         engine::OutputSet output_set,
                         videoio::FrameSource &frame_source);

} // namespace svapp
