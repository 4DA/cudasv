#ifndef SV_VIEWPOINT_CONTROL_HPP
#define SV_VIEWPOINT_CONTROL_HPP

#include <world.hpp>
#include <rf/renderer/cudarf/cudarf.hpp>
#include <rf/camera_control/camera_control.hpp>
#include <rf/camera_control/viewpoint_animation.hpp>

namespace engine
{

int sv_viewpoint_controls_init(
    cudarf::pipe::Ctx &desc,
    std::vector<rf::PrimitiveComponent *> &viewpoint_controls,
    const VirtualControlConfig *config,
    World &world,
    const std::vector<rf::Viewpoint> &viewpoints,
    cudaStream_t cuStream);

}

#endif
