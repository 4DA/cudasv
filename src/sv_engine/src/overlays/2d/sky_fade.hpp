#ifndef SV_SKY_FADE_HPP
#define SV_SKY_FADE_HPP

#include <chrono>
#include <vector>

#include <engine/views_config.hpp>
#include <rf/renderer/virtual_camera.hpp>

namespace engine
{
namespace view
{

int compute_sky_fade(const SkyFade3D *fade_config,
                     const rf::VirtualCamera &camera,
                     float &fadeStart,
                     float &fadeMax);
}
}

#endif
