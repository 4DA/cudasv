#ifndef CUDARF_CAMERA_HPP
#define CUDARF_CAMERA_HPP

// cudarf_camera.hpp — bridge between cudarf and rf::VirtualCamera.
//
// Kept separate from cudarf.hpp so that the core rasterizer interface does not
// pull in virtual_camera.hpp. Include this header (instead of cudarf.hpp) in
// files that work with rf::VirtualCamera.

#include <rf/renderer/cudarf/cudarf.hpp>
#include <rf/renderer/virtual_camera.hpp>

namespace cudarf
{

cudarf::CommonUniforms make_common(const rf::VirtualCamera *camera);

} // namespace cudarf

#endif
