#ifndef SOURCE_VALIDATION_HPP
#define SOURCE_VALIDATION_HPP

#include <engine/camera_rig.hpp>
#include <engine/frame_source.hpp>

namespace svapp
{

void report_source(const videoio::SourceInfo &sourceInfo,
                   const camera::CameraRig &rig);

bool validate_render_bridge_4cam(const camera::CameraRig &rig);

} // namespace svapp

#endif // SOURCE_VALIDATION_HPP
