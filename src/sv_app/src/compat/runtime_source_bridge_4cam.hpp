#ifndef RUNTIME_SOURCE_BRIDGE_4CAM_HPP
#define RUNTIME_SOURCE_BRIDGE_4CAM_HPP

#include <engine/camera_rig.hpp>
#include <engine/engine.hpp>
#include <engine/frame_source.hpp>

namespace svapp
{

bool prepare_source_for_runtime_bridge_4cam(engine::CalibrationConfig *calibrationConfig,
                                            const videoio::SourceInfo &sourceInfo,
                                            const camera::CameraRig &rig);

} // namespace svapp

#endif // RUNTIME_SOURCE_BRIDGE_4CAM_HPP
