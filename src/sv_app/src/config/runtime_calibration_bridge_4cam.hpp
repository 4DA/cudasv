#ifndef RUNTIME_CALIBRATION_BRIDGE_4CAM_HPP
#define RUNTIME_CALIBRATION_BRIDGE_4CAM_HPP

#include <string>

#include <engine/camera_rig.hpp>
#include <engine/engine.hpp>

// Current rendering bridge:
// - runtime rendering still expects exactly 4 cameras
// - roles must map to the current fixed 4-camera render slots

int load_camera_rig_into_runtime_calibration(engine::CalibrationConfig *calibration_config,
                                             const camera::CameraRig &rig);

int load_camera_rig_into_runtime_calibration(engine::CalibrationConfig *calibration_config,
                                             const std::string &path);

#endif // RUNTIME_CALIBRATION_BRIDGE_4CAM_HPP
