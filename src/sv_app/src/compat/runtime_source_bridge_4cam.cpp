#include <spdlog/spdlog.h>

#include "compat/runtime_source_bridge_4cam.hpp"
#include "config/runtime_calibration_bridge_4cam.hpp"
#include "sources/source_validation.hpp"

namespace svapp
{

bool prepare_source_for_runtime_bridge_4cam(engine::CalibrationConfig *calibrationConfig,
                                            const videoio::SourceInfo &sourceInfo,
                                            const camera::CameraRig &rig)
{
    report_source(sourceInfo, rig);

    switch (sourceInfo.kind) {
    case videoio::SourceKind::FileSequence:
        if (!validate_render_bridge_4cam(rig)) {
            SPDLOG_ERROR("Resolved source and rig are not compatible with the current 4-camera runtime bridge");
            return false;
        }

        if (load_camera_rig_into_runtime_calibration(calibrationConfig, rig)) {
            SPDLOG_ERROR("Failed to convert canonical rig into runtime calibration");
            return false;
        }

        return true;
    case videoio::SourceKind::NuScenes:
        SPDLOG_ERROR("NuScenes source loading is not ready for the current 4-camera runtime bridge yet");
        return false;
    case videoio::SourceKind::Unknown:
    default:
        SPDLOG_ERROR("Unsupported source kind in bootstrap");
        return false;
    }
}

} // namespace svapp
