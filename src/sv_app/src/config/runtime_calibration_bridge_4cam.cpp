#include "config/runtime_calibration_bridge_4cam.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>

#include <spdlog/spdlog.h>

#include "config/canonical_rig.hpp"

namespace
{

camera::CameraID render_slot_for_role(camera::CameraRole role)
{
    switch (role)
    {
    case camera::CameraRole::Front:
        return camera::CAMERA_FRONT;
    case camera::CameraRole::Rear:
        return camera::CAMERA_REAR;
    case camera::CameraRole::Left:
        return camera::CAMERA_LEFT;
    case camera::CameraRole::Right:
        return camera::CAMERA_RIGHT;
    default:
        throw std::runtime_error("role is not renderable through the current 4-camera bridge");
    }
}

camera::CameraLensType lens_model_from_projection(camera::ProjectionModel projection_model)
{
    switch (projection_model)
    {
    case camera::ProjectionModel::Fisheye:
        return camera::CAMERA_LENS_FISHEYE;
    case camera::ProjectionModel::Perspective:
        return camera::CAMERA_LENS_NARROW;
    }

    throw std::runtime_error("unknown projection model");
}

void canonical_pose_to_extrinsics(const camera::CanonicalPose &pose, camera::Extrinsics &extrinsics)
{
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t col = 0; col < 3; ++col) {
            extrinsics.R[row + col * 3] = pose.rotation[row][col];
        }
    }

    extrinsics.T = pose.translation;
}

void fill_runtime_camera_config(const camera::CameraRigCamera &camera_desc,
                                camera::CameraID slot,
                                camera::CameraConfig &camera_cfg)
{
    camera_cfg.camera_id = slot;
    camera_cfg.intrinsics.camera.fill(0.0f);
    camera_cfg.intrinsics.distortion_coeffs.fill(0.0f);
    camera_cfg.intrinsics.lens_shading_coeffs.fill(0.0f);

    camera_cfg.intrinsics.camera[0] = camera_desc.intrinsics.focal_length_px[0];
    camera_cfg.intrinsics.camera[4] = camera_desc.intrinsics.focal_length_px[1];
    camera_cfg.intrinsics.camera[3] = camera_desc.intrinsics.skew;
    camera_cfg.intrinsics.camera[6] = camera_desc.intrinsics.principal_point_px[0];
    camera_cfg.intrinsics.camera[7] = camera_desc.intrinsics.principal_point_px[1];
    camera_cfg.intrinsics.camera[8] = 1.0f;

    const auto coeff_count = std::min<std::size_t>(
        camera_desc.distortion.coefficients.size(),
        camera_cfg.intrinsics.distortion_coeffs.size());

    for (std::size_t i = 0; i < coeff_count; ++i) {
        camera_cfg.intrinsics.distortion_coeffs[i] = camera_desc.distortion.coefficients[i];
    }

    camera_cfg.intrinsics.lens_model = lens_model_from_projection(camera_desc.projection_model);

    canonical_pose_to_extrinsics(camera_desc.pose_vehicle, camera_cfg.extrinsics);
}

} // namespace

int load_camera_rig_into_runtime_calibration(engine::CalibrationConfig *calibration_config,
                                             const camera::CameraRig &rig)
{
    if (!calibration_config) {
        SPDLOG_ERROR("Calibration output pointer is null");
        return -1;
    }

    try {
        if (rig.cameras.size() != camera::CAMERAS_COUNT) {
            throw std::runtime_error("current render bridge requires exactly 4 cameras in the canonical rig");
        }

        std::set<camera::CameraID> seen_slots;

        for (const auto &camera_desc: rig.cameras) {
            camera::CameraID slot = render_slot_for_role(camera_desc.role);
            if (!seen_slots.insert(slot).second) {
                throw std::runtime_error("duplicate canonical camera role mapped to the same render slot");
            }

            fill_runtime_camera_config(camera_desc, slot, calibration_config->camera_cfg[slot]);
        }

        if (seen_slots.size() != camera::CAMERAS_COUNT) {
            throw std::runtime_error("canonical rig must define exactly front, rear, left, and right roles");
        }
    } catch (const std::exception &e) {
        SPDLOG_ERROR("Failed to convert canonical rig [{}] into runtime calibration",
                     rig.rig_name.empty() ? "<unnamed>" : rig.rig_name);
        SPDLOG_ERROR("{}", e.what());
        return -1;
    }

    return 0;
}

int load_camera_rig_into_runtime_calibration(engine::CalibrationConfig *calibration_config,
                                             const std::string &path)
{
    camera::CameraRig rig;
    if (load_canonical_rig(&rig, path)) {
        return -1;
    }

    return load_camera_rig_into_runtime_calibration(calibration_config, rig);
}
