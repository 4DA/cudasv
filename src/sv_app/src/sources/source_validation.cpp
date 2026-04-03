#include <array>
#include <sstream>
#include <string>

#include <spdlog/spdlog.h>

#include "sources/source_validation.hpp"
#include "sources/render_bridge_4cam.hpp"

namespace
{

static const char *source_kind_to_string(videoio::SourceKind kind)
{
    switch (kind) {
    case videoio::SourceKind::FileSequence:
        return "file_sequence";
    case videoio::SourceKind::NuScenes:
        return "nuscenes";
    case videoio::SourceKind::Unknown:
        break;
    }

    return "unknown";
}

static const char *camera_role_to_string(camera::CameraRole role)
{
    switch (role) {
    case camera::CameraRole::Front:
        return "front";
    case camera::CameraRole::Rear:
        return "rear";
    case camera::CameraRole::Left:
        return "left";
    case camera::CameraRole::Right:
        return "right";
    case camera::CameraRole::FrontLeft:
        return "front_left";
    case camera::CameraRole::FrontRight:
        return "front_right";
    case camera::CameraRole::RearLeft:
        return "rear_left";
    case camera::CameraRole::RearRight:
        return "rear_right";
    case camera::CameraRole::Unknown:
        break;
    }

    return "unknown";
}

static const char *projection_model_to_string(camera::ProjectionModel model)
{
    switch (model) {
    case camera::ProjectionModel::Perspective:
        return "perspective";
    case camera::ProjectionModel::Fisheye:
        return "fisheye";
    }

    return "unknown";
}

static std::string join_render_roles(
    const std::array<camera::CameraRole, camera::CAMERAS_TOTAL> &roles)
{
    std::ostringstream out;

    for (std::size_t index = 0; index < roles.size(); ++index) {
        if (index > 0) {
            out << ", ";
        }
        out << camera_role_to_string(roles[index]);
    }

    return out.str();
}

static std::string format_vec3(const std::array<float, 3> &value)
{
    std::ostringstream out;
    out << "[" << value[0] << ", " << value[1] << ", " << value[2] << "]";
    return out.str();
}

static std::string format_source_contract(const videoio::SourceInfo::Contract &contract)
{
    std::ostringstream out;
    out << "synchronized_samples=" << (contract.synchronized_samples ? "yes" : "no")
        << ", sample_identity=" << (contract.provides_sample_identity ? "yes" : "no")
        << ", source_timestamp=" << (contract.provides_source_timestamp ? "yes" : "no")
        << ", per_camera_timestamps=" << (contract.provides_per_camera_timestamps ? "yes" : "no")
        << ", ego_pose=" << (contract.provides_ego_pose ? "yes" : "no");
    return out.str();
}

static std::array<float, 3> camera_axis_in_vehicle(const camera::CanonicalPose &pose,
                                                   std::size_t axisIndex)
{
    return {
        pose.rotation[0][axisIndex],
        pose.rotation[1][axisIndex],
        pose.rotation[2][axisIndex],
    };
}

} // namespace

namespace svapp
{

void report_source(const videoio::SourceInfo &sourceInfo,
                   const camera::CameraRig &rig)
{
    SPDLOG_INFO("Source validation: kind='{}', name='{}', dataset_root='{}', sequence_id='{}'",
                source_kind_to_string(sourceInfo.kind),
                sourceInfo.source_name,
                sourceInfo.dataset_root,
                sourceInfo.sequence_id);

    if (sourceInfo.has_sequence_frame_count) {
        SPDLOG_INFO("Source validation: sequence_frame_count={}", sourceInfo.sequence_frame_count);
    }

    SPDLOG_INFO("Source validation: source contract [{}]",
                format_source_contract(sourceInfo.contract));
    SPDLOG_INFO("Source validation: source roles [{}]", join_render_roles(sourceInfo.render_roles));

    for (const auto &cameraDesc : rig.cameras) {
        SPDLOG_INFO("Source validation: rig camera id='{}', role='{}', model='{}', image_size={}x{}",
                    cameraDesc.id,
                    camera_role_to_string(cameraDesc.role),
                    projection_model_to_string(cameraDesc.projection_model),
                    cameraDesc.image_size[0],
                    cameraDesc.image_size[1]);
        SPDLOG_INFO("Source validation:   pose translation={}, right_axis={}, up_axis={}, forward_axis={}",
                    format_vec3(cameraDesc.pose_vehicle.translation),
                    format_vec3(camera_axis_in_vehicle(cameraDesc.pose_vehicle, 0)),
                    format_vec3(camera_axis_in_vehicle(cameraDesc.pose_vehicle, 1)),
                    format_vec3(camera_axis_in_vehicle(cameraDesc.pose_vehicle, 2)));
    }
}

bool validate_render_bridge_4cam(const camera::CameraRig &rig)
{
    std::array<camera::CameraRole, camera::CAMERAS_TOTAL> bridgeRoles;
    std::array<uint32_t, 2> uniformImageSize = {0, 0};
    const bool compatible = resolve_render_bridge_4cam(rig, bridgeRoles, &uniformImageSize);

    SPDLOG_INFO("Source validation: current 4-camera bridge compatibility={}",
                compatible ? "yes" : "no");

    if (compatible) {
        SPDLOG_INFO("Source validation: render bridge roles [{}], uniform_image_size={}x{}",
                    join_render_roles(bridgeRoles),
                    uniformImageSize[0],
                    uniformImageSize[1]);
    }

    return compatible;
}

} // namespace svapp
