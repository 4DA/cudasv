#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "sources/nuscenes_source.hpp"

namespace
{

using json = nlohmann::json;
namespace fs = std::filesystem;

struct ResolvedNuScenesTarget
{
    std::string sampleToken;
    std::string sceneName;
    uint64_t sceneSampleCount = 0;
    bool hasSceneSampleCount = false;
};

struct ResolvedCameraSample
{
    camera::CameraRole role = camera::CameraRole::Unknown;
    std::string channelName;
    std::string relativePath;
    std::string calibratedSensorToken;
    uint32_t width = 0;
    uint32_t height = 0;
};

static const char *camera_role_to_string(camera::CameraRole role)
{
    switch (role) {
    case camera::CameraRole::Front:
        return "front";
    case camera::CameraRole::Rear:
        return "rear";
    case camera::CameraRole::FrontLeft:
        return "front_left";
    case camera::CameraRole::FrontRight:
        return "front_right";
    case camera::CameraRole::RearLeft:
        return "rear_left";
    case camera::CameraRole::RearRight:
        return "rear_right";
    case camera::CameraRole::Left:
    case camera::CameraRole::Right:
    case camera::CameraRole::Unknown:
        break;
    }

    return "unknown";
}

static bool has_required_table_files(const fs::path &root)
{
    static const std::array<const char *, 6> requiredFiles = {
        "scene.json",
        "sample.json",
        "sample_data.json",
        "sensor.json",
        "calibrated_sensor.json",
        "ego_pose.json",
    };

    for (const char *fileName : requiredFiles) {
        if (!fs::is_regular_file(root / fileName)) {
            return false;
        }
    }

    return true;
}

static fs::path resolve_nuscenes_version_root(const fs::path &datasetRoot)
{
    if (!fs::exists(datasetRoot)) {
        throw std::runtime_error("dataset root does not exist");
    }
    if (!fs::is_directory(datasetRoot)) {
        throw std::runtime_error("dataset root is not a directory");
    }

    if (has_required_table_files(datasetRoot)) {
        return datasetRoot;
    }

    std::vector<fs::path> candidates;
    for (const auto &entry : fs::directory_iterator(datasetRoot)) {
        if (!entry.is_directory()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (name.rfind("v1.0-", 0) != 0) {
            continue;
        }
        if (has_required_table_files(entry.path())) {
            candidates.push_back(entry.path());
        }
    }

    if (candidates.empty()) {
        throw std::runtime_error(
            "dataset root does not contain a nuScenes version directory with scene/sample metadata");
    }
    if (candidates.size() == 1) {
        return candidates.front();
    }

    for (const fs::path &candidate : candidates) {
        const std::string name = candidate.filename().string();
        if (name == "v1.0-mini") {
            return candidate;
        }
    }
    for (const fs::path &candidate : candidates) {
        const std::string name = candidate.filename().string();
        if (name == "v1.0-trainval") {
            return candidate;
        }
    }

    throw std::runtime_error(
        "dataset root contains multiple nuScenes version directories; pass one specific version directory");
}

static json load_json_file(const fs::path &path)
{
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open JSON file: " + path.string());
    }

    json document;
    input >> document;
    return document;
}

static ResolvedNuScenesTarget resolve_target_sample(const json &scenes,
                                                    const json &samples,
                                                    const std::string &sequenceId)
{
    if (!scenes.is_array()) {
        throw std::runtime_error("scene.json does not contain an array");
    }
    if (!samples.is_array()) {
        throw std::runtime_error("sample.json does not contain an array");
    }

    for (const auto &scene : scenes) {
        const std::string name = scene.value("name", "");
        if (name != sequenceId) {
            continue;
        }

        ResolvedNuScenesTarget target;
        target.sampleToken = scene.value("first_sample_token", "");
        target.sceneName = name;
        target.sceneSampleCount = scene.value("nbr_samples", 0);
        target.hasSceneSampleCount = scene.contains("nbr_samples");

        if (target.sampleToken.empty()) {
            throw std::runtime_error("scene entry is missing first_sample_token");
        }

        return target;
    }

    for (const auto &sample : samples) {
        const std::string token = sample.value("token", "");
        if (token != sequenceId) {
            continue;
        }

        ResolvedNuScenesTarget target;
        target.sampleToken = token;
        return target;
    }

    throw std::runtime_error(
        "sequence_id must match a scene name or a sample token in the nuScenes tables");
}

static camera::CameraRole role_from_nuscenes_channel(const std::string &channel)
{
    if (channel == "CAM_FRONT") {return camera::CameraRole::Front;}
    if (channel == "CAM_BACK") {return camera::CameraRole::Rear;}
    if (channel == "CAM_FRONT_LEFT") {return camera::CameraRole::FrontLeft;}
    if (channel == "CAM_FRONT_RIGHT") {return camera::CameraRole::FrontRight;}
    if (channel == "CAM_BACK_LEFT") {return camera::CameraRole::RearLeft;}
    if (channel == "CAM_BACK_RIGHT") {return camera::CameraRole::RearRight;}
    return camera::CameraRole::Unknown;
}

static camera::CanonicalPose canonical_pose_from_nuscenes_quaternion(
    const json &rotationValue,
    const json &translationValue)
{
    if (!rotationValue.is_array() || rotationValue.size() != 4) {
        throw std::runtime_error("nuScenes rotation must be a quaternion array [w, x, y, z]");
    }
    if (!translationValue.is_array() || translationValue.size() != 3) {
        throw std::runtime_error("nuScenes translation must be a vec3 array");
    }

    const double w = rotationValue.at(0).get<double>();
    const double x = rotationValue.at(1).get<double>();
    const double y = rotationValue.at(2).get<double>();
    const double z = rotationValue.at(3).get<double>();
    const double norm = std::sqrt(w * w + x * x + y * y + z * z);
    if (norm <= 0.0) {
        throw std::runtime_error("nuScenes quaternion has zero length");
    }

    const double qw = w / norm;
    const double qx = x / norm;
    const double qy = y / norm;
    const double qz = z / norm;

    camera::CanonicalPose pose;
    // nuScenes stores sensor pose as a unit quaternion [w, x, y, z].
    // Build the equivalent 3x3 rotation matrix in row-major form for
    // CanonicalPose::rotation.
    pose.rotation = {{
        {{
            static_cast<float>(1.0 - 2.0 * (qy * qy + qz * qz)),
            static_cast<float>(2.0 * (qx * qy - qz * qw)),
            static_cast<float>(2.0 * (qx * qz + qy * qw)),
        }},
        {{
            static_cast<float>(2.0 * (qx * qy + qz * qw)),
            static_cast<float>(1.0 - 2.0 * (qx * qx + qz * qz)),
            static_cast<float>(2.0 * (qy * qz - qx * qw)),
        }},
        {{
            static_cast<float>(2.0 * (qx * qz - qy * qw)),
            static_cast<float>(2.0 * (qy * qz + qx * qw)),
            static_cast<float>(1.0 - 2.0 * (qx * qx + qy * qy)),
        }},
    }};

    for (std::size_t i = 0; i < 3; ++i) {
        pose.translation[i] = translationValue.at(i).get<float>();
    }

    return pose;
}

static camera::CanonicalIntrinsics canonical_intrinsics_from_matrix(const json &intrinsicValue)
{
    if (!intrinsicValue.is_array() || intrinsicValue.size() != 3) {
        throw std::runtime_error("camera_intrinsic must be a 3x3 array");
    }

    camera::CanonicalIntrinsics intrinsics;
    intrinsics.focal_length_px[0] = intrinsicValue.at(0).at(0).get<float>();
    intrinsics.skew = intrinsicValue.at(0).at(1).get<float>();
    intrinsics.principal_point_px[0] = intrinsicValue.at(0).at(2).get<float>();
    intrinsics.focal_length_px[1] = intrinsicValue.at(1).at(1).get<float>();
    intrinsics.principal_point_px[1] = intrinsicValue.at(1).at(2).get<float>();
    return intrinsics;
}

static std::vector<ResolvedCameraSample> resolve_camera_samples_for_target(
    const fs::path &versionRoot,
    const json &sampleData,
    const json &calibratedSensors,
    const json &sensors,
    const std::string &sampleToken)
{
    if (!sampleData.is_array()) {
        throw std::runtime_error("sample_data.json does not contain an array");
    }
    if (!calibratedSensors.is_array()) {
        throw std::runtime_error("calibrated_sensor.json does not contain an array");
    }
    if (!sensors.is_array()) {
        throw std::runtime_error("sensor.json does not contain an array");
    }

    std::unordered_map<std::string, std::string> sensorChannelByToken;
    for (const auto &sensor : sensors) {
        if (sensor.value("modality", "") != "camera") {
            continue;
        }
        const std::string token = sensor.value("token", "");
        const std::string channel = sensor.value("channel", "");
        if (!token.empty() && !channel.empty()) {
            sensorChannelByToken[token] = channel;
        }
    }

    std::unordered_map<std::string, std::string> calibratedSensorToChannel;
    for (const auto &calibratedSensor : calibratedSensors) {
        const std::string calibratedToken = calibratedSensor.value("token", "");
        const std::string sensorToken = calibratedSensor.value("sensor_token", "");
        auto sensorIt = sensorChannelByToken.find(sensorToken);
        if (calibratedToken.empty() || sensorIt == sensorChannelByToken.end()) {
            continue;
        }
        calibratedSensorToChannel[calibratedToken] = sensorIt->second;
    }

    std::vector<ResolvedCameraSample> cameraSamples;
    std::set<camera::CameraRole> seenRoles;

    for (const auto &sampleDataEntry : sampleData) {
        if (sampleDataEntry.value("sample_token", "") != sampleToken) {
            continue;
        }

        const std::string calibratedToken = sampleDataEntry.value("calibrated_sensor_token", "");
        auto calibratedIt = calibratedSensorToChannel.find(calibratedToken);
        if (calibratedIt == calibratedSensorToChannel.end()) {
            continue;
        }

        const std::string &channel = calibratedIt->second;
        const camera::CameraRole role = role_from_nuscenes_channel(channel);
        if (role == camera::CameraRole::Unknown) {
            continue;
        }

        const std::string relativePath = sampleDataEntry.value("filename", "");
        if (relativePath.empty()) {
            throw std::runtime_error("camera sample_data entry is missing filename");
        }
        if (!fs::is_regular_file(versionRoot / relativePath)) {
            throw std::runtime_error("camera image file is missing: " + relativePath);
        }
        if (!seenRoles.insert(role).second) {
            throw std::runtime_error("sample resolves multiple camera files for the same role");
        }

        ResolvedCameraSample sample;
        sample.role = role;
        sample.channelName = channel;
        sample.relativePath = relativePath;
        sample.calibratedSensorToken = calibratedToken;
        sample.width = sampleDataEntry.value("width", 0);
        sample.height = sampleDataEntry.value("height", 0);
        if (sample.width == 0 || sample.height == 0) {
            throw std::runtime_error("camera sample_data entry is missing image dimensions");
        }
        cameraSamples.push_back(std::move(sample));
    }

    if (cameraSamples.size() != 6) {
        throw std::runtime_error("target sample does not expose the expected 6 nuScenes camera channels");
    }

    return cameraSamples;
}

static camera::CameraRig build_camera_rig_for_target(
    const json &calibratedSensors,
    const std::vector<ResolvedCameraSample> &cameraSamples,
    const std::string &rigName)
{
    if (!calibratedSensors.is_array()) {
        throw std::runtime_error("calibrated_sensor.json does not contain an array");
    }

    std::unordered_map<std::string, const json *> calibratedSensorByToken;
    for (const auto &calibratedSensor : calibratedSensors) {
        const std::string token = calibratedSensor.value("token", "");
        if (!token.empty()) {
            calibratedSensorByToken[token] = &calibratedSensor;
        }
    }

    camera::CameraRig rig;
    rig.rig_name = rigName;

    for (const auto &cameraSample : cameraSamples) {
        auto calibratedIt = calibratedSensorByToken.find(cameraSample.calibratedSensorToken);
        if (calibratedIt == calibratedSensorByToken.end()) {
            throw std::runtime_error("missing calibrated_sensor entry for camera sample");
        }

        const json &calibratedSensor = *calibratedIt->second;

        camera::CameraRigCamera cameraDesc;
        cameraDesc.id = cameraSample.channelName;
        cameraDesc.role = cameraSample.role;
        cameraDesc.projection_model = camera::ProjectionModel::Perspective;
        cameraDesc.image_size = {cameraSample.width, cameraSample.height};
        cameraDesc.intrinsics =
            canonical_intrinsics_from_matrix(calibratedSensor.at("camera_intrinsic"));
        cameraDesc.distortion.model = camera::DistortionModel::None;
        cameraDesc.pose_vehicle =
            canonical_pose_from_nuscenes_quaternion(calibratedSensor.at("rotation"),
                                                    calibratedSensor.at("translation"));

        rig.cameras.push_back(std::move(cameraDesc));
    }

    return rig;
}

} // namespace

namespace svapp
{

NuScenesSource::NuScenesSource(std::string datasetRoot, std::string sequenceId):
    _datasetRoot(std::move(datasetRoot)),
    _sequenceId(std::move(sequenceId))
{
    _info.kind = videoio::SourceKind::NuScenes;
    _info.source_name = "nuscenes";
    _info.dataset_root = _datasetRoot;
    _info.sequence_id = _sequenceId;
    _info.contract.synchronized_samples = true;
    _info.contract.provides_sample_identity = true;
    _info.contract.provides_source_timestamp = true;
    _info.contract.provides_per_camera_timestamps = true;
    _info.contract.provides_ego_pose = true;
    _info.source_roles = {
        camera::CameraRole::Front,
        camera::CameraRole::FrontLeft,
        camera::CameraRole::FrontRight,
        camera::CameraRole::Rear,
        camera::CameraRole::RearLeft,
        camera::CameraRole::RearRight,
    };
}

bool NuScenesSource::open()
{
    try {
        if (_sequenceId.empty()) {
            throw std::runtime_error("sequence_id must not be empty");
        }

        _versionRoot.clear();
        _resolvedSampleToken.clear();
        _resolvedSceneName.clear();
        _opened = false;
        _rig = {};
        _cameraSamples.clear();
        _info.has_sequence_frame_count = false;
        _info.sequence_frame_count = 0;

        const fs::path versionRoot = resolve_nuscenes_version_root(_datasetRoot);
        const json scenes = load_json_file(versionRoot / "scene.json");
        const json samples = load_json_file(versionRoot / "sample.json");
        const json sampleData = load_json_file(versionRoot / "sample_data.json");
        const json calibratedSensors = load_json_file(versionRoot / "calibrated_sensor.json");
        const json sensors = load_json_file(versionRoot / "sensor.json");

        const ResolvedNuScenesTarget target =
            resolve_target_sample(scenes, samples, _sequenceId);

        const auto resolvedCameraSamples = resolve_camera_samples_for_target(versionRoot,
                                                                            sampleData,
                                                                            calibratedSensors,
                                                                            sensors,
                                                                            target.sampleToken);

        _versionRoot = versionRoot.string();
        _resolvedSampleToken = target.sampleToken;
        _resolvedSceneName = target.sceneName;
        _opened = true;

        _info.dataset_root = _versionRoot;
        _info.source_name = _resolvedSceneName.empty()
            ? "nuscenes_sample"
            : "nuscenes_scene";
        if (target.hasSceneSampleCount) {
            _info.sequence_frame_count = target.sceneSampleCount;
            _info.has_sequence_frame_count = true;
        }

        _rig = build_camera_rig_for_target(
            calibratedSensors,
            resolvedCameraSamples,
            _resolvedSceneName.empty() ? _resolvedSampleToken : _resolvedSceneName);

        _cameraSamples.reserve(resolvedCameraSamples.size());
        for (const auto &resolvedCameraSample : resolvedCameraSamples) {
            CameraSample sample;
            sample.role = resolvedCameraSample.role;
            sample.channel_name = resolvedCameraSample.channelName;
            sample.relative_path = resolvedCameraSample.relativePath;
            sample.calibrated_sensor_token = resolvedCameraSample.calibratedSensorToken;
            sample.width = resolvedCameraSample.width;
            sample.height = resolvedCameraSample.height;
            _cameraSamples.push_back(std::move(sample));
        }

        SPDLOG_INFO("Opened NuScenes metadata source [version_root='{}', scene='{}', sample_token='{}', camera_count={}]",
                    _versionRoot,
                    _resolvedSceneName.empty() ? "<direct-sample>" : _resolvedSceneName,
                    _resolvedSampleToken,
                    _cameraSamples.size());
        for (const auto &cameraSample : _cameraSamples) {
            SPDLOG_INFO("NuScenes camera sample: role='{}', channel='{}', image_size={}x{}, path='{}'",
                        camera_role_to_string(cameraSample.role),
                        cameraSample.channel_name,
                        cameraSample.width,
                        cameraSample.height,
                        cameraSample.relative_path);
        }

        return true;
    } catch (const std::exception &e) {
        SPDLOG_ERROR("Failed to open NuScenes source [dataset_root='{}', sequence_id='{}']",
                     _datasetRoot,
                     _sequenceId);
        SPDLOG_ERROR("{}", e.what());
        return false;
    }
}

const camera::CameraRig& NuScenesSource::rig() const
{
    return _rig;
}

const videoio::SourceInfo& NuScenesSource::info() const
{
    return _info;
}

bool NuScenesSource::get_next_frame(videoio::FramePacket &packet)
{
    (void)packet;
    if (!_opened) {
        SPDLOG_ERROR("NuScenesSource::get_next_frame() called before successful open()");
        return false;
    }

    SPDLOG_ERROR("NuScenesSource sample loading is not implemented yet [sample_token='{}']",
                 _resolvedSampleToken);
    return false;
}

bool NuScenesSource::release_frame(const videoio::FramePacket &packet)
{
    (void)packet;
    return true;
}

} // namespace svapp
