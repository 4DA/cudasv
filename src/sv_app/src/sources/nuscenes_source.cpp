#include <array>
#include <algorithm>
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
    uint64_t timestampNs = 0;
};

struct ResolvedNuScenesRoots
{
    fs::path dataRoot;
    fs::path versionRoot;
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

static ResolvedNuScenesRoots resolve_nuscenes_roots(const fs::path &datasetRoot)
{
    if (!fs::exists(datasetRoot)) {
        throw std::runtime_error("dataset root does not exist");
    }
    if (!fs::is_directory(datasetRoot)) {
        throw std::runtime_error("dataset root is not a directory");
    }

    if (has_required_table_files(datasetRoot)) {
        ResolvedNuScenesRoots roots;
        roots.versionRoot = datasetRoot;
        if (datasetRoot.filename().string().rfind("v1.0-", 0) == 0) {
            roots.dataRoot = datasetRoot.parent_path();
        } else {
            roots.dataRoot = datasetRoot;
        }
        return roots;
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
        return {datasetRoot, candidates.front()};
    }

    for (const fs::path &candidate : candidates) {
        const std::string name = candidate.filename().string();
        if (name == "v1.0-mini") {
            return {datasetRoot, candidate};
        }
    }
    for (const fs::path &candidate : candidates) {
        const std::string name = candidate.filename().string();
        if (name == "v1.0-trainval") {
            return {datasetRoot, candidate};
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
    const fs::path &dataRoot,
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
        if (!sampleDataEntry.value("is_key_frame", false)) {
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
        if (!fs::is_regular_file(dataRoot / relativePath)) {
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
        sample.timestampNs = sampleDataEntry.value("timestamp", 0ULL);
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

static std::vector<std::string> collect_sample_tokens_for_target(
    const json &samples,
    const ResolvedNuScenesTarget &target)
{
    if (!samples.is_array()) {
        throw std::runtime_error("sample.json does not contain an array");
    }

    if (target.sceneName.empty()) {
        return {target.sampleToken};
    }

    std::unordered_map<std::string, std::string> nextTokenBySampleToken;
    for (const auto &sample : samples) {
        const std::string token = sample.value("token", "");
        if (token.empty()) {
            continue;
        }
        nextTokenBySampleToken[token] = sample.value("next", "");
    }

    std::vector<std::string> sampleTokens;
    std::set<std::string> visitedTokens;
    std::string token = target.sampleToken;
    while (!token.empty()) {
        if (!visitedTokens.insert(token).second) {
            throw std::runtime_error("scene sample chain contains a loop");
        }

        sampleTokens.push_back(token);

        const auto nextIt = nextTokenBySampleToken.find(token);
        if (nextIt == nextTokenBySampleToken.end()) {
            throw std::runtime_error("scene sample token is missing from sample.json");
        }

        token = nextIt->second;
    }

    if (target.hasSceneSampleCount &&
        sampleTokens.size() != target.sceneSampleCount) {
        throw std::runtime_error("scene sample chain length does not match nbr_samples");
    }

    return sampleTokens;
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
        _decoded_sample_ready = false;
        _frameId = 0;
        _currentSampleIndex = 0;
        _rig = {};
        _samples.clear();
        _decodedFrames.clear();
        _info.has_sequence_frame_count = false;
        _info.sequence_frame_count = 0;

        const ResolvedNuScenesRoots roots = resolve_nuscenes_roots(_datasetRoot);
        const fs::path &dataRoot = roots.dataRoot;
        const fs::path &versionRoot = roots.versionRoot;
        const json scenes = load_json_file(versionRoot / "scene.json");
        const json samples = load_json_file(versionRoot / "sample.json");
        const json sampleData = load_json_file(versionRoot / "sample_data.json");
        const json calibratedSensors = load_json_file(versionRoot / "calibrated_sensor.json");
        const json sensors = load_json_file(versionRoot / "sensor.json");

        const ResolvedNuScenesTarget target =
            resolve_target_sample(scenes, samples, _sequenceId);
        const std::vector<std::string> sampleTokens =
            collect_sample_tokens_for_target(samples, target);

        std::vector<ResolvedCameraSample> firstResolvedCameraSamples;
        _samples.reserve(sampleTokens.size());
        for (std::size_t sampleIndex = 0; sampleIndex < sampleTokens.size(); ++sampleIndex) {
            const auto resolvedCameraSamples = resolve_camera_samples_for_target(dataRoot,
                                                                                sampleData,
                                                                                calibratedSensors,
                                                                                sensors,
                                                                                sampleTokens[sampleIndex]);

            if (sampleIndex == 0) {
                firstResolvedCameraSamples = resolvedCameraSamples;
            }

            SampleFrame sampleFrame;
            sampleFrame.sample_token = sampleTokens[sampleIndex];
            // nuScenes exposes slightly different timestamps per camera sample;
            // use the earliest one as the sample-level timestamp anchor.
            sampleFrame.source_timestamp_ns =
                std::min_element(resolvedCameraSamples.begin(),
                                 resolvedCameraSamples.end(),
                                 [](const ResolvedCameraSample &lhs,
                                    const ResolvedCameraSample &rhs) {
                                     return lhs.timestampNs < rhs.timestampNs;
                                 })->timestampNs;

            sampleFrame.cameras.reserve(resolvedCameraSamples.size());
            for (const auto &resolvedCameraSample : resolvedCameraSamples) {
                CameraSample sample;
                sample.role = resolvedCameraSample.role;
                sample.channel_name = resolvedCameraSample.channelName;
                sample.relative_path = resolvedCameraSample.relativePath;
                sample.calibrated_sensor_token = resolvedCameraSample.calibratedSensorToken;
                sample.width = resolvedCameraSample.width;
                sample.height = resolvedCameraSample.height;
                sample.timestamp_ns = resolvedCameraSample.timestampNs;
                sampleFrame.cameras.push_back(std::move(sample));
            }

            _samples.push_back(std::move(sampleFrame));
        }

        _dataRoot = dataRoot.string();
        _versionRoot = versionRoot.string();
        _resolvedSampleToken = target.sampleToken;
        _resolvedSceneName = target.sceneName;
        _opened = true;

        _info.dataset_root = _dataRoot;
        _info.source_name = _resolvedSceneName.empty()
            ? "nuscenes_sample"
            : "nuscenes_scene";
        if (target.hasSceneSampleCount) {
            _info.sequence_frame_count = target.sceneSampleCount;
            _info.has_sequence_frame_count = true;
        }

        _rig = build_camera_rig_for_target(
            calibratedSensors,
            firstResolvedCameraSamples,
            _resolvedSceneName.empty() ? _resolvedSampleToken : _resolvedSceneName);

        SPDLOG_INFO("Opened NuScenes metadata source [data_root='{}', version_root='{}', scene='{}', sample_token='{}', camera_count={}]",
                    _dataRoot,
                    _versionRoot,
                    _resolvedSceneName.empty() ? "<direct-sample>" : _resolvedSceneName,
                    _resolvedSampleToken,
                    _samples.empty() ? 0 : _samples.front().cameras.size());
        for (const auto &cameraSample : _samples.front().cameras) {
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
    if (!_opened) {
        SPDLOG_ERROR("NuScenesSource::get_next_frame() called before successful open()");
        return false;
    }
    if (_samples.empty()) {
        SPDLOG_ERROR("NuScenesSource has no resolved samples");
        return false;
    }

    if (!_decoded_sample_ready) {
        const SampleFrame &sample = _samples[_currentSampleIndex];
        _decodedFrames.clear();
        _decodedFrames.resize(sample.cameras.size());

        for (std::size_t index = 0; index < sample.cameras.size(); ++index) {
            const std::string imagePath =
                (std::filesystem::path(_dataRoot) / sample.cameras[index].relative_path).string();
            std::string errorMessage;
            if (!videoio::load_rgb_image(imagePath, _decodedFrames[index], &errorMessage)) {
                SPDLOG_ERROR("Failed to decode NuScenes camera image '{}': {}",
                             imagePath,
                             errorMessage);
                _decodedFrames.clear();
                return false;
            }
        }

        _decoded_sample_ready = true;
    }

    const SampleFrame &sample = _samples[_currentSampleIndex];

    packet.metadata.frame_id = _frameId++;
    packet.metadata.source_frame_sequence = _currentSampleIndex;
    packet.metadata.sample_id = sample.sample_token;
    packet.metadata.has_sample_id = true;
    packet.metadata.synchronized_cameras = true;
    packet.metadata.source_timestamp_ns = sample.source_timestamp_ns;
    packet.metadata.has_source_timestamp = true;

    packet.cameras.clear();
    packet.cameras.reserve(sample.cameras.size());

    for (std::size_t index = 0; index < sample.cameras.size(); ++index) {
        videoio::SourceCameraFrame cameraFrame;
        cameraFrame.role = sample.cameras[index].role;
        cameraFrame.data = _decodedFrames[index].data;
        cameraFrame.userdata = _decodedFrames[index].owner.get();
        cameraFrame.width = _decodedFrames[index].width;
        cameraFrame.height = _decodedFrames[index].height;
        cameraFrame.stride = _decodedFrames[index].stride;
        cameraFrame.timestamp_ns = sample.cameras[index].timestamp_ns;
        cameraFrame.has_timestamp = true;
        cameraFrame.valid = true;
        packet.cameras.push_back(cameraFrame);
    }

    return true;
}

bool NuScenesSource::release_frame(const videoio::FramePacket &packet)
{
    (void)packet;
    return true;
}

bool NuScenesSource::step_next_sample()
{
    return step_samples(1);
}

bool NuScenesSource::step_previous_sample()
{
    if (_samples.size() <= 1) {
        return true;
    }

    return step_samples(-1);
}

bool NuScenesSource::step_samples(int delta)
{
    if (_samples.size() <= 1 || delta == 0) {
        return true;
    }

    const int64_t sampleCount = static_cast<int64_t>(_samples.size());
    const int64_t currentIndex = static_cast<int64_t>(_currentSampleIndex);
    const int64_t wrappedIndex =
        ((currentIndex + static_cast<int64_t>(delta)) % sampleCount + sampleCount) % sampleCount;

    _currentSampleIndex = static_cast<std::size_t>(wrappedIndex);
    _decoded_sample_ready = false;
    return true;
}

std::size_t NuScenesSource::sample_count() const
{
    return _samples.size();
}

std::size_t NuScenesSource::current_sample_index() const
{
    return _currentSampleIndex;
}

} // namespace svapp
