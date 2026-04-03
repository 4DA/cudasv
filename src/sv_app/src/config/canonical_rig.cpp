#include "canonical_rig.hpp"

#include <fstream>
#include <stdexcept>

#include <spdlog/spdlog.h>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace
{

camera::CameraRole camera_role_from_string(const std::string &value)
{
    if (value == "front") {return camera::CameraRole::Front;}
    if (value == "rear") {return camera::CameraRole::Rear;}
    if (value == "left") {return camera::CameraRole::Left;}
    if (value == "right") {return camera::CameraRole::Right;}
    if (value == "front_left") {return camera::CameraRole::FrontLeft;}
    if (value == "front_right") {return camera::CameraRole::FrontRight;}
    if (value == "rear_left") {return camera::CameraRole::RearLeft;}
    if (value == "rear_right") {return camera::CameraRole::RearRight;}
    return camera::CameraRole::Unknown;
}

camera::ProjectionModel projection_model_from_string(const std::string &value)
{
    if (value == "fisheye") {return camera::ProjectionModel::Fisheye;}
    if (value == "perspective") {return camera::ProjectionModel::Perspective;}
    throw std::runtime_error("unsupported projection_model: " + value);
}

camera::DistortionModel distortion_model_from_string(const std::string &value)
{
    if (value == "none") {return camera::DistortionModel::None;}
    if (value == "radial_tangential") {return camera::DistortionModel::RadialTangential;}
    if (value == "fisheye_polynomial4") {return camera::DistortionModel::FisheyePolynomial4;}
    if (value == "fisheye_polynomial8") {return camera::DistortionModel::FisheyePolynomial8;}
    if (value == "equidistant") {return camera::DistortionModel::Equidistant;}
    throw std::runtime_error("unsupported distortion.model: " + value);
}

std::array<float, 2> parse_float2(const json &node, const char *field_name)
{
    if (!node.is_array() || node.size() != 2) {
        throw std::runtime_error(std::string(field_name) + " must be an array of size 2");
    }

    return {
        node.at(0).get<float>(),
        node.at(1).get<float>(),
    };
}

std::array<float, 3> parse_float3(const json &node, const char *field_name)
{
    if (!node.is_array() || node.size() != 3) {
        throw std::runtime_error(std::string(field_name) + " must be an array of size 3");
    }

    return {
        node.at(0).get<float>(),
        node.at(1).get<float>(),
        node.at(2).get<float>(),
    };
}

std::array<std::array<float, 3>, 3> parse_rotation_rows(const json &node)
{
    if (!node.is_array() || node.size() != 3) {
        throw std::runtime_error("pose_vehicle.rotation must be a 3x3 array");
    }

    std::array<std::array<float, 3>, 3> out{};

    for (std::size_t row = 0; row < 3; ++row) {
        const auto &row_node = node.at(row);
        if (!row_node.is_array() || row_node.size() != 3) {
            throw std::runtime_error("pose_vehicle.rotation must be a 3x3 array");
        }

        for (std::size_t col = 0; col < 3; ++col) {
            out[row][col] = row_node.at(col).get<float>();
        }
    }

    return out;
}

camera::CameraRig parse_camera_rig(const json &doc)
{
    camera::CameraRig rig;

    rig.schema_version = doc.value("schema_version", 1u);
    rig.rig_name = doc.value("rig_name", std::string());

    std::string rotation_repr = "matrix3x3";
    if (doc.contains("coordinate_system")) {
        const auto &coordinate_system = doc.at("coordinate_system");
        rotation_repr =
            coordinate_system.value("rotation_representation", std::string("matrix3x3"));
    }

    if (rotation_repr != "matrix3x3") {
        throw std::runtime_error("runtime loader only supports coordinate_system.rotation_representation = matrix3x3");
    }

    const auto &cameras = doc.at("cameras");
    if (!cameras.is_array()) {
        throw std::runtime_error("cameras must be an array");
    }

    rig.cameras.reserve(cameras.size());

    for (const auto &camera_node: cameras) {
        camera::CameraRigCamera camera_desc;

        camera_desc.id = camera_node.value("id", std::string());
        camera_desc.role = camera_role_from_string(camera_node.at("role").get<std::string>());
        camera_desc.projection_model =
            projection_model_from_string(camera_node.at("projection_model").get<std::string>());

        const auto &image_size = camera_node.at("image_size");
        if (!image_size.is_array() || image_size.size() != 2) {
            throw std::runtime_error("image_size must be an array of size 2");
        }
        camera_desc.image_size = {
            image_size.at(0).get<uint32_t>(),
            image_size.at(1).get<uint32_t>(),
        };

        const auto &intrinsics = camera_node.at("intrinsics");
        camera_desc.intrinsics.focal_length_px =
            parse_float2(intrinsics.at("focal_length_px"), "intrinsics.focal_length_px");
        camera_desc.intrinsics.principal_point_px =
            parse_float2(intrinsics.at("principal_point_px"), "intrinsics.principal_point_px");
        camera_desc.intrinsics.skew = intrinsics.value("skew", 0.0f);

        const auto &distortion = camera_node.at("distortion");
        const std::string distortion_model =
            distortion.value("model", std::string("fisheye_polynomial4"));
        camera_desc.distortion.model =
            distortion_model_from_string(distortion_model);
        camera_desc.distortion.coefficients =
            distortion.at("coefficients").get<std::vector<float>>();

        const auto &pose_vehicle = camera_node.at("pose_vehicle");
        camera_desc.pose_vehicle.rotation =
            parse_rotation_rows(pose_vehicle.at("rotation"));
        camera_desc.pose_vehicle.translation =
            parse_float3(pose_vehicle.at("translation"), "pose_vehicle.translation");

        rig.cameras.push_back(std::move(camera_desc));
    }

    return rig;
}

} // namespace

int load_canonical_rig(camera::CameraRig *rig, const std::string &path)
{
    if (!rig) {
        SPDLOG_ERROR("CameraRig output pointer is null");
        return -1;
    }

    std::ifstream in(path);
    if (!in.is_open()) {
        SPDLOG_ERROR("Failed to open canonical rig file: {}", path);
        return -1;
    }

    try {
        json doc;
        in >> doc;
        *rig = parse_camera_rig(doc);
    } catch (const std::exception &e) {
        SPDLOG_ERROR("Failed to parse canonical rig [{}]: {}", path, e.what());
        return -1;
    }

    return 0;
}
