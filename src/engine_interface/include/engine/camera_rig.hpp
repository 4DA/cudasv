#ifndef CAMERA_RIG_HPP
#define CAMERA_RIG_HPP

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace camera
{

enum class ProjectionModel
{
    Perspective,
    Fisheye,
};

enum class DistortionModel
{
    None,
    RadialTangential,
    FisheyePolynomial4,
    FisheyePolynomial8,
    Equidistant,
};

enum class CameraRole
{
    Front,
    Rear,
    Left,
    Right,
    FrontLeft,
    FrontRight,
    RearLeft,
    RearRight,
    Unknown,
};

struct CanonicalIntrinsics
{
    std::array<float, 2> focal_length_px = {0.0f, 0.0f};
    std::array<float, 2> principal_point_px = {0.0f, 0.0f};
    float skew = 0.0f;
};

struct CanonicalDistortion
{
    DistortionModel model = DistortionModel::None;
    std::vector<float> coefficients;
};

struct CanonicalPose
{
    // Stored as JSON rows for readability. Runtime conversion into
    // camera::Extrinsics::R must transpose into column-major flat storage.
    std::array<std::array<float, 3>, 3> rotation = {{
        {{1.0f, 0.0f, 0.0f}},
        {{0.0f, 1.0f, 0.0f}},
        {{0.0f, 0.0f, 1.0f}},
    }};
    std::array<float, 3> translation = {0.0f, 0.0f, 0.0f};
};

struct CameraRigCamera
{
    std::string id;
    CameraRole role = CameraRole::Unknown;
    ProjectionModel projection_model = ProjectionModel::Fisheye;
    std::array<uint32_t, 2> image_size = {0, 0};
    CanonicalIntrinsics intrinsics;
    CanonicalDistortion distortion;
    CanonicalPose pose_vehicle;
};

struct CameraRig
{
    uint32_t schema_version = 1;
    std::string rig_name;
    std::vector<CameraRigCamera> cameras;
};

}

#endif // CAMERA_RIG_HPP
