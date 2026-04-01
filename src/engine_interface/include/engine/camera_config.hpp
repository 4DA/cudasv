#ifndef CAMERA_CONFIG_HPP
#define CAMERA_CONFIG_HPP

#include <cstdint>

#include <engine/extrinsics.hpp>
#include <engine/intrinsics.hpp>

namespace camera
{

enum CameraID
{
    CAMERA_RIGHT,
    CAMERA_LEFT,
    CAMERA_FRONT,
    CAMERA_REAR,
    CAMERAS_TOTAL,
    CAMERAS_COUNT = CAMERAS_TOTAL,
};

struct CameraConfig
{
    CameraID camera_id;

    // Camera intrinsics
    Intrinsics intrinsics;

    // Camera extrinsics
    // stored as camera -> vehicle transform matrix
    Extrinsics extrinsics;
};

}

#endif // CAMERA_CONFIG_HPP
