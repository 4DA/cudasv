#ifndef CONVERSION_UTILS_HPP
#define CONVERSION_UTILS_HPP

#include <cmath>

#include "engine/extrinsics.hpp"

namespace camera
{

struct EulerAngles
{
    float roll;
    float pitch;
    float yaw;

    // normalize angle to (-M_PI, M_PI]
    void normalize();
};


// Converts roll/pitch/yaw to surroundview coordinate system matrix
void roll_pitch_yaw_2_mat(EulerAngles angles,
                          Extrinsics::Rotation &R);

}

#endif
