#include <cmath>

#include "conversion_utils.hpp"

namespace camera
{

// Make matrix for rotation around Z axis
static void z_rotation(float z_angle, Extrinsics::Rotation &rot)
{
    float  sin_ = std::sin(z_angle);
    float  cos_ = std::cos(z_angle);

    rot[0 + 3*0] = cos_;
    rot[0 + 3*1] = -sin_;
    rot[0 + 3*2] = 0.f;

    rot[1 + 3*0] = sin_;
    rot[1 + 3*1] = cos_;
    rot[1 + 3*2] = 0.f;

    rot[2 + 3*0] = 0.f;
    rot[2 + 3*1] = 0.f;
    rot[2 + 3*2] = 1.f;
}

// Make matrix for rotation around Y axis
static void y_rotation(float y_angle, Extrinsics::Rotation &rot)
{
    float  sin_ = std::sin(y_angle);
    float  cos_ = std::cos(y_angle);

    rot[0 + 3*0] = cos_;
    rot[0 + 3*1] = 0.f;
    rot[0 + 3*2] = sin_;

    rot[1 + 3*0] = 0.f;
    rot[1 + 3*1] = 1.f;
    rot[1 + 3*2] = 0.f;

    rot[2 + 0*3] = -sin_;
    rot[2 + 1*3] = 0.f;
    rot[2 + 2*3] = cos_;
}

// Make matrix for rotation around X axis
static void x_rotation(float x_angle, Extrinsics::Rotation &rot)
{
    float  sin_ = std::sin(x_angle);
    float  cos_ = std::cos(x_angle);

    rot[0 + 3*0] = 1.f;
    rot[0 + 3*1] = 0.f;
    rot[0 + 3*2] = 0.f;

    rot[1 + 3*0] = 0.f;
    rot[1 + 3*1] = cos_;
    rot[1 + 3*2] = -sin_;

    rot[2 + 0*3] = 0.f;
    rot[2 + 1*3] = sin_;
    rot[2 + 2*3] = cos_;
}


//  Multiply two 3x3 matrixes (column major order)
//  C = A * B
static void mul_mat(const Extrinsics::Rotation &A,
                    const Extrinsics::Rotation &B,
                    Extrinsics::Rotation &C)
{
    for(int y = 0; y< 3; y++) {
        const float* aRow = &A[y];
        for(int x = 0; x< 3; x++) {
            const float* bCol =  &B[3*x];
            float ret = 0;
            ret = bCol[0]*aRow[0] + bCol[1]*aRow[3] + bCol[2]*aRow[6];
            C[y + x*3] = ret;
        }
    }
}


// build rotation matrix from x, y, z angles (in degrees)
static void euler2matrix(float x, float y, float z, Extrinsics::Rotation &r_out)
{
    float TO_RAD = ((float)M_PI / 180.f);
    Extrinsics::Rotation r_x, r_y, r_z, r_yx;

    x_rotation(x * TO_RAD, r_x);
    y_rotation(y * TO_RAD, r_y);
    z_rotation(z * TO_RAD, r_z);
    mul_mat(r_y, r_x, r_yx);
    mul_mat(r_z, r_yx, r_out);
}

// normalize angle to (-M_PI, M_PI]
float normalize_angle(float in)
{
    float angle = std::remainder(in, (float)(2. * M_PI));
    if (angle <= (float)(-M_PI)) {
        angle = in + (float)(2. * M_PI);
    }

    if (angle > (float)(M_PI)) {
        angle = angle - (float)(2. * M_PI);
    }
    return angle;
}


// normalizes yaw, pitch, roll to to (-M_PI, M_PI]
void normalize_yaw_pitch_roll(float &yawInOut,
                              float &pitchInOut,
                              float &rollInOut)
{
    float yaw = normalize_angle(yawInOut);
    float pitch = normalize_angle(pitchInOut);
    float roll = normalize_angle(rollInOut);

    if (std::abs(pitch) > (float)(M_PI / 2.)) {
        yaw = normalize_angle(yaw + (float)(M_PI));
        pitch = normalize_angle(-pitch + (float)(M_PI));
        roll = normalize_angle(roll + (float)(M_PI));
    }

    yawInOut   = yaw;
    pitchInOut = pitch;
    rollInOut  = roll;
}

// Converts roll/pitch/yaw to surroundview coordinate system matrix
void roll_pitch_yaw_2_mat(EulerAngles angles,
                          Extrinsics::Rotation &R)
{
    // Vehicle frame: X forward, Y left, Z up.
    // OpenCV camera frame C: X right, Y down, Z forward.
    // below matrix is camera frame -> vehicle frame conversion

    Extrinsics::Rotation CV_to_V = {
        0.f, -1.f, 0.f,
        0.f, 0.f, -1.f,
        1.f, 0.f, 0.f
    };

    // build matrix from euler angles in vehicle frame
    Extrinsics::Rotation rot_V;
    euler2matrix(angles.roll, angles.pitch, angles.yaw, rot_V);

    // R = rot_V * CV_to_V
    // = camera -> vehicle
    mul_mat(rot_V, CV_to_V, R);
}

void EulerAngles::normalize()
{
    yaw = normalize_angle(yaw);
    pitch = normalize_angle(pitch);
    roll = normalize_angle(roll);

    if (std::abs(pitch) > (float)(M_PI / 2.)) {
        yaw = normalize_angle(yaw + (float)(M_PI));
        pitch = normalize_angle(-pitch + (float)(M_PI));
        roll = normalize_angle(roll + (float)(M_PI));
    }
}
}
