#ifndef INTRINSICS_H
#define INTRINSICS_H

#include <cstdint>
namespace camera
{

enum CameraLensType
{
    CAMERA_LENS_FISHEYE,
    CAMERA_LENS_NARROW,
};

struct Intrinsics
{
    using KMat = std::array<float, 9>;
    using DistortionCoeffs = std::array<float, 8>;

    // 3x3 intrinsic parameters matrix (scale, shear, and principal point)
    // Upper-triangular, stored in column-major order
    KMat camera;

    // Distortion coefficients
    // Can be either:
    // - k1, k2, k3, and k4 distortion coefficients for Kannala-Brandt model
    // - k1, k2, p1, p2, k3, and [k4, k5, k6] for narrow lens
    DistortionCoeffs distortion_coeffs;

    // 3x3 matrix with k1, k2, k3 coefficients used in the formula
    // k1 * r^4 + k2 * r^2 + k3
    // where r is the fisheye radius
    // Coefficients for each color channel (RGB) are stored as:
    // [k1R k1G k1B]
    // [k2R k2G k2B]
    // [k3R k3G k3B]
    std::array<float, 9> lens_shading_coeffs;

    CameraLensType   lens_model;
};

}
#endif  /* INTRINSICS_HPP */
