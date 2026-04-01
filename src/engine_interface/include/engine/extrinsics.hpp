#ifndef EXTRINSICS_HPP
#define EXTRINSICS_HPP

#include <array>

namespace camera
{

// The transformation from the camera coordinate system to the vehicle
// coordinate system is stored as f(x) = Rx+T
//
struct Extrinsics
{
    using Rotation = std::array<float, 9>;
    using Translation = std::array<float, 3>;

    // Camera view rotation matrix (column-major)
    Rotation R;

    // Camera view translation vector
    Translation T;
};

}

#endif // EXTRINSICS_HPP
