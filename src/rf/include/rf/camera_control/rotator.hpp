#ifndef RF_ELLIPTIC_HPP
#define RF_ELLIPTIC_HPP

#include <rf/renderer/glm_common.hpp>
#include <rf/renderer/trs_transform.hpp>
#include <rf/renderer/virtual_camera.hpp>

namespace rf
{

float normalize_azimuthal(float phi);

struct SphericalCoord {
    // Polar angle, valid range: [0; π], where 0 is the topmost point
    float polar;

    // Azimuthal angle, valid range: [0; 2π), where 0 represents Y = 0 and X > 0
    float azimuthal;

    // Adds two spherical coordinates as translations from the origin
    const SphericalCoord operator+(const SphericalCoord &other) const {
        return SphericalCoord(polar + other.polar, azimuthal + other.azimuthal);
    }

    // Multiplies the current coordinate by an integer, summing it with itself N times
    const SphericalCoord operator*(unsigned int num) const {
        SphericalCoord result = *this;

        for (; num > 0; num--) {
            result = result + *this;
        }

        return result;
    }

    // Performs linear interpolation between two spherical coordinates
    static SphericalCoord interpolate(const SphericalCoord &c1,
                                      const SphericalCoord &c2,
                                      float t);

    SphericalCoord(float polar, float azimuthal):
        polar(polar),
        azimuthal(normalize_azimuthal(azimuthal)) {}

    SphericalCoord(): polar(0.0f), azimuthal(0.0f) {}
};

// Defines the boundaries for possible positions in spherical coordinates
struct SphericalBoundary
{
    float start_azimuthal;
    float end_azimuthal;
    float start_polar;
    float end_polar;

    SphericalBoundary() = default;

    SphericalBoundary(float start_polar,
                      float end_polar,
                      float start_azimuthal,
                      float end_azimuthal);

    bool point_inside(const SphericalCoord &) const;
};

// Rectangular area that limits topview camera controller
struct RectangularBoundary
{
    float top;
    float bottom;
    float left;
    float right;

    RectangularBoundary() = default;

    RectangularBoundary(float top,
                        float bottom,
                        float left,
                        float right);

    bool is_inside(const glm::vec3 &) const;
};


// Geometric figure represented by the implicit equation:
// x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 0,
// where a, b, and c are positive real values.

// The axes of the ellipsoid align with the coordinate axes. The points (a, 0, 0), (0, b, 0)
// and (0, 0, c) lie on the surface of the ellipsoid. The line segments extending from the origin
// to these points are referred to as the principal semi-axes of the ellipsoid.

struct EllipticRotator {
    glm::vec3 center = glm::vec3(0.0f, 0.0f, 0.0f);

    float A; // half-length of X principal axis
    float B; // half-length of Y principal axis
    float C; // half-length of Z principal axis

    EllipticRotator(): A(0.0f), B(0.0f), C(0.0f) {}

    EllipticRotator(float A, float B, float C):
        A(A), B(B), C(C) {}

    // Convert spherical coordinates to world Cartesian coordinates
    glm::vec3 from_spherical(SphericalCoord coord) const { return from_spherical(A, B, C, coord); }

    // Convert spherical coordinates to world Euclidean coordinates
    static glm::vec3 from_spherical(float a, float b, float c, SphericalCoord coord);

    // Retrieve camera orientation based on spherical coordinates
    TRSTransform get_orientation(SphericalCoord coord) const;

    // Obtain the normalized world up vector for camera matrix construction
    static glm::vec3 get_world_up(const SphericalCoord &coord);

    // Estimate spherical coordinates from the camera's position
    SphericalCoord get_position(const VirtualCamera &camera) const;

    static EllipticRotator mix(const EllipticRotator &from, const EllipticRotator &to, float t);

    // Uniformly scale all axes of the ellipsoid, adjusting the camera's distance to the focal points
    void scale(float mult) {
        A *= mult;
        B *= mult;
        C *= mult;
    }

    bool is_inside(glm::vec3 point) {
        return ((point.x / A) * (point.x / A) +
                (point.y / B) * (point.y / B) +
                (point.z / C) * (point.z / C)) <= 1.0f;
    }

    bool is_inside_box(glm::vec3 point) {
        return (point.x < A) && (point.y < B) && (point.z < C);
    }
};

}

#endif
