#include <limits>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/printf.h>

#include "rotator.hpp"

#include <rf/renderer/glm_common.hpp>


using namespace rf;

const float PI = glm::pi<float>();

// azimuthal must belong [0, 2π) interval
float rf::normalize_azimuthal(float azimuthal)
{
    if (azimuthal < 0.0f) {
        azimuthal = 2.0f * PI + azimuthal;
    }

    azimuthal = std::fmod(azimuthal, 2.0f * PI);

    return azimuthal;
}

SphericalCoord SphericalCoord::interpolate(const SphericalCoord &c1,
                                           const SphericalCoord &c2,
                                           float t)
{
    float start = normalize_azimuthal(c1.azimuthal);
    float end = normalize_azimuthal(c2.azimuthal);

    float min = std::min(start, end);
    float max = std::max(start, end);

    // length of arc from start to end that is passing through zero degree point
    float dist_zero = min + (2.0f * PI - max);

    // If the length of the arc that passes through zero is less than that of
    // the ordinary arc, then adjust the endpoint closer to 2π in the negative
    // space to ensure the interval [start; end) is continuous.
    if (std::abs(end - start) > dist_zero) {
        if (end > start) {
            end = end - 2.0f * PI;
        } else {
            start = start - 2.0f * PI;
        }
    }

    return SphericalCoord(c1.polar + (c2.polar - c1.polar) * t,
                          start + (end - start) * t);
}

SphericalBoundary::SphericalBoundary(float start_polar,
                                     float end_polar,
                                     float start_azimuthal,
                                     float end_azimuthal):
    start_azimuthal(start_azimuthal),
    end_azimuthal(end_azimuthal),
    start_polar(start_polar),
    end_polar(end_polar)
{
    assert(start_azimuthal != end_azimuthal);
    assert(start_polar != end_polar);

    assert (start_polar >= 0.0f && end_polar >= 0.0f && start_azimuthal >= 0.0f);
    assert (start_polar <= 2.0f * PI && end_polar <= 2.0f * PI);

    assert(start_azimuthal < end_azimuthal);
    assert(start_azimuthal < PI + std::numeric_limits<float>::epsilon());

    // clamp polar to π range
    end_polar = glm::clamp(end_polar, 0.0f, PI);
}

bool SphericalBoundary::point_inside(const SphericalCoord &c) const
{
    // polar must belong to [0; π] interval and must not wrap
    if (c.polar < 0 || c.polar > PI)
    {
        return false;
    }

    bool azimuthalOk = false;
    bool polarOk = true;

    // verify if the angle falls within the arc defined by [start_azimuthal; end_azimuthal] and [start_polar; end_polar]
    if (start_azimuthal > end_azimuthal)
    {
        bool less_than_start = c.azimuthal <= start_azimuthal;
        bool less_than_end = c.azimuthal <= end_azimuthal;

        SPDLOG_TRACE("{}", fmt::sprintf("is_inside[rev] azimuthal %f in [%f; %f] = %d",
        c.azimuthal, start_azimuthal, end_azimuthal, less_than_start == less_than_end));

        azimuthalOk = less_than_start == less_than_end;
    }
    else
    {
        SPDLOG_TRACE("{}", fmt::sprintf("is_inside[dir] azimuthal %f in [%f; %f] = %d",
            c.azimuthal, start_azimuthal, end_azimuthal, (c.azimuthal >= start_azimuthal) && (c.azimuthal <= end_azimuthal)));

        azimuthalOk = (c.azimuthal >= start_azimuthal) && (c.azimuthal <= end_azimuthal);
    }

    polarOk = (c.polar >= start_polar) && (c.polar <= end_polar);

    SPDLOG_TRACE("{}", fmt::sprintf("is_inside[dir] polar %f in [%f; %f] = %d", c.polar, start_polar, end_polar, polarOk));

    return azimuthalOk && polarOk;
}


glm::vec3 EllipticRotator::from_spherical(float a, float b, float c, SphericalCoord coord)
{
    return glm::vec3(a * std::sin(coord.polar) * std::cos(coord.azimuthal),
                     b * std::sin(coord.polar) * std::sin(coord.azimuthal),
                     c * std::cos(coord.polar));
}

glm::vec3 EllipticRotator::get_world_up(const SphericalCoord &coord)
{

    // Camera is positioned vertically; calculate the up vector using the azimuthal angle
    if (coord.polar == 0.0f) {
        return glm::vec3(-std::cos(coord.azimuthal), -std::sin(coord.azimuthal), 0.0f);
    } else { // Align with the Z-axis; this will suffice
        return glm::vec3(0.0f, 0.0f, 1.0f);
    }
}

TRSTransform EllipticRotator::get_orientation(SphericalCoord c) const
{
    glm::vec3 newUp = get_world_up(c);
    return get_look_at_trs(from_spherical(c) + center, center, newUp);
}

EllipticRotator EllipticRotator::mix(const EllipticRotator &from, const EllipticRotator &to, float t)
{
    return EllipticRotator {
        glm::mix(from.A, to.A, t),
        glm::mix(from.B, to.B, t),
        glm::mix(from.C, to.C, t)
    };
}

// try to get camera position in spherical coordinates, relative to the lookAt point
SphericalCoord EllipticRotator::get_position(const VirtualCamera &camera) const
{
    glm::vec3 p = camera.transform.translation;
    glm::vec3 up = camera.get_up_vector();

    // y / x = B / A * tan(azimuthal);
    // azimuthal = arctan(Ay / Bx);

    SphericalCoord result(std::acos(p.z / C), std::atan2(A * p.y, B * p.x));

    // if camera is oriented striclty vertically, then get azimuthal from camera up
    // vector that must lie in XY plane
    if (result.polar == 0.0f) {
        result.azimuthal = std::atan2(up.y, up.x) + PI;
        result.azimuthal = std::fmod(result.azimuthal, 2.0f * PI);
    }

    return result;
}

RectangularBoundary::RectangularBoundary(float top,
                                         float bottom,
                                         float left,
                                         float right):
    top(top),
    bottom(bottom),
    left(left),
    right(right)
{
    assert(top != bottom);
    assert(left != right);

    assert(top > bottom);
    assert(left > right);
}

bool RectangularBoundary::is_inside(const glm::vec3 &t) const
{
    if (t.x > top) return false;
    if (t.x < bottom) return false;

    if (t.y > left) return false;
    if (t.y < right) return false;

    return true;
}
