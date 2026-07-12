#include <spdlog/spdlog.h>

#include "sky_fade.hpp"

using namespace engine::view;

struct SkyFadeParams
{
    float radius;     /// cylinder radius (in mm)
    float fade_start; /// intersection height (in mm) when fading should start
    float fade_end;   /// intersection height (in mm) when fading should be maximum
};

/*
  good reference:
  http://skuld.bmsc.washington.edu/people/merritt/graphics/quadrics.html

  The general quadric surface equation is

  F(x, y, z) = Ax2 + By2 + Cz2 + Dxy + Exz + Fyz + Gx + Hy + Iz + J = 0

  Then substitute in ray equation R(t) = Ro + Rd and we get quadratic equation:

  Aqt2 + Bqt + Cq = 0 with

  Aq = Axd2 + Byd2 + Czd2 + Dxdyd + Exdzd + Fydzd
  Bq = 2*Axoxd + 2*Byoyd + 2*Czozd + D(xoyd + yoxd) + E(xozd + zoxd) + F(yozd + ydzo) + Gxd + Hyd + Izd
  Cq = Axo2 + Byo2 + Czo2 + Dxoyo + Exozo + Fyozo + Gxo + Hyo + Izo + J

  for cylinder x^2 + y^2 = r^2 we have A = 1, B = 1 ; J = -r^2 ; C, D, E, F, G, H, I = 0;
*/

static std::vector<float> intersect_cylinder(const glm::vec3 &d, const glm::vec3 &o, float r)
{
    float A = 1.0f;
    float B = 1.0f;

    float Aq = A * d.x * d.x + B * d.y * d.y;
    float Bq = 2.0f * A * o.x * d.x + 2.0f * B * o.y * d.y;
    float Cq = A * o.x * o.x + B * o.y * o.y - r * r;

    if (Aq == 0.0f) {return {-Cq / Bq};}

    float D = Bq * Bq - 4.0f * Aq * Cq;

    if (D < 0.0f) {return {};}

    return {(-Bq - std::sqrt(D)) / (2.0f * Aq),
            (-Bq + std::sqrt(D)) / (2.0f * Aq)};
}

static glm::vec3 compute_intersection(const SkyFadeParams &params,
                                      const rf::VirtualCamera &camera)
{
    assert(camera.m_projection.is_perspective);
    glm::vec3 origin = camera.transform.translation;
    glm::vec3 direction = camera.get_ray(glm::vec2(0.5f, 1.0f)); // center-top direction

    std::vector<float> ts =
        intersect_cylinder(direction, origin, params.radius);

    // remove all intersection points that are behind camera
    std::remove_if(ts.begin(), ts.end(),
        [](float val) {if (val < 0.0f) return true; else return false;});

    if (ts.empty()) {return glm::vec3(0.0f);}

    std::vector<glm::vec3> points;

    for (auto t: ts) {
        points.push_back(origin + direction * t);
    }

    // sort points in descending order by their height
    std::sort(points.begin(), points.end(),
        [](const glm::vec3 &p1, const glm::vec3 &p2) {
            if (p1.z > p2.z) return true;
            else return false;
        });

    SPDLOG_TRACE("cylinder intersection t = {:2.f}, p = {:.2f}, {:2.f}, {:2.f}", ts[0], points[0].x, points[0].y, points[0].z);

    return points[0];
}


namespace engine
{
namespace view
{

int compute_sky_fade(const SkyFade3D *fade_config,
                     const rf::VirtualCamera &camera,
                     float &fadeStart,
                     float &fadeMax)
{
    SkyFadeParams params({fade_config->radius_mm,
                              fade_config->start_mm,
                              fade_config->start_mm +
                              fade_config->gradient_mm});

    glm::vec3 point = compute_intersection(params, camera);

    if (point.z < 0.0f) {
        return 0;
    }

    glm::mat4 MVP = camera.projection_matrix * rf::get_camera_matrix(camera.transform);

    // UV of fade start
    glm::vec3 start_ws(point.x, point.y, params.fade_start);
    glm::vec4 start_hom = MVP * glm::vec4(start_ws, 1.0f);
    glm::vec3 start = glm::vec3(start_hom / start_hom.w);

    // UV of fade max
    glm::vec3 end_ws(point.x, point.y, params.fade_end);
    glm::vec4 end_hom = MVP * glm::vec4(end_ws, 1.0f);
    glm::vec3 end = glm::vec3(end_hom / end_hom.w);

    fadeStart = start.y / 2.0f + 0.5f;
    fadeMax = end.y / 2.0f + 0.5f;

    return 0;
}

}
}
