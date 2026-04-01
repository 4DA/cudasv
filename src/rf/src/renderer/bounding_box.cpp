#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/printf.h>

#include <array>

#include <rf/renderer/glm_common.hpp>
#include "bounding_box.hpp"


using namespace rf;

const glm::vec3 BoundingBox::UNSET_MIN = glm::vec3(std::numeric_limits<float>::max());
const glm::vec3 BoundingBox::UNSET_MAX = glm::vec3(std::numeric_limits<float>::lowest());

namespace rf
{

// plane equation: (p - p0) * n = 0
// ray equation: O + r * t = p
// plugging p => (O + r * t - p0) * n = 0
// t = (p0 - O) * n / (n * r), where * is inner product
static bool intersect_plane(const glm::vec3 &n,
                            const glm::vec3 &p0,
                            const glm::vec3 &O,
                            const glm::vec3 &r,
                            HitRecord &rec)
{
    // all vectors must be normalized
    float denom = dot(n, r);
    if (std::abs(denom) > 1e-6) {

        float nT = dot(p0 - O, n) / denom;
        if (nT >= 0.0) {
            rec = HitRecord{nT, n};
            return true;
        }
    }

    return false;
}

static bool intersect_aa_rect(const glm::vec3 &O,
                              const glm::vec3 &r,
                              const glm::vec3 &rect_min,
                              const glm::vec3 &rect_max,
                              HitRecord &hit)
{
    glm::vec3 N;
    glm::vec3 diag = rect_max - rect_min;
    glm::vec3 p0 =  rect_min;
    bool skip_x = false, skip_y = false, skip_z = false;

    if (diag.x < 2.0f * std::numeric_limits<float>::epsilon()) {
        N = glm::vec3(1.0f, 0.0f, 0.0f);
        skip_x = true;
    }
    else if (diag.y < 2.0f * std::numeric_limits<float>::epsilon()) {
        N = glm::vec3(0.0f, 1.0f, 0.0f);
        skip_y = true;
    }
    else if (diag.z < 2.0f * std::numeric_limits<float>::epsilon()) {
        N = glm::vec3(0.0f, 0.0f, 1.0f);
        skip_z = true;
    }
    else {
        assert(false /* not axis aligned rect */);
    }

    if (intersect_plane(N, p0, O, r, hit)) {
        glm::vec3 P = O + r * hit.t;
        bool x_ok = skip_x || (P.x >= rect_min.x && P.x <= rect_max.x);
        bool y_ok = skip_y || (P.y >= rect_min.y && P.y <= rect_max.y);
        bool z_ok = skip_z || (P.z >= rect_min.z && P.z <= rect_max.z);

        return (x_ok && y_ok && z_ok);
    }

    return false;
}

bool intersect(const glm::vec3 O,
               const glm::vec3 &r,
               const BoundingBox &box,
               HitRecord &hit)
{
    HitRecord nearest{std::numeric_limits<float>::max(), glm::vec3(0.0f)};

    // check bottom Z rect
    if (intersect_aa_rect(O, r,
                          glm::vec3(box.min().x, box.min().y, box.min().z),
                          glm::vec3(box.max().x, box.max().y, box.min().z), hit)) {
        if (hit.t < nearest.t) {
            nearest = hit;
        }
    }

    // check top Z rect
    if (intersect_aa_rect(O, r,
                          glm::vec3(box.min().x, box.min().y, box.max().z),
                          glm::vec3(box.max().x, box.max().y, box.max().z), hit) &&
        hit.t < nearest.t)
    {
        nearest = hit;
    }

    // check min X rect
    if (intersect_aa_rect(O, r,
                          glm::vec3(box.min().x, box.min().y, box.min().z),
                          glm::vec3(box.min().x, box.max().y, box.max().z), hit) &&
        hit.t < nearest.t)
    {
        nearest = hit;
    }

    // check max X rect
    if (intersect_aa_rect(O, r,
                          glm::vec3(box.max().x, box.min().y, box.min().z),
                          glm::vec3(box.max().x, box.max().y, box.max().z), hit) &&
        hit.t < nearest.t)
    {
        nearest = hit;
    }

    // check min Y rect
    if (intersect_aa_rect(O, r,
                          glm::vec3(box.min().x, box.min().y, box.min().z),
                          glm::vec3(box.max().x, box.min().y, box.max().z), hit) &&
        hit.t < nearest.t)
    {
        nearest = hit;
    }

    // check max Y rect
    if (intersect_aa_rect(O, r,
                          glm::vec3(box.min().x, box.max().y, box.min().z),
                          glm::vec3(box.max().x, box.max().y, box.max().z), hit) &&
        hit.t < nearest.t)
    {
        nearest = hit;
    }

    if (nearest.t < std::numeric_limits<float>::max()) {
        hit = nearest;
        return true;
    }
    else {
        return false;
    }
}

} // namespace rf

BoundingBox::BoundingBox(const glm::vec3 &vertex_min, const glm::vec3 &vertex_max):
    vertex_min(vertex_min), vertex_max(vertex_max)
{
    if (is_good()) {
        assert(vertex_min.x <= vertex_max.x);
        assert(vertex_min.y <= vertex_max.y);
        assert(vertex_min.z <= vertex_max.z);

        // if bb is a point than smth is wrong with it
        assert(vertex_min.x != vertex_max.x ||
               vertex_min.y != vertex_max.y ||
               vertex_min.z != vertex_max.z);
    }
}

BoundingBox::BoundingBox(const BoundingBox &bb):
    vertex_min(bb.min()),
    vertex_max(bb.max()) {}

BoundingBox BoundingBox::union_with(const BoundingBox &other) const
{
    // if other is not defined, return this
    if (!other.is_good()) {
        return *this;
    }

    if (is_good()) {
        return BoundingBox(glm::min(vertex_min, other.min()),
                           glm::max(vertex_max, other.max()));
    } else {
        return BoundingBox(other.min(), other.max());
    }
}

BoundingBox BoundingBox::transform(const glm::mat4 &mat) const
{
    if (is_good()) {
        glm::vec4 new_pos_max = mat * glm::vec4(vertex_max, 1.0f);
        glm::vec4 new_pos_min = mat * glm::vec4(vertex_min, 1.0f);
        glm::vec3 vertex_max = glm::max(glm::vec3(new_pos_max), glm::vec3(new_pos_min));
        glm::vec3 vertex_min = glm::min(glm::vec3(new_pos_max), glm::vec3(new_pos_min));
        return BoundingBox(vertex_min, vertex_max);
    } else {
        return BoundingBox();
    }
}

BoundingBox BoundingBox::transform(const TRSTransform &t) const
{
    if (is_good()) {
        std::array<glm::vec3, 8> points = {
            glm::vec3(vertex_min.x, vertex_min.y, vertex_min.z),
            glm::vec3(vertex_min.x, vertex_min.y, vertex_max.z),
            glm::vec3(vertex_min.x, vertex_max.y, vertex_min.z),
            glm::vec3(vertex_min.x, vertex_max.y, vertex_max.z),
            glm::vec3(vertex_max.x, vertex_min.y, vertex_min.z),
            glm::vec3(vertex_max.x, vertex_min.y, vertex_max.z),
            glm::vec3(vertex_max.x, vertex_max.y, vertex_min.z),
            glm::vec3(vertex_max.x, vertex_max.y, vertex_max.z),
        };

        glm::vec3 new_min = BoundingBox::UNSET_MIN;
        glm::vec3 new_max = BoundingBox::UNSET_MAX;

        for (const glm::vec3 &p: points) {
            glm::vec3 res = glm::vec3(t.apply(glm::vec4(p, 1.0f)));
            new_min = glm::min(new_min, res);
            new_max = glm::max(new_max, res);
        }

        return BoundingBox(new_min, new_max);
    } else {
        return BoundingBox();
    }
}

glm::vec3 BoundingBox::center() const
{
    return (vertex_min + vertex_max) / 2.0f;
}

bool BoundingBox::is_good() const
{
    return vertex_min != UNSET_MIN && vertex_max != UNSET_MAX;
}

void BoundingBox::dump(std::string msg) const
{
    glm::vec3 c = center();
    SPDLOG_INFO("{}", fmt::sprintf("%s min: {%.2f, %.2f, %.2f}, max: {%.2f, %.2f, %.2f}, c: {%.2f, %.2f, %.2f}",
           msg.c_str(),
           vertex_min.x, vertex_min.y, vertex_min.z,
           vertex_max.x, vertex_max.y, vertex_max.z,
           c.x, c.y, c.z));
}
