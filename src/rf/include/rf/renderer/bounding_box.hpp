#ifndef RF_BOUNDING_BOX_HPP
#define RF_BOUNDING_BOX_HPP

#include <rf/renderer/glm_common.hpp>
#include <rf/renderer/trs_transform.hpp>

namespace rf
{

struct BoundingBox
{
    BoundingBox union_with(const BoundingBox &other) const;

    glm::vec3 min() const {return vertex_min;}
    glm::vec3 max() const {return vertex_max;}
    glm::vec3 center() const;

    float height() const {return max().z - min().z;}

    BoundingBox transform(const TRSTransform &transform) const;

    BoundingBox transform(const glm::mat4 &transform) const;

    bool is_good() const;

    void dump(std::string msg = "") const;

    BoundingBox(const glm::vec3 &min, const glm::vec3 &max);
    BoundingBox(const BoundingBox &);

    BoundingBox(): vertex_min(UNSET_MIN),
                   vertex_max(UNSET_MAX) {}

    BoundingBox & operator=(BoundingBox const &bb) = default;

    static const glm::vec3 UNSET_MIN;
    static const glm::vec3 UNSET_MAX;

private:
    glm::vec3 vertex_min;
    glm::vec3 vertex_max;
};

struct HitRecord
{
    float t;
    glm::vec3 n;
};

bool intersect(const glm::vec3 O,
               const glm::vec3 &r,
               const BoundingBox &box,
               HitRecord &hit);

} //namespace rf

#endif
