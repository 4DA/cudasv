#ifndef RF_POINT_LIGHT_COMPONENT_HPP
#define RF_POINT_LIGHT_COMPONENT_HPP

#include <rf/renderer/glm_common.hpp>
#include <rf/renderer/scene_component.hpp>
#include <rf/renderer/trs_transform.hpp>

namespace rf
{

// light component contains point light and its position
struct PointLightComponent: public SceneComponent {
    // position in SVS space
    glm::vec3 position;

    // GLTF KHR_LIGHTS_punctual extension uses candela (lm/sr) as its unit
    float intensity;

    // name      - Name of component. Must be unique in scene.
    // toLocal   - Local space transform affecting geometry and children
    // parent    - Parent, null if this is root component
    // intensity - Radiant intensity (in watt)
    PointLightComponent(const std::string &name,
                        const TRSTransform &toLocal,
                        SceneComponent &parent,
                        float intensity);
};
}

#endif
