#include <rf/renderer/point_light_component.hpp>

using namespace rf;

// 1 model unit = 1000 SVS units
constexpr float UNIT_SCALE = 1000.0f;
constexpr float UNIT_SCALE_2 = (UNIT_SCALE * UNIT_SCALE);

PointLightComponent::PointLightComponent(const std::string &name,
                                         const TRSTransform &toLocal,
                                         SceneComponent &parent,
                                         float intensity):
    SceneComponent(name, toLocal, &parent),
    intensity(intensity * UNIT_SCALE_2) {}
