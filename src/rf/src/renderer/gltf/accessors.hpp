#ifndef RF_RENDERER_CUDARF_GLTF_ACCESSORS_HPP
#define RF_RENDERER_CUDARF_GLTF_ACCESSORS_HPP

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <rf/renderer/glm_common.hpp>
#include <rf/renderer/gltf_common.hpp>
#include <rf/renderer/mesh_geometry.hpp>

namespace loader::gltf
{

std::optional<glm::vec2> to_glm_vec2(const std::vector<double> &std_vec);
std::optional<glm::vec3> to_glm_vec3(const std::vector<double> &std_vec);
std::optional<glm::vec4> to_glm_vec4(const std::vector<double> &std_vec);

bool init_attributes_accessor(const tinygltf::Model &model,
                              rf::AttributesAccessor &attribPtr,
                              const std::string &name,
                              const tinygltf::Accessor &accessor);

} // namespace loader::gltf

#endif
