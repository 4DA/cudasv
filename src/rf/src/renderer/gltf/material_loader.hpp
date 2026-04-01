#ifndef RF_RENDERER_CUDARF_GLTF_MATERIAL_LOADER_HPP
#define RF_RENDERER_CUDARF_GLTF_MATERIAL_LOADER_HPP

#include <memory>
#include <string>

#include <rf/renderer/cudarf/types.hpp>
#include <rf/renderer/gltf_common.hpp>

namespace rf
{
class Scene;
}

namespace cudarf
{
struct Material;
}

namespace loader::gltf
{

std::shared_ptr<cudarf::Material>
create_material(const tinygltf::Model &model,
                rf::Scene &scene,
                const tinygltf::Primitive &primitive,
                const std::string &prefix,
                cudaStream_t cuStream);

} // namespace loader::gltf

#endif
