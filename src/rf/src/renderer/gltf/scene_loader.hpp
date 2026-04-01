#ifndef RF_RENDERER_CUDARF_GLTF_SCENE_LOADER_HPP
#define RF_RENDERER_CUDARF_GLTF_SCENE_LOADER_HPP

#include <string>

#include <rf/renderer/gltf_loader.hpp>

namespace loader::gltf
{

bool load_scene(cudarf::pipe::Ctx *desc,
                const tinygltf::Model &model,
                rf::Scene &scene,
                rf::SceneComponent *parent,
                const std::string &namePrefix,
                loader::PrimitiveComponentCB cb,
                cudaStream_t cuStream);

} // namespace loader::gltf

#endif
