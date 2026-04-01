#ifndef RF_RENDERER_CUDARF_GLTF_ANIMATION_LOADER_HPP
#define RF_RENDERER_CUDARF_GLTF_ANIMATION_LOADER_HPP

#include <string>

#include <rf/renderer/animation.hpp>
#include <rf/renderer/gltf_common.hpp>

namespace loader::gltf
{

rf::AnimationMap load_animations(const tinygltf::Model &model, const std::string &prefix);

} // namespace loader::gltf

#endif
