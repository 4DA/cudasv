#ifndef RF_RENDERER_GLTF_LOADER_HPP
#define RF_RENDERER_GLTF_LOADER_HPP

#include <rf/renderer/gltf_common.hpp>
#include <rf/renderer/scene.hpp>
#include <rf/renderer/animation.hpp>

namespace cudarf { namespace pipe { struct Ctx; } }

namespace loader
{

struct ShapeModifier {
    glm::vec3 recenter;
    rf::SceneComponent *reparent;
};

/// if callback return value is true, then loader doesn't add the new component
/// to the scene hierarchy, assuming the callback already handled it
using PrimitiveComponentCB = std::function<bool(const rf::PrimitiveComponent &, rf::Scene &)>;

bool load_gltf_model(cudarf::pipe::Ctx *desc,
              const std::string &input_filename,
              rf::Scene &scene,
              rf::SceneComponent *parent,
              const std::string &prefix,
              rf::AnimationMap &animations,
              tinygltf::Model &model,
              PrimitiveComponentCB inserter_cb,
              cudaStream_t cuStream);

rf::SceneComponent *
add_naive_mesh(cudarf::pipe::Ctx *desc,
               const std::string &compo_name,
               rf::NaiveMeshPtr meshPtr,
               const std::shared_ptr<cudarf::Material> &material,
               const rf::TRSTransform &transform,
               rf::Scene &scene,
               rf::SceneComponent *parent,
               cudaStream_t cuStream);
}

#endif
