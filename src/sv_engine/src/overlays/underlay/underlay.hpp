#ifndef SV_UNDERLAY_HPP
#define SV_UNDERLAY_HPP

#include <vector>

#include <cuda_runtime.h>

#include <engine/engine.hpp>

#include <rf/renderer/cudarf/cudarf.hpp>

#include <rf/renderer/gltf_common.hpp>
#include <rf/renderer/image.hpp>
#include <rf/renderer/scene.hpp>

/// root component name for car underlay
extern const std::string underlayCompoName;

class Underlay
{
public:
    int init(cudarf::pipe::Ctx *desc, rf::Scene &scene, const engine::Config *config, cudaStream_t cuStream);
    std::shared_ptr<cudarf::Material> get_material() { return material; }

private:
    const CarUnderlayConfig *underlay_config;

    std::shared_ptr<cudarf::Material> material;

    rf::Image underlay_image_description;
    cudaTextureObject_t underlay_image = 0;

    rf::NaiveMeshPtr generate_underlay_mesh(const engine::Config *config);
};


#endif  // SV_UNDERLAY_HPP
