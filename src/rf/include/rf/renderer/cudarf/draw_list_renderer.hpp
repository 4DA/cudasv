#pragma once

#include <map>
#include <memory>
#include <iterator>

#include <rf/renderer/glm_common.hpp>

#include <rf/renderer/cudarf/cudarf_camera.hpp>

#include <rf/renderer/virtual_camera.hpp>
#include <rf/renderer/scene.hpp>

namespace cudarf
{

class DrawListRenderer
{
public:
    template<rf::render_pass_type T>
    struct WorkDescription {
        cudarf::DrawList pbr;
        cudarf::DrawList flat;
        cudarf::CommonUniforms common;
    };

    struct Stats {
        std::size_t pbrSize;
        std::size_t flatSize;
    };

    DrawListRenderer(const rf::Scene &scene,
                     cudarf::profiling::Events *eventDB);

    DrawListRenderer() = delete;

    int register_material(std::shared_ptr<cudarf::Material> material, const std::string &name);

    template<rf::render_pass_type T>
    void add_work(DrawListRenderer::WorkDescription<T> &desc,
                  const rf::Scene &scene,
                  const std::string &root,
                  rf::VirtualCamera &camera,
                  const std::set<std::string> &exclusionSet,
                  cudarf::LayerComputeFn layerFn = nullptr) const
        {
            scene.build_draw_list(root,
                                  T,
                                  cudarf::SHADER_TYPE_PBR,
                                  camera,
                                  materialPtrMap,
                                  exclusionSet,
                                  layerFn,
                                  desc.pbr);

            scene.build_draw_list(root,
                                  T,
                                  cudarf::SHADER_TYPE_UNLIT,
                                  camera,
                                  materialPtrMap,
                                  exclusionSet,
                                  layerFn,
                                  desc.flat);

            desc.common = cudarf::make_common(&camera);
        }


    Stats render(cudarf::pipe::Ctx* rasterization_desc,
                 rf::Scene &scene,
                 rf::VirtualCamera& camera,
                 const DrawListRenderer::WorkDescription<rf::RENDER_PASS_OPAQUE> &work,
#ifdef WITH_TAA
                 const DrawListRenderer::WorkDescription<rf::RENDER_PASS_OPAQUE> &workHist,
#endif
                 bool withOpaqueVisibuf,
                 cudarf::Framebuffer output,
                 unsigned int frameCounter,
                 cudaStream_t cuStream);

    Stats render(cudarf::pipe::Ctx* rasterization_desc,
                 rf::Scene &scene,
                 rf::VirtualCamera& camera,
                 const DrawListRenderer::WorkDescription<rf::RENDER_PASS_TRANSLUCENT> &work,
#ifdef WITH_TAA
                 const DrawListRenderer::WorkDescription<rf::RENDER_PASS_TRANSLUCENT> &workHist,
#endif
                 cudarf::Framebuffer output,
                 cudarf::ShaderType shaderType,
                 unsigned int frameCounter,
                 cudaStream_t cuStream);

private:
    /// dictionary that stores internal material Id -> MaterialPtr records
    ///
    cudarf::MaterialMap objMaterials;

    /// dictionary that stores MaterialPtr -> internal material ID records
    ///
    cudarf::MaterialPtrMap materialPtrMap;
    std::shared_ptr<cudarf::profiling::Events> opaqueTime = nullptr;
    std::shared_ptr<cudarf::profiling::Events> opaqueTimeFlat = nullptr;
    std::shared_ptr<cudarf::profiling::Events> translucentTime = nullptr;
    std::shared_ptr<cudarf::profiling::Events> translucentTimeFlat = nullptr;
};

void sort_draw_list(cudarf::DrawListRenderer::WorkDescription<rf::RENDER_PASS_TRANSLUCENT> &work,
                    rf::VirtualCamera& camera);
}
