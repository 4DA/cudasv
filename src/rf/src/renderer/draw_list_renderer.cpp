#include <spdlog/spdlog.h>

#include "cudarf/draw_list_renderer.hpp"

#include "gltf_loader.hpp"
#include "trs_transform.hpp"

cudarf::DrawListRenderer::DrawListRenderer(const rf::Scene &scene,
                                           cudarf::profiling::Events *eventDB)
{
    for (auto it : scene.get_materials()) {
        register_material(it.second, it.first);
    }

    if (eventDB) {
        opaqueTime = eventDB->add_child("car opaque pbr");
        opaqueTimeFlat = eventDB->add_child("car opaque flat");
        translucentTime = eventDB->add_child("car xlucent pbr");
        translucentTimeFlat = eventDB->add_child("car xlucent flat");
    }
}

int cudarf::DrawListRenderer::register_material(std::shared_ptr<cudarf::Material> material, const std::string &name)
{
    (void)name;
    int matID = objMaterials.size();

    materialPtrMap[material] = matID;
    objMaterials[matID] = material;

    SPDLOG_DEBUG("material id={} [name=`{}`]", matID, name.c_str());

    return matID;
}

cudarf::DrawListRenderer::Stats
cudarf::DrawListRenderer::render(cudarf::pipe::Ctx* rasterization_desc,
                                 rf::Scene &scene,
                                 rf::VirtualCamera& camera,
                                 const DrawListRenderer::WorkDescription<rf::RENDER_PASS_OPAQUE> &work,
#ifdef WITH_TAA
                                 const DrawListRenderer::WorkDescription<rf::RENDER_PASS_OPAQUE> &workHist,
#endif
                                 bool withOpaqueVisibuf,
                                 cudarf::Framebuffer output,
                                 unsigned int frameCounter,
                                 bool testBinTiler,
                                 cudaStream_t cuStream)
{
    assert(rasterization_desc);

    // TODO: fill this up in DrawListRenderer::add_work to avoid possibility of
    // having different camera-derived parameters during ::add_work() and ::render()
    auto camera_translation = make_float3(camera.transform.translation.x,
                                          camera.transform.translation.y,
                                          camera.transform.translation.z);

    std::vector<cudarf::CUDARFLight> lightList;

    for (auto compIt: scene.get_lights()) {
        rf::PointLightComponent &comp = *compIt.second;
        auto trans = comp.toWorld.translation;

        cudarf::CUDARFLight light {
            .intensity = comp.intensity,
            .position = make_float3(trans.x, trans.y, trans.z),
            .range = 10000.0
        };

        lightList.push_back(light);
    }

    const rf::IBL &ibl = scene.get_ibl();

    assert(ibl);
    cudarf::PBRParams pbrCommon{camera_translation, camera.exposure,
        lightList, ibl.get_sh_matrix(), ibl.brdfLUT.view().textureObject,
        ibl.specular.view().textureObject,
        ibl.specular.view().mipLevels};

    assert(ibl.specular.view().textureObject);
    assert(pbrCommon.specular);

    if (work.pbr) {
        int opaqueTotal;
        if (opaqueTime != nullptr) {
            opaqueTotal = opaqueTime->start_interval("run_pipe", cuStream);
        }

        cudarf::pipe::run_pipe(rasterization_desc,
                                     // cull backfaces, no blending
                                     cudarf::RenderParams {
                                         true,
                                         false
                                     },
                                     work.pbr.uniforms,
#ifdef WITH_TAA
                                     workHist.pbr.uniforms,
#endif
                                     work.pbr.drawPacketIds,
                                     work.pbr.matIds,
                                     objMaterials,
                                     cudarf::pipe::LaunchConfig(true,
                                                                withOpaqueVisibuf,
                                                                frameCounter,
                                                                output,
                                                                opaqueTime,
                                                                testBinTiler),
                                     cuStream);

        if (opaqueTime != nullptr) {opaqueTime->stop_interval(opaqueTotal);}
    }

    if (work.flat) {
        int opaqueTotal;
        if (opaqueTimeFlat != nullptr) {opaqueTotal = opaqueTimeFlat->start_interval("run_pipe", cuStream);}

        cudarf::pipe::run_pipe(rasterization_desc,
                             // cull backfaces, no blending
                             cudarf::RenderParams {
                                 true,
                                 false
                             },
                             work.flat.uniforms,
#ifdef WITH_TAA
                             workHist.flat.uniforms,
#endif
                             work.flat.drawPacketIds,
                             work.flat.matIds,
                             objMaterials,
                             cudarf::pipe::LaunchConfig(true,
                                                        withOpaqueVisibuf,
                                                        frameCounter,
                                                        output,
                                                        opaqueTimeFlat,
                                                        testBinTiler),
                             cuStream);

        if (opaqueTimeFlat != nullptr) {opaqueTimeFlat->stop_interval(opaqueTotal);}
    }

    return DrawListRenderer::Stats {
        work.pbr.drawPacketIds.size(),
        work.flat.drawPacketIds.size(),
    };
}

cudarf::DrawListRenderer::Stats
cudarf::DrawListRenderer::render(cudarf::pipe::Ctx* rasterization_desc,
                                 rf::Scene &scene,
                                 rf::VirtualCamera& camera,
                                 const DrawListRenderer::WorkDescription<rf::RENDER_PASS_TRANSLUCENT> &work,
#ifdef WITH_TAA
                                 const DrawListRenderer::WorkDescription<rf::RENDER_PASS_TRANSLUCENT> &workHist,
#endif
                                 cudarf::Framebuffer output,
                                 cudarf::ShaderType shaderType,
                                 unsigned int frameCounter,
                                 bool testBinTiler,
                                 cudaStream_t cuStream)
{
    assert(rasterization_desc);

    auto camera_translation = make_float3(camera.transform.translation.x,
                                          camera.transform.translation.y,
                                          camera.transform.translation.z);

    std::vector<cudarf::CUDARFLight> lightList;

    for (auto compIt: scene.get_lights()) {
        rf::PointLightComponent &comp = *compIt.second;
        auto trans = comp.toWorld.translation;

        cudarf::CUDARFLight light {
            .intensity = comp.intensity,
            .position = make_float3(trans.x, trans.y, trans.z),
            .range = 10000.0
        };

        lightList.push_back(light);
    }

    const rf::IBL &ibl = scene.get_ibl();

    assert(ibl.specular.view().textureObject);

    cudarf::PBRParams pbrCommon{camera_translation, camera.exposure,
        lightList, ibl.get_sh_matrix(),
        ibl.brdfLUT.view().textureObject, ibl.specular.view().textureObject,
        ibl.specular.view().mipLevels};

    if (shaderType == cudarf::SHADER_TYPE_PBR && work.pbr) {
        int translucentTotal;
        if (translucentTime != nullptr) {translucentTotal = translucentTime->start_interval("run_pipe", cuStream);}

        cudarf::pipe::run_pipe(rasterization_desc,
                             // don't render backfacing glass to save some time
                             // blending enabled
                             cudarf::RenderParams {
                                 true,
                                 true
                             },
                             work.pbr.uniforms,
#ifdef WITH_TAA
                             workHist.pbr.uniforms,
#endif
                             work.pbr.drawPacketIds,
                             work.pbr.matIds,
                             objMaterials,
                             cudarf::pipe::LaunchConfig(true,
                                                        false,
                                                        frameCounter,
                                                        output,
                                                        translucentTime,
                                                        testBinTiler),
                             cuStream
            );

        if (translucentTime != nullptr) {translucentTime->stop_interval(translucentTotal);}
    }

    if (shaderType == cudarf::SHADER_TYPE_UNLIT && work.flat) {
        int translucentTotal;

        if (translucentTime != nullptr) {translucentTotal = translucentTimeFlat->start_interval("run_pipe", cuStream);}

        cudarf::pipe::run_pipe(rasterization_desc,
                                     // don't render backfacing glass to save some time
                                     // blending enabled
                                     cudarf::RenderParams {
                                         true,
                                         true
                                     },
                                     work.flat.uniforms,
#ifdef WITH_TAA
                                     workHist.flat.uniforms,
#endif
                                     work.flat.drawPacketIds,
                                     work.flat.matIds,
                                     objMaterials,
                                     cudarf::pipe::LaunchConfig(true,
                                                                false,
                                                                frameCounter,
                                                                output,
                                                                translucentTimeFlat,
                                                                testBinTiler),
                                     cuStream
            );

        if (translucentTime != nullptr) {translucentTimeFlat->stop_interval(translucentTotal);}
    }

    return DrawListRenderer::Stats {
        work.pbr.drawPacketIds.size(),
        work.flat.drawPacketIds.size(),
    };
}


cudarf::DrawListRenderer::Stats
cudarf::DrawListRenderer::render(cudarf::pipe::Ctx* rasterization_desc,
                                 rf::Scene &scene,
                                 rf::VirtualCamera& camera,
                                 const DrawListRenderer::WorkDescription<rf::RENDER_PASS_UI> &work,
#ifdef WITH_TAA
                                 const DrawListRenderer::WorkDescription<rf::RENDER_PASS_UI> &workHist,
#endif
                                 cudarf::Framebuffer output,
                                 cudarf::ShaderType shaderType,
                                 unsigned int frameCounter,
                                 bool testBinTiler,
                                 cudaStream_t cuStream)
{
    assert(rasterization_desc);

    auto camera_translation = make_float3(camera.transform.translation.x,
                                          camera.transform.translation.y,
                                          camera.transform.translation.z);

    std::vector<cudarf::CUDARFLight> lightList;

    for (auto compIt: scene.get_lights()) {
        rf::PointLightComponent &comp = *compIt.second;
        auto trans = comp.toWorld.translation;

        cudarf::CUDARFLight light {
            .intensity = comp.intensity,
            .position = make_float3(trans.x, trans.y, trans.z),
            .range = 10000.0
        };

        lightList.push_back(light);
    }

    const rf::IBL &ibl = scene.get_ibl();

    assert(ibl.specular.view().textureObject);
    cudarf::PBRParams pbrCommon{camera_translation, camera.exposure,
        lightList, ibl.get_sh_matrix(), ibl.brdfLUT.view().textureObject,
        ibl.specular.view().textureObject,
        ibl.specular.view().mipLevels};

    assert(pbrCommon.specular);

    if (shaderType == cudarf::SHADER_TYPE_PBR && work.pbr) {
        int translucentTotal;
        if (translucentTime != nullptr) {translucentTotal = translucentTime->start_interval("run_pipe", cuStream);}

        cudarf::pipe::run_pipe(rasterization_desc,
                             cudarf::RenderParams {
                                 true,
                                 true
                             },
                             work.pbr.uniforms,
#ifdef WITH_TAA
                             workHist.pbr.uniforms,
#endif
                             work.pbr.drawPacketIds,
                             work.pbr.matIds,
                             objMaterials,
                             cudarf::pipe::LaunchConfig(true,
                                                        false,
                                                        frameCounter,
                                                        output,
                                                        translucentTime,
                                                        testBinTiler),
                             cuStream
            );

        if (translucentTime != nullptr) {translucentTime->stop_interval(translucentTotal);}
    }

    if (shaderType == cudarf::SHADER_TYPE_UNLIT && work.flat) {
        int translucentTotal;

        if (translucentTime != nullptr) {translucentTotal = translucentTimeFlat->start_interval("run_pipe", cuStream);}

        cudarf::pipe::run_pipe(rasterization_desc,
                                     cudarf::RenderParams {
                                         true,
                                         true
                                     },
                                     work.flat.uniforms,
#ifdef WITH_TAA
                                     workHist.flat.uniforms,
#endif
                                     work.flat.drawPacketIds,
                                     work.flat.matIds,
                                     objMaterials,
                                     cudarf::pipe::LaunchConfig(true,
                                                                false,
                                                                frameCounter,
                                                                output,
                                                                translucentTimeFlat,
                                                                testBinTiler),
                                     cuStream
            );

        if (translucentTime != nullptr) {translucentTimeFlat->stop_interval(translucentTotal);}
    }

    return DrawListRenderer::Stats {
        work.pbr.drawPacketIds.size(),
        work.flat.drawPacketIds.size(),
    };
}
