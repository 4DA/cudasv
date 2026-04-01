#include <cassert>
#include <string>

#include <spdlog/spdlog.h>

#include "cudarf.hpp"
#include "gltf_loader.hpp"

#include "renderer/gltf/animation_loader.hpp"
#include "renderer/gltf/scene_loader.hpp"

#include "primitive_component.hpp"
#include "scene.hpp"

using namespace loader;

using rf::Scene;
using rf::SceneComponent;
using rf::PrimitiveComponent;

rf::SceneComponent *
loader::add_naive_mesh(cudarf::pipe::Ctx *desc,
                       const std::string &name,
                       rf::NaiveMeshPtr meshPtr,
                       const std::shared_ptr<cudarf::Material> &material,
                       const rf::TRSTransform &transform,
                       rf::Scene &scene,
                       rf::SceneComponent *parent,
                       cudaStream_t cuStream)
{
    assert(scene.get_scene_component(name) == nullptr);

    rf::PrimitiveComponent *newCompo = new rf::PrimitiveComponent(name, transform, parent, false, false);

    int drawPacketId = cudarf::pipe::alloc_draw_packet(desc);
    unsigned int idxSize = meshPtr->idx.data() ? meshPtr->idx.size() : meshPtr->vertices.size();

    cudarf::pipe::set_draw_packet_buffers(desc,
                                          meshPtr->idx.data(),
                                          idxSize,
                                          meshPtr->vertices.data(),
                                          meshPtr->vertices.size(),
                                          nullptr, // TODO: handle vertex colors
                                          meshPtr->normals.data(),
                                          meshPtr->texcoords.data(),
                                          drawPacketId,
                                          cuStream);

    newCompo->add_primitive(rf::MeshInfo(0, 0, 0, 0, 0),
                            meshPtr->vertexMin,
                            meshPtr->vertexMax,
                            drawPacketId,
                            material);

    scene.add_primitive_component(std::unique_ptr<rf::PrimitiveComponent>(newCompo), parent);

    SPDLOG_INFO("Added compo[name={}]", name);

    return newCompo;
}


static std::string get_file_path_extension(const std::string &name) {
    if (name.find_last_of(".") != std::string::npos) {
        return name.substr(name.find_last_of(".") + 1);
    }

    return "";
}


bool loader::load_gltf_model(cudarf::pipe::Ctx *desc,
                             const std::string &file,
                             rf::Scene &scene,
                             rf::SceneComponent *parent,
                             const std::string &namePrefix,
                             rf::AnimationMap &animations,
                             tinygltf::Model &model,
                             PrimitiveComponentCB cb,
                             cudaStream_t cuStream)
{
    assert(file.length() > 0);

    std::string errMsg;
    std::string warnMsg;

    tinygltf::TinyGLTF loader;
    std::string ext = get_file_path_extension(file);

    // what about single channel images?
    // loader.SetPreserveImageChannels(true);

    bool loadOK = false;

    if (ext.compare("glb") == 0) {
        loadOK = loader.LoadBinaryFromFile(&model, &errMsg, &warnMsg, file.c_str());
    } else {
        // assume ascii glTF.
        loadOK = loader.LoadASCIIFromFile(&model, &errMsg, &warnMsg, file.c_str());
    }

    if (!warnMsg.empty()) {
        SPDLOG_INFO("Warning: {}", warnMsg);
    }

    if (!errMsg.empty()) {
        SPDLOG_INFO("Error: {}", errMsg);
    }

    if (!loadOK) {
        SPDLOG_ERROR("Failed to load GLTF '{}'", file);
        return false;
    }

    SPDLOG_INFO("Loaded GLTF {} meshes", model.meshes.size());

    if (!loader::gltf::load_scene(desc, model, scene, parent, namePrefix, cb, cuStream)) {
        SPDLOG_ERROR("Failed to load GLTF scene graph '{}'", file);
        return false;
    }

    animations = loader::gltf::load_animations(model, namePrefix);

    return true;
}
