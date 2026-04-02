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

namespace
{

template <typename T>
bool validate_optional_vertex_attribute(const std::vector<T> &attribute,
                                        const std::size_t vertexCount,
                                        const char *label,
                                        const std::string &meshName)
{
    if (attribute.empty()) {
        return true;
    }

    if (attribute.size() != vertexCount) {
        SPDLOG_ERROR("Cannot add naive mesh '{}': {} count {} does not match vertex count {}",
                     meshName,
                     label,
                     attribute.size(),
                     vertexCount);
        return false;
    }

    return true;
}

} // namespace

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
    if (desc == nullptr) {
        SPDLOG_ERROR("Cannot add naive mesh '{}': raster context is null", name);
        return nullptr;
    }

    if (meshPtr == nullptr) {
        SPDLOG_ERROR("Cannot add naive mesh '{}': mesh is null", name);
        return nullptr;
    }

    if (parent == nullptr) {
        SPDLOG_ERROR("Cannot add naive mesh '{}': parent component is null", name);
        return nullptr;
    }

    if (scene.get_scene_component(name) != nullptr) {
        SPDLOG_ERROR("Cannot add naive mesh '{}': scene component already exists", name);
        return nullptr;
    }

    if (meshPtr->vertices.empty()) {
        SPDLOG_ERROR("Cannot add naive mesh '{}': mesh has no vertices", name);
        return nullptr;
    }

    if (!validate_optional_vertex_attribute(meshPtr->colors, meshPtr->vertices.size(), "color", name) ||
        !validate_optional_vertex_attribute(meshPtr->normals, meshPtr->vertices.size(), "normal", name) ||
        !validate_optional_vertex_attribute(meshPtr->texcoords, meshPtr->vertices.size(), "texcoord", name)) {
        return nullptr;
    }

    auto newCompo = std::make_unique<rf::PrimitiveComponent>(name, transform, parent, false, false);

    int drawPacketId = cudarf::pipe::alloc_draw_packet(desc);
    unsigned int idxSize = meshPtr->idx.data() ? meshPtr->idx.size() : meshPtr->vertices.size();
    const cudarf::Vec4f *colorData = meshPtr->colors.empty() ? nullptr : meshPtr->colors.data();
    const cudarf::Vec3f *normalData = meshPtr->normals.empty() ? nullptr : meshPtr->normals.data();
    const cudarf::Vec2f *texcoordData = meshPtr->texcoords.empty() ? nullptr : meshPtr->texcoords.data();
    const std::uint8_t normalComponentCount = meshPtr->normals.empty() ? 0 : 3;
    const std::uint8_t texcoordComponentCount = meshPtr->texcoords.empty() ? 0 : 2;
    const std::uint8_t vertexColorComponentCount = meshPtr->colors.empty() ? 0 : 4;

    cudarf::pipe::set_draw_packet_buffers(desc,
                                          meshPtr->idx.data(),
                                          idxSize,
                                          meshPtr->vertices.data(),
                                          meshPtr->vertices.size(),
                                          colorData,
                                          normalData,
                                          texcoordData,
                                          drawPacketId,
                                          cuStream);

    newCompo->add_primitive(rf::MeshInfo(normalComponentCount,
                                         texcoordComponentCount,
                                         0,
                                         0,
                                         vertexColorComponentCount),
                            meshPtr->vertexMin,
                            meshPtr->vertexMax,
                            drawPacketId,
                            material);

    rf::PrimitiveComponent *addedComponent = scene.add_primitive_component(std::move(newCompo), parent);

    SPDLOG_INFO("Added compo[name={}]", name);

    return addedComponent;
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
    if (desc == nullptr) {
        SPDLOG_ERROR("Failed to load GLTF: raster context is null");
        return false;
    }

    if (parent == nullptr) {
        SPDLOG_ERROR("Failed to load GLTF '{}': parent component is null", file);
        return false;
    }

    if (file.empty()) {
        SPDLOG_ERROR("Failed to load GLTF: input file path is empty");
        return false;
    }

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
