#include "renderer/gltf/scene_loader.hpp"

#include <memory>
#include <string>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/printf.h>

#include <rf/renderer/glm_common.hpp>
#include <rf/renderer/mesh_geometry.hpp>
#include <rf/renderer/trs_transform.hpp>

#include "cudarf.hpp"
#include "primitive_component.hpp"
#include "renderer/gltf/accessors.hpp"
#include "renderer/gltf/material_loader.hpp"
#include "scene.hpp"
#include "trs_transform.hpp"

using rf::AttributesAccessor;
using rf::MeshInfo;
using rf::PointLightComponent;
using rf::PrimitiveComponent;
using rf::Scene;
using rf::SceneComponent;
using rf::TRSTransform;

namespace
{

const std::string ATTRIB_POSITION = "POSITION";
const std::string ATTRIB_NORMAL = "NORMAL";
const std::string ATTRIB_COLOR_0 = "COLOR_0";
const std::string ATTRIB_TEXCOORD_0 = "TEXCOORD_0";
const std::string ATTRIB_TEXCOORD_1 = "TEXCOORD_1";
const std::string ATTRIB_TANGENT = "TANGENT";

float to_float(const tinygltf::Value &value)
{
    if (value.IsInt()) {
        return static_cast<float>(value.Get<int>());
    }
    if (value.IsNumber()) {
        return static_cast<float>(value.Get<double>());
    }

    SPDLOG_ERROR("Not a number in tinygltf::Value");
    return 0.0f;
}

bool make_gltf_mesh(const tinygltf::Model &model,
                    const tinygltf::Primitive &gltfPrim,
                    rf::GltfMesh &gltfMesh)
{
    for (auto it: gltfPrim.attributes) {
        if (it.second < 0) {
            SPDLOG_ERROR("Primitive attribute {} has invalid accessor index {}", it.first, it.second);
            return false;
        }

        AttributesAccessor attrib;
        const tinygltf::Accessor &gltf_accessor = model.accessors[it.second];

        if (!loader::gltf::init_attributes_accessor(model, attrib, it.first, gltf_accessor)) {
            SPDLOG_ERROR("Failed to initialize accessor for attribute '{}'", it.first);
            return false;
        }

        if (it.first == ATTRIB_POSITION) {
            gltfMesh.vertices = std::move(attrib);
        } else if (it.first == ATTRIB_NORMAL) {
            gltfMesh.normals = std::move(attrib);
        } else if (it.first == ATTRIB_TEXCOORD_0) {
            gltfMesh.texcoords = std::move(attrib);
        } else if (it.first == ATTRIB_TANGENT) {
            gltfMesh.tangents = std::move(attrib);
        } else {
            SPDLOG_INFO("{}", fmt::sprintf("Skipped vertex attrib %s", it.first.c_str()));
        }
    }

    if (gltfPrim.indices < 0 || gltfPrim.indices >= static_cast<int>(model.accessors.size())) {
        SPDLOG_ERROR("Primitive has invalid index accessor {}", gltfPrim.indices);
        return false;
    }

    const tinygltf::Accessor &indexAccessor = model.accessors[gltfPrim.indices];
    if (indexAccessor.bufferView < 0 ||
        indexAccessor.bufferView >= static_cast<int>(model.bufferViews.size())) {
        SPDLOG_ERROR("Index accessor {} references invalid bufferView {}",
                     gltfPrim.indices,
                     indexAccessor.bufferView);
        return false;
    }

    const tinygltf::BufferView &bufferView = model.bufferViews[indexAccessor.bufferView];
    if (bufferView.buffer < 0 || bufferView.buffer >= static_cast<int>(model.buffers.size())) {
        SPDLOG_ERROR("Index bufferView {} references invalid buffer {}",
                     indexAccessor.bufferView,
                     bufferView.buffer);
        return false;
    }
    const tinygltf::Buffer &buffer = model.buffers[bufferView.buffer];

    if (gltfPrim.mode != TINYGLTF_MODE_TRIANGLES) {
        SPDLOG_ERROR("Unsupported primitive mode {}. Only triangles are supported", gltfPrim.mode);
        return false;
    }

    if (indexAccessor.componentType != TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT &&
        indexAccessor.componentType != TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
        SPDLOG_ERROR("Unsupported index component type {}", indexAccessor.componentType);
        return false;
    }

    if (bufferView.byteStride != 0) {
        SPDLOG_ERROR("Unsupported index buffer stride {}", bufferView.byteStride);
        return false;
    }

    gltfMesh.convertIdxTo32 = (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT);
    gltfMesh.indexPtr = (uint32_t *)(&buffer.data.at(0) + bufferView.byteOffset + indexAccessor.byteOffset);
    gltfMesh.count = indexAccessor.count;
    return true;
}

bool create_mesh_component(cudarf::pipe::Ctx *desc,
                           const tinygltf::Model &model,
                           Scene &scene,
                           const tinygltf::Node &node,
                           const std::string &compoName,
                           const glm::vec3 &translation,
                           const glm::quat &rotation,
                           const glm::vec3 &scale,
                           SceneComponent *parent,
                           const std::string &namePrefix,
                           loader::PrimitiveComponentCB cb,
                           cudaStream_t cuStream,
                           SceneComponent **outComponent)
{
    const tinygltf::Mesh &mesh = model.meshes[node.mesh];

    if (scene.get_scene_component(compoName) != nullptr) {
        SPDLOG_ERROR("Duplicate scene component name '{}'", compoName);
        return false;
    }

    auto newCompo = std::unique_ptr<PrimitiveComponent>(
        new PrimitiveComponent(compoName, TRSTransform(translation, rotation, scale), parent, false, false));

    for (const tinygltf::Primitive &gltfPrim : mesh.primitives) {
        MeshInfo info;
        glm::vec3 vertexMin;
        glm::vec3 vertexMax;

        for (const auto &attribPair : gltfPrim.attributes) {
            size_t accessorId = attribPair.second;

            if (accessorId >= model.accessors.size()) {
                SPDLOG_ERROR("{}", fmt::sprintf("Invalid accessorId: %lu, total accessors count: %lu",
                                                accessorId, model.accessors.size()));
                return false;
            }

            const tinygltf::Accessor &accessor = model.accessors[accessorId];

            if (attribPair.first == ATTRIB_POSITION) {
                auto resMin = loader::gltf::to_glm_vec3(accessor.minValues);
                auto resMax = loader::gltf::to_glm_vec3(accessor.maxValues);

                if (resMin && resMax) {
                    vertexMin = *resMin;
                    vertexMax = *resMax;
                } else {
                    SPDLOG_ERROR("{}", fmt::sprintf("Accessor contains invalid min/max values"));
                    return false;
                }
            } else if (attribPair.first == ATTRIB_NORMAL) {
                info.normal_component_count = 3;
                if (accessor.type != TINYGLTF_TYPE_VEC3) {
                    SPDLOG_ERROR("NORMAL accessor has unexpected type {}", accessor.type);
                    return false;
                }
            } else if (attribPair.first == ATTRIB_COLOR_0) {
                info.vertex_color_component_count = 4;
                if (accessor.type != TINYGLTF_TYPE_VEC4) {
                    SPDLOG_ERROR("COLOR_0 accessor has unexpected type {}", accessor.type);
                    return false;
                }
            } else if (attribPair.first == ATTRIB_TEXCOORD_0) {
                info.texcoord1_component_count = 2;
                if (accessor.type != TINYGLTF_TYPE_VEC2) {
                    SPDLOG_ERROR("TEXCOORD_0 accessor has unexpected type {}", accessor.type);
                    return false;
                }
            } else if (attribPair.first == ATTRIB_TEXCOORD_1) {
                info.texcoord2_component_count = 2;
                if (accessor.type != TINYGLTF_TYPE_VEC2) {
                    SPDLOG_ERROR("TEXCOORD_1 accessor has unexpected type {}", accessor.type);
                    return false;
                }
            } else if (attribPair.first == ATTRIB_TANGENT) {
                info.tangent_component_count = 4;
                if (accessor.type != TINYGLTF_TYPE_VEC4) {
                    SPDLOG_ERROR("TANGENT accessor has unexpected type {}", accessor.type);
                    return false;
                }
            } else {
                SPDLOG_INFO("{}", fmt::sprintf("Skipping attribute %s", attribPair.first.c_str()));
                continue;
            }

            if (info.tangent_component_count && !info.normal_component_count) {
                SPDLOG_ERROR("primitive [gltf mesh = {}] has tangents, but not normals, disabling tangents",
                             mesh.name);
                info.tangent_component_count = 0;
            }
        }

        std::shared_ptr<cudarf::Material> material =
            loader::gltf::create_material(model, scene, gltfPrim, namePrefix, cuStream);
        if (material == nullptr) {
            SPDLOG_ERROR("Failed to create material for primitive in mesh '{}'", mesh.name);
            return false;
        }

        rf::GltfMesh gltfMesh;
        if (!make_gltf_mesh(model, gltfPrim, gltfMesh)) {
            return false;
        }

        int drawPacketId = cudarf::pipe::alloc_draw_packet(desc);

        newCompo->add_primitive(info, vertexMin, vertexMax, drawPacketId, material);

        cudarf::pipe::set_draw_packet_buffers(desc, gltfMesh, drawPacketId, cuStream);
    }

    SPDLOG_DEBUG("{}", fmt::sprintf("added prim_component [name = %s, toLocal='%s'",
                                    newCompo->name.c_str(), newCompo->toLocal.to_string().c_str()));

    if (cb && cb(*newCompo, scene)) {
        *outComponent = nullptr;
        return true;
    }

    *outComponent = scene.add_primitive_component(std::move(newCompo), parent);
    return true;
}

bool create_light_component(const tinygltf::Model &model,
                            Scene &scene,
                            const tinygltf::Node &node,
                            const std::string &compoName,
                            const glm::vec3 &translation,
                            const glm::quat &rotation,
                            const glm::vec3 &scale,
                            SceneComponent *parent,
                            SceneComponent **outComponent)
{
    int id = node.extensions.at("KHR_lights_punctual").Get("light").Get<int>();

    auto extension_it = model.extensions.find("KHR_lights_punctual");
    if (extension_it == model.extensions.end() || !extension_it->second.Has("lights")) {
        SPDLOG_INFO("{}", fmt::sprintf("No lights are found in model"));
        return false;
    }

    auto lights = extension_it->second.Get("lights");
    if (!lights.IsArray() || id < 0 || id >= static_cast<int>(lights.ArrayLen())) {
        SPDLOG_ERROR("KHR_lights_punctual references invalid light index {}", id);
        return false;
    }

    auto gltf_light = lights.Get(id);
    float intensity = to_float(gltf_light.Get("intensity"));

    PointLightComponent *newCompo =
        scene.add_light_component(compoName, TRSTransform(translation, rotation, scale), intensity, parent);

    SPDLOG_DEBUG("{}", fmt::sprintf("created light component [name = %s, int = %f, xform = %s]",
                                    newCompo->name.c_str(), newCompo->intensity, newCompo->to_world.to_string().c_str()));
    *outComponent = newCompo;
    return true;
}

bool create_scene_component(cudarf::pipe::Ctx *desc,
                            const tinygltf::Model &model,
                            int nodeIndex,
                            Scene &scene,
                            const tinygltf::Node &node,
                            SceneComponent *parent,
                            const std::string &namePrefix,
                            loader::PrimitiveComponentCB cb,
                            cudaStream_t cuStream,
                            SceneComponent **outComponent)
{
    if (desc == nullptr) {
        SPDLOG_ERROR("Cannot create GLTF scene component: raster context is null");
        return false;
    }

    if (parent == nullptr) {
        SPDLOG_ERROR("Cannot create GLTF scene component for node {}: parent is null", nodeIndex);
        return false;
    }

    SPDLOG_DEBUG("parent component [name = {}] [toLocal = {}]",
                 parent->name, parent->toLocal.to_string());

    std::string compoName = node.name.empty()
        ? namePrefix + "__gltf_node_" + std::to_string(nodeIndex)
        : namePrefix + node.name;

    glm::vec3 scale(1.0f);
    glm::vec3 translation(0.0f);
    glm::quat rotation(1.0f, 0.0f, 0.0f, 0.0f);

    if (node.matrix.size() == 16) {
        glm::vec3 skew;
        glm::vec4 persp;
        glm::mat4 srcMat = glm::make_mat4(node.matrix.data());
        glm::decompose(srcMat, scale, rotation, translation, skew, persp);
        rotation = glm::normalize(rotation);
    } else {
        if (node.scale.size() == 3) {
            scale = glm::vec3(node.scale[0], node.scale[1], node.scale[2]);
        }

        if (node.rotation.size() == 4) {
            rotation.x = static_cast<float>(node.rotation[0]);
            rotation.y = static_cast<float>(node.rotation[1]);
            rotation.z = static_cast<float>(node.rotation[2]);
            rotation.w = static_cast<float>(node.rotation[3]);
        }

        if (node.translation.size() == 3) {
            translation = glm::vec3(node.translation[0], node.translation[1], node.translation[2]);
        }
    }

    if (node.mesh >= 0) {
        return create_mesh_component(desc,
                                     model,
                                     scene,
                                     node,
                                     compoName,
                                     translation,
                                     rotation,
                                     scale,
                                     parent,
                                     namePrefix,
                                     cb,
                                     cuStream,
                                     outComponent);
    }

    if (node.extensions.count("KHR_lights_punctual")) {
        return create_light_component(model,
                                      scene,
                                      node,
                                      compoName,
                                      translation,
                                      rotation,
                                      scale,
                                      parent,
                                      outComponent);
    }

    if (scene.get_scene_component(compoName) != nullptr) {
        SPDLOG_ERROR("Duplicate scene component name '{}'", compoName);
        return false;
    }

    SceneComponent *newCompo =
        scene.add_scene_component(compoName, TRSTransform(translation, rotation, scale), parent);

    SPDLOG_DEBUG("{}", fmt::sprintf("added scene_component [name = %s] [toLocal %s]",
                                    newCompo->name), newCompo->toLocal.to_string());

    *outComponent = newCompo;
    return true;
}

bool load_scene_tree(cudarf::pipe::Ctx *desc,
                     const tinygltf::Model &model,
                     int nodeIndex,
                     const tinygltf::Node &node,
                     rf::Scene &scene,
                     rf::SceneComponent *parent,
                     const std::string &namePrefix,
                     loader::PrimitiveComponentCB cb,
                     cudaStream_t cuStream)
{
    if (desc == nullptr) {
        SPDLOG_ERROR("Cannot load GLTF scene tree at node {}: raster context is null", nodeIndex);
        return false;
    }

    if (parent == nullptr) {
        SPDLOG_ERROR("Cannot load GLTF scene tree at node {}: parent is null", nodeIndex);
        return false;
    }

    SceneComponent *compo = nullptr;
    if (!create_scene_component(desc, model, nodeIndex, scene, node, parent, namePrefix, cb, cuStream, &compo)) {
        return false;
    }

    SceneComponent *childParent = compo ? compo : parent;

    for (int child: node.children) {
        if (child < 0 || child >= static_cast<int>(model.nodes.size())) {
            SPDLOG_ERROR("GLTF node {} references invalid child index {}", nodeIndex, child);
            return false;
        }

        if (!load_scene_tree(desc, model, child, model.nodes[child], scene, childParent, namePrefix, cb, cuStream)) {
            return false;
        }
    }

    if (parent) {
        parent->compute_bounding_box();
    }

    return true;
}

} // namespace

namespace loader::gltf
{

bool load_scene(cudarf::pipe::Ctx *desc,
                const tinygltf::Model &model,
                rf::Scene &scene,
                rf::SceneComponent *parent,
                const std::string &namePrefix,
                loader::PrimitiveComponentCB cb,
                cudaStream_t cuStream)
{
    if (model.scenes.empty()) {
        SPDLOG_ERROR("GLTF model has no scenes");
        return false;
    }

    int scene_id = model.defaultScene >= 0 ? model.defaultScene : 0;
    if (scene_id < 0 || scene_id >= static_cast<int>(model.scenes.size())) {
        SPDLOG_ERROR("GLTF default scene index {} is out of range (scene count = {})",
                     scene_id, model.scenes.size());
        return false;
    }

    const tinygltf::Scene &gltf_scene = model.scenes[scene_id];

    for (const auto &nodeIndex : gltf_scene.nodes) {
        if (nodeIndex < 0 || nodeIndex >= static_cast<int>(model.nodes.size())) {
            SPDLOG_ERROR("GLTF scene references invalid node index {} (node count = {})",
                         nodeIndex, model.nodes.size());
            return false;
        }

        if (!load_scene_tree(desc, model, nodeIndex, model.nodes[nodeIndex], scene, parent, namePrefix, cb, cuStream)) {
            return false;
        }
    }

    return true;
}

} // namespace loader::gltf
