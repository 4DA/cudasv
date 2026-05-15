#include <optional>
#include <sstream>
#include <string>
#include <utility>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/format.h>
#include <spdlog/fmt/bundled/printf.h>

#include <rf/renderer/glm_common.hpp>
#include <rf/renderer/material.hpp>
#include <rf/renderer/image.hpp>
#include <rf/renderer/utils.hpp>

#include "material.hpp"
#include "scene.hpp"

#include "renderer/gltf/material_loader.hpp"

namespace
{

const std::string UV_TRANSFORM_OFFSET = "offset";
const std::string UV_TRANSFORM_SCALE = "scale";
const std::string UV_TRANSFORM_ROTATION = "rotation";
const std::string UV_TRANSFORM_TEXCOORD = "texCoord";

const unsigned int MATERIAL_MIP_COUNT = 6;

std::optional<glm::vec2> get_texture_transform_vec2(const tinygltf::Value::Object &object,
                                                    const std::string &key)
{
    auto it = object.find(key);
    if (it == object.end()) {
        return std::nullopt;
    }

    const tinygltf::Value &value = it->second;
    if (!value.IsArray() || value.ArrayLen() != 2 ||
        !value.Get(0).IsNumber() || !value.Get(1).IsNumber()) {
        SPDLOG_ERROR("KHR_texture_transform::{} must be a 2-element numeric array", key);
        return std::nullopt;
    }

    return glm::vec2(static_cast<float>(value.Get(0).GetNumberAsDouble()),
                     static_cast<float>(value.Get(1).GetNumberAsDouble()));
}

std::optional<float> get_texture_transform_float(const tinygltf::Value::Object &object,
                                                 const std::string &key)
{
    auto it = object.find(key);
    if (it == object.end()) {
        return std::nullopt;
    }

    if (!it->second.IsNumber()) {
        SPDLOG_ERROR("KHR_texture_transform::{} must be numeric", key);
        return std::nullopt;
    }

    return static_cast<float>(it->second.GetNumberAsDouble());
}

std::optional<cudarf::Vec3f> to_vec3(const std::vector<double> &std_vec)
{
    if (std_vec.size() != 3) {
        return std::nullopt;
    }

    cudarf::Vec3f out_vec;
    out_vec.x = static_cast<float>(std_vec[0]);
    out_vec.y = static_cast<float>(std_vec[1]);
    out_vec.z = static_cast<float>(std_vec[2]);

    return out_vec;
}

std::optional<cudarf::Vec4f> to_vec4(const std::vector<double> &std_vec)
{
    if (std_vec.size() != 4) {
        return std::nullopt;
    }

    cudarf::Vec4f out_vec;
    out_vec.x = static_cast<float>(std_vec[0]);
    out_vec.y = static_cast<float>(std_vec[1]);
    out_vec.z = static_cast<float>(std_vec[2]);
    out_vec.w = static_cast<float>(std_vec[3]);

    return out_vec;
}

bool get_uv_transform_matrix(const tinygltf::Value::Object &texform, glm::mat3 &outTransform)
{
    glm::mat3 offset_mat(1.0f);
    glm::mat3 scale_mat(1.0f);
    glm::mat3 rot_mat(1.0f);

    glm::vec2 offset(0.0f, 0.0f);
    glm::vec2 scale(1.0f, 1.0f);
    float rotation = 0.0f;

    if (auto parsedOffset = get_texture_transform_vec2(texform, UV_TRANSFORM_OFFSET)) {
        offset = *parsedOffset;
        offset_mat = glm::translate(glm::mat3(1.0f), offset);
    } else if (texform.count(UV_TRANSFORM_OFFSET)) {
        return false;
    }

    if (auto parsedScale = get_texture_transform_vec2(texform, UV_TRANSFORM_SCALE)) {
        scale = *parsedScale;
        scale_mat = glm::scale(glm::mat3(1.0f), scale);
    } else if (texform.count(UV_TRANSFORM_SCALE)) {
        return false;
    }

    if (auto parsedRotation = get_texture_transform_float(texform, UV_TRANSFORM_ROTATION)) {
        rotation = *parsedRotation;
        rot_mat = glm::rotate(glm::mat3(1.0f), rotation);
    } else if (texform.count(UV_TRANSFORM_ROTATION)) {
        return false;
    }

    if (texform.count(UV_TRANSFORM_TEXCOORD)) {
        SPDLOG_ERROR("KHR_texture_transform::texCoord is not supported");
        return false;
    }

    SPDLOG_DEBUG("KHR_texture_transform: oft: {}, scale: {}, rot: {} -> mat: {}",
                 glm::to_string(offset),
                 glm::to_string(scale),
                 rotation,
                 glm::to_string(offset_mat * rot_mat * scale_mat));

    outTransform = offset_mat * rot_mat * scale_mat;
    return true;
}

bool get_uv_transform(const tinygltf::ExtensionMap &extensions, std::optional<glm::mat3> &outTransform)
{
    auto it = extensions.find("KHR_texture_transform");
    if (it != extensions.end()) {
        glm::mat3 transform(1.0f);
        if (!get_uv_transform_matrix(it->second.Get<tinygltf::Value::Object>(), transform)) {
            return false;
        }

        outTransform = transform;
        return true;
    }

    outTransform = std::nullopt;
    return true;
}

std::string to_string(const cudarf::Material &mat)
{
    std::stringstream ss;
    ss.precision(2);

    ss << "|col:<"
       << mat.baseColor.x << ","
       << mat.baseColor.y << ","
       << mat.baseColor.z << "," << mat.baseColor.w << ">"
       << ",met:" << mat.metallic
       << ",rgh:" << mat.roughness
       << ",em:" << mat.emissive.x
       << "," << mat.emissive.y
       << "," << mat.emissive.z << ">"
       << " | cc: " << mat.clearcoatFactor
       << " | ccr: " << mat.clearcoatRoughness;

    return ss.str();
}

std::optional<cudarf::Texture> load_gltf_image(const tinygltf::Model &model,
                                               int image_id,
                                               const std::string &image_name,
                                               const std::string &usageStr,
                                               std::optional<glm::mat3> uvTransform,
                                               unsigned int mipCount,
                                               cudaStream_t cuStream)
{
    if (image_id < 0 || image_id >= static_cast<int>(model.images.size())) {
        SPDLOG_ERROR("Invalid image id {}", image_id);
        return std::nullopt;
    }

    const tinygltf::Image &gltf_image = model.images[image_id];

    if (!gltf_image.image.size()) {
        SPDLOG_ERROR("image [index = {}, name = {}] has no data", image_id, gltf_image.name.c_str());
        return std::nullopt;
    }

    if (gltf_image.bits != 8) {
        SPDLOG_ERROR("image [name = {}]: bit depth not supported: {}", gltf_image.name.c_str(), gltf_image.bits);
        return std::nullopt;
    }

    rf::Image image;
    image.w = gltf_image.width;
    image.h = gltf_image.height;
    image.channels = gltf_image.component;
    image.name = gltf_image.name;
    image.data = &gltf_image.image[0];

    switch (gltf_image.pixel_type) {
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        image.pixel_type = rf::PixelChannelType::U8;
        break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        image.pixel_type = rf::PixelChannelType::U16;
        break;
    case TINYGLTF_COMPONENT_TYPE_FLOAT:
        image.pixel_type = rf::PixelChannelType::FLOAT32;
        break;
    default:
        SPDLOG_ERROR("image [name = {}]: unsupported pixel type {}", gltf_image.name.c_str(), gltf_image.pixel_type);
        return std::nullopt;
    }

    SPDLOG_INFO("created gltf image[name={}]: {} x {} @ {}-{}, usage: {}",
                image_name.c_str(), image.w, image.h, image.channels,
                static_cast<int>(image.pixel_type), usageStr);

    if (image.pixel_type != rf::PixelChannelType::U8) {
        SPDLOG_ERROR("Only U8 texture compo type is supported");
        return std::nullopt;
    }

    // The mipmapped CUDA texture path currently expects RGBA8. TinyGLTF's
    // default stb loader expands decoded images to 4 channels as long as
    // SetPreserveImageChannels(true) is not enabled.
    auto texOpt = cudarf::create_cuda_texture(image, cudaAddressModeWrap, mipCount, cuStream);

    if (!texOpt) {
        return texOpt;
    }

    if (uvTransform) {
        texOpt->hasUVTransform = true;
        texOpt->uvTransform = *uvTransform;
    }

    return texOpt;
}

std::optional<int> get_parameter_texture_index(const tinygltf::Parameter &parameter,
                                               const char *label)
{
    const int texIndex = parameter.TextureIndex();
    if (texIndex < 0) {
        SPDLOG_ERROR("{} is present but does not contain a valid texture index", label);
        return std::nullopt;
    }

    return texIndex;
}

std::optional<cudarf::Texture> load_pbr_texture(const tinygltf::Model &model,
                                                int tex_index,
                                                const tinygltf::ExtensionMap &extensions,
                                                const char *label,
                                                unsigned int mipCount,
                                                cudaStream_t cuStream)
{
    if (tex_index < 0 || tex_index >= static_cast<int>(model.textures.size())) {
        SPDLOG_ERROR("Bad texture id: {}", tex_index);
        return std::nullopt;
    }

    int image_id = model.textures[tex_index].source;
    if (image_id < 0 || image_id >= static_cast<int>(model.images.size())) {
        SPDLOG_ERROR("Texture {} references invalid image id {}", tex_index, image_id);
        return std::nullopt;
    }

    std::string name = model.images[image_id].name;
    std::optional<glm::mat3> uvTransform;
    if (!get_uv_transform(extensions, uvTransform)) {
        return std::nullopt;
    }

    auto tex = load_gltf_image(model, image_id, name, label, uvTransform, mipCount, cuStream);
    if (!tex) {
        return std::nullopt;
    }

    return tex;
}

std::string get_material_name(const tinygltf::Material *gltfMaterial,
                              int materialIndex,
                              const std::string &prefix)
{
    if (gltfMaterial == nullptr) {
        return prefix + "__gltf_default_material";
    }

    if (!gltfMaterial->name.empty()) {
        return prefix + gltfMaterial->name;
    }

    return prefix + "__gltf_material_" + std::to_string(materialIndex);
}

} // namespace

namespace loader::gltf
{

std::shared_ptr<cudarf::Material>
create_material(const tinygltf::Model &model,
                rf::Scene &scene,
                const tinygltf::Primitive &primitive,
                const std::string &prefix,
                cudaStream_t cuStream)
{
    const int materialIndex = primitive.material;
    if (materialIndex >= static_cast<int>(model.materials.size())) {
        SPDLOG_ERROR("Primitive references invalid material index {}", materialIndex);
        return nullptr;
    }

    const tinygltf::Material *gltfMaterial =
        materialIndex >= 0 ? &model.materials[materialIndex] : nullptr;
    std::string materialName = get_material_name(gltfMaterial, materialIndex, prefix);

    auto existingMat = scene.get_material(materialName);
    if (existingMat != nullptr) {
        return existingMat;
    }

    bool isOpaque = true;
    bool isDoubleSided = false;
    bool isEmissive = false;
    bool withClearcoat = false;
    bool withTransmission = false;
    bool isUnlit = false;

    unsigned int albedoTexChannels = 0;
    unsigned int normalTexChannels = 0;
    unsigned int OMRTexChannels = 0;
    unsigned int emissiveTexChannels = 0;

    cudarf::Vec4f baseColorFactor = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    float metallicFactor = 1.0f;
    float roughnessFactor = 1.0f;
    cudarf::Vec3f emissiveFactor = make_float3(0.0f, 0.0f, 0.0f);
    float clearcoatFactor = 0.0f;
    float clearcoatRoughnessFactor = 0.0f;

    rf::UVTransform uvt;

    cudarf::Texture albedoTexture;
    cudarf::Texture normalTexture;
    cudarf::Texture emissiveTexture;
    cudarf::Texture metRoughTexture;

    if (gltfMaterial != nullptr) {
        if (!gltfMaterial->values.size()) {
            SPDLOG_ERROR("material {}: No \"pbrMetallicRoughness\" parameters found",
                         materialName.c_str());
        }

        auto baseRes = to_vec4(gltfMaterial->pbrMetallicRoughness.baseColorFactor);
        if (baseRes) {
            baseColorFactor = *baseRes;
        } else {
            SPDLOG_ERROR("Base factor vec contains invalid values");
            return nullptr;
        }

        metallicFactor = static_cast<float>(gltfMaterial->pbrMetallicRoughness.metallicFactor);
        roughnessFactor = static_cast<float>(gltfMaterial->pbrMetallicRoughness.roughnessFactor);

        auto emfRes = to_vec3(gltfMaterial->emissiveFactor);
        if (emfRes) {
            emissiveFactor = *emfRes;
        } else {
            SPDLOG_ERROR("Emissive factor contains invalid values");
            return nullptr;
        }

        isOpaque = !(gltfMaterial->alphaMode == "BLEND");
        isDoubleSided = gltfMaterial->doubleSided;
        isUnlit = (gltfMaterial->extensions.find("KHR_materials_unlit") != gltfMaterial->extensions.end());
        withTransmission =
            (gltfMaterial->extensions.find("KHR_materials_transmission") != gltfMaterial->extensions.end());

        const tinygltf::PbrMetallicRoughness &pbr = gltfMaterial->pbrMetallicRoughness;

        if (pbr.baseColorTexture.index != -1) {
            auto loaded = load_pbr_texture(model,
                                           pbr.baseColorTexture.index,
                                           pbr.baseColorTexture.extensions,
                                           "albedo",
                                           MATERIAL_MIP_COUNT,
                                           cuStream);
            if (!loaded) {
                return nullptr;
            }

            albedoTexture = *loaded;
            albedoTexChannels = loaded->channels;
            if (loaded->hasUVTransform) {
                uvt.transforms[rf::UVTransform::TEXTURE_KEY_BASECOLOR] = loaded->uvTransform;
            }
        }

        auto extClearcoat = gltfMaterial->extensions.find("KHR_materials_clearcoat");
        if (extClearcoat != gltfMaterial->extensions.end()) {
            withClearcoat = true;

            if (extClearcoat->second.Has("clearcoatFactor")) {
                auto &factor = extClearcoat->second.Get("clearcoatFactor");
                clearcoatFactor = float(factor.IsNumber() ? factor.Get<double>() : factor.Get<int>());
            }

            if (extClearcoat->second.Has("clearcoatRoughnessFactor")) {
                const auto &factor = extClearcoat->second.Get("clearcoatRoughnessFactor");
                clearcoatRoughnessFactor =
                    float(factor.IsNumber() ? factor.Get<double>() : factor.Get<int>());
            }
        }

        if (pbr.metallicRoughnessTexture.index != -1) {
            auto loaded = load_pbr_texture(model,
                                           pbr.metallicRoughnessTexture.index,
                                           pbr.metallicRoughnessTexture.extensions,
                                           "met-rough",
                                           MATERIAL_MIP_COUNT,
                                           cuStream);
            if (!loaded) {
                return nullptr;
            }

            metRoughTexture = *loaded;
            OMRTexChannels = loaded->channels;
            if (loaded->hasUVTransform) {
                uvt.transforms[rf::UVTransform::TEXTURE_KEY_METROUGH] = loaded->uvTransform;
            }
        }

        if (gltfMaterial->normalTexture.index != -1) {
            auto loaded = load_pbr_texture(model,
                                           gltfMaterial->normalTexture.index,
                                           gltfMaterial->normalTexture.extensions,
                                           "normal-tex",
                                           MATERIAL_MIP_COUNT,
                                           cuStream);
            if (!loaded) {
                return nullptr;
            }

            normalTexture = *loaded;
            normalTexChannels = loaded->channels;
            if (loaded->hasUVTransform) {
                uvt.transforms[rf::UVTransform::TEXTURE_KEY_NORMAL] = loaded->uvTransform;
            }
        }

        auto emissiveIt = gltfMaterial->additionalValues.find("emissiveTexture");
        if (emissiveIt != gltfMaterial->additionalValues.end()) {
            auto texId = get_parameter_texture_index(emissiveIt->second, "emissiveTexture");
            if (!texId) {
                return nullptr;
            }

            auto loaded = load_pbr_texture(model,
                                           *texId,
                                           gltfMaterial->emissiveTexture.extensions,
                                           "emissive",
                                           MATERIAL_MIP_COUNT,
                                           cuStream);
            if (!loaded) {
                return nullptr;
            }

            emissiveTexture = *loaded;
            emissiveTexChannels = loaded->channels;
            isEmissive = true;
            if (loaded->hasUVTransform) {
                uvt.transforms[rf::UVTransform::TEXTURE_KEY_EMISSIVE] = loaded->uvTransform;
            }
        }
    }

    rf::MaterialInfo info(isOpaque,
                          isDoubleSided,
                          isUnlit,
                          isEmissive,
                          withClearcoat,
                          withTransmission,
                          uvt,
                          albedoTexChannels,
                          normalTexChannels,
                          OMRTexChannels,
                          emissiveTexChannels);

    auto newMat                = std::make_shared<cudarf::Material>();
    newMat->type               = isUnlit ? cudarf::SHADER_TYPE_UNLIT : cudarf::SHADER_TYPE_PBR;
    newMat->baseColor          = baseColorFactor;
    newMat->emissive           = emissiveFactor;
    newMat->metallic           = metallicFactor;
    newMat->roughness          = roughnessFactor;
    newMat->isTranslucent      = !isOpaque;
    newMat->isDoubleSided      = isDoubleSided;
    newMat->albedoTex          = albedoTexture;
    newMat->emissiveTex        = emissiveTexture;
    newMat->metRoughTex        = metRoughTexture;
    newMat->normalTex          = normalTexture;
    newMat->clearcoatFactor    = clearcoatFactor;
    newMat->clearcoatRoughness = clearcoatRoughnessFactor;

    scene.add_material(materialName, newMat);

    SPDLOG_INFO("created material '{}' ptr:{}, t:{} | {} {}",
                materialName,
                static_cast<void *>(newMat.get()),
                static_cast<int>(newMat->type),
                info.to_string(),
                to_string(*newMat));

    return newMat;
}

} // namespace loader::gltf
