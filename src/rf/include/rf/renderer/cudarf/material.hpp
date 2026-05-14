#ifndef CUDARF_MATERIAL_HPP
#define CUDARF_MATERIAL_HPP

#include <map>
#include <unordered_map>

#include <rf/renderer/glm_common.hpp>
#include <rf/renderer/cudarf/texture.hpp>

namespace cudarf
{
enum ShaderType {
    SHADER_TYPE_PBR = 0,
    SHADER_TYPE_UNLIT = 1,
    SHADER_TYPE_COUNT = 2,
};

struct Material {
    ShaderType type;
    float4 baseColor;
    float3 emissive;
    float metallic;
    float roughness;
    float clearcoatFactor;
    float clearcoatRoughness;
    bool isTranslucent;
    bool isDoubleSided;
    Texture albedoTex;
    Texture normalTex;
    Texture emissiveTex;
    Texture metRoughTex;
};


struct CubeMap {
    cudaTextureObject_t tex = 0;
    operator bool() const {return tex != 0 && mipCount != 0;}
    long unsigned int mipCount = 0;
};

using MaterialMap = std::unordered_map<int32_t, std::shared_ptr<cudarf::Material>>;
using MaterialNames = std::unordered_map<std::string, std::shared_ptr<cudarf::Material>>;
using MaterialPtrMap = std::unordered_map<std::shared_ptr<cudarf::Material>, unsigned int>;

}

#endif
