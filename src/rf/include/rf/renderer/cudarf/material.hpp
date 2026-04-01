#ifndef CUDARF_MATERIAL_HPP
#define CUDARF_MATERIAL_HPP

#include <map>
#include <unordered_map>

#include <rf/renderer/glm_common.hpp>

namespace cudarf
{
enum ShaderType {
    SHADER_TYPE_PBR = 0,
    SHADER_TYPE_UNLIT = 1,
    SHADER_TYPE_COUNT = 2,
};

struct Texture {
    cudaTextureObject_t textureObject;
    bool hasUVTransform;
    glm::mat3 uvTransform;
    unsigned int channels;

    __device__ __host__ Texture(cudaTextureObject_t textureObject,
                                bool hasUVTransform,
                                glm::mat3 uvTransform,
                                unsigned int channels):
        textureObject(textureObject),
        hasUVTransform(hasUVTransform),
        uvTransform(uvTransform),
        channels(channels)
        {}

    __device__ __host__ Texture():
        textureObject(0),
        hasUVTransform(false),
        uvTransform(1.0f),
        channels(0)
        {}
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
    Texture albedoTex;
    Texture emissiveTex;
    Texture occlusionTex;
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
