#ifndef RF_MATERIAL_HPP
#define RF_MATERIAL_HPP

#include <unordered_map>
#include <cstddef>
#include <type_traits>
#include <map>

#include "texture.hpp"
#include <rf/renderer/glm_common.hpp>

namespace rf
{

struct UVTransform;

// Stores data needed to apply affine transform to UV coordinate
//
struct UVTransform {
    static const std::string TEXTURE_KEY_BASECOLOR;
    static const std::string TEXTURE_KEY_NORMAL;
    static const std::string TEXTURE_KEY_EMISSIVE;
    static const std::string TEXTURE_KEY_METROUGH;

    std::unordered_map<std::string, glm::mat3> transforms;

    // Returns a unique key
    std::size_t get_key() const;

    // Checks if the transform is active
    operator bool() const {return transforms.size();}

    // Returns all activated UV stream names
    std::vector<std::string> get_keys() const;
};

struct MaterialInfo
{
    // Indicates if the object is fully opaque; false means it is translucent.
    bool isOpaque;

    // Indicates if the object should be rendered without back-face culling.
    bool isDoubleSided;

    // Indicates if the material does not interact with light.
    bool isUnlit;

    // Indicates if the material acts as a light source.
    bool isEmissive;

    // Indicates if the material is covered with a dielectric translucent layer.
    bool withClearcoat;

    // Indicates if the material has transparent properties as per the KHR_materials_transmission extension.
    bool withTransmission;

    // Greater than zero if KHR_texture_transform is applied.
    std::size_t uvTextureTransformKey;

    // List of textures that have UV transforms.
    std::vector<std::string> uvTextureKeys;

    // Number of channels in the albedo texture.
    unsigned int albedoTexChannels;

    // Number of channels in the normal texture.
    unsigned int normalTexChannels;

    // Indicates the type of textures; 0 - none, 1 - oc
    unsigned int OMRTexChannels;

    // number of channels in emissive texture
    // "0" means attribute is // not present
    unsigned int emissiveTexChannels;

    // key that uniquely defines a material
    std::size_t materialKey;

    MaterialInfo(bool isOpaque, bool isDoubleSided,
                 bool isUnlit, bool isEmissive,
                 bool withClearcoat,
                 bool withTransmission,
                 const UVTransform &uvt,
                 unsigned int albedoTexChannels,
                 unsigned int normalTexChannels,
                 unsigned int OMRTexChannels,
                 unsigned int emissiveTexChannels):
        isOpaque(isOpaque),
        isDoubleSided(isDoubleSided),
        isUnlit(isUnlit),
        isEmissive(isEmissive),
        withClearcoat(withClearcoat),
        withTransmission(withTransmission),
        uvTextureTransformKey(uvt.get_key()),
        uvTextureKeys(uvt.get_keys()),
        albedoTexChannels(albedoTexChannels),
        normalTexChannels(normalTexChannels),
        OMRTexChannels(OMRTexChannels),
        emissiveTexChannels(emissiveTexChannels),
        materialKey(compute_key()) {}

    bool operator<=>(const MaterialInfo &other) const = default;

    size_t compute_key() const;

    std::string to_string() const;
};

} // namespace rf

#endif
