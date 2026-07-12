#include <algorithm>
#include <sstream>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/printf.h>


#include "math.hpp"
#include "material.hpp"


using namespace rf;

const std::string UVTransform::TEXTURE_KEY_BASECOLOR = "BaseColor";
const std::string UVTransform::TEXTURE_KEY_NORMAL    = "Normal";
const std::string UVTransform::TEXTURE_KEY_EMISSIVE  = "Emissive";
const std::string UVTransform::TEXTURE_KEY_METROUGH  = "MetRough";

std::size_t UVTransform::get_key() const
{
    std::size_t seed = 0;

    std::size_t has_basecolor = (transforms.find(TEXTURE_KEY_BASECOLOR) != transforms.end());
    std::size_t has_normal = (transforms.find(TEXTURE_KEY_NORMAL) != transforms.end());
    std::size_t has_emissive = (transforms.find(TEXTURE_KEY_EMISSIVE) != transforms.end());
    std::size_t has_metrough = (transforms.find(TEXTURE_KEY_METROUGH) != transforms.end());

    elegant_pair(seed, has_basecolor);
    elegant_pair(seed, has_normal);
    elegant_pair(seed, has_emissive);
    elegant_pair(seed, has_metrough);
    return seed;
}

std::vector<std::string> UVTransform::get_keys() const {
    std::vector<std::string> result;

    for (const auto &it: transforms) {
        result.push_back(it.first);
    }

    std::sort(result.begin(), result.end());

    return result;
}

size_t MaterialInfo::compute_key() const
{
    std::size_t seed = 0;

    elegant_pair(seed, isOpaque);
    elegant_pair(seed, isDoubleSided);
    elegant_pair(seed, isUnlit);
    elegant_pair(seed, withClearcoat);
    elegant_pair(seed, withTransmission);
    elegant_pair(seed, uvTextureTransformKey);
    elegant_pair(seed, albedoTexChannels);
    elegant_pair(seed, normalTexChannels);
    elegant_pair(seed, OMRTexChannels);
    elegant_pair(seed, emissiveTexChannels);

    return seed;
}

std::string MaterialInfo::to_string() const {
    std::stringstream ss;
    ss.precision(2);

    ss << "opq:" << isOpaque
       << ",ds:" << isDoubleSided
       << ",em:" << isEmissive
       << ",unlt:" << isUnlit
       << ",cc:" << withClearcoat
       << ",tr:" << withTransmission
       << ",uvt:" << static_cast<int>(uvTextureTransformKey != 0)
       << ",alb_ch:" << albedoTexChannels
       << ",nrm_ch:" << normalTexChannels
       << ",omr_ch:" << OMRTexChannels
       << ",em_ch:" << emissiveTexChannels;

    return ss.str();
}

