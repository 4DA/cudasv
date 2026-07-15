#ifndef RF_IBL_HPP
#define RF_IBL_HPP
#include <memory>

#include <cuda_runtime.h>

#include <rf/renderer/image.hpp>
#include "cudarf/material.hpp"
#include "rf/renderer/cudarf/texture.hpp"

namespace rf
{

/** @brief The incoming diffuse irradiance can be expressed using spherical
    harmonics. The irradiance function is transformed into the frequency domain and
    stored as coefficients based on Legendre polynomial basis functions.

    Sources:
    - http://silviojemma.com/public/papers/lighting/spherical-harmonic-lighting.pdf
    - https://web.archive.org/web/20030521145638/http://www.research.scea.com/gdc2003/spherical-harmonic-lighting.pdf
    - https://github.com/sebh/HLSL-Spherical-Harmonics/blob/master/SphericalHarmonics.hlsl
    - https://github.com/kayru/Probulator
    - https://www.ppsloan.org/publications/StupidSH36.pdf
    - https://cseweb.ucsd.edu/~ravir/papers/envmap/envmap.pdf
    - https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2011/06/10-14.pdf
    - http://www.patapom.com/blog/SHPortal/
    - https://www.ppsloan.org/publications/SHJCGT.pdf
    - https://grahamhazel.com/blog/2017/12/22/converting-sh-radiance-to-irradiance/
    - http://limbicsoft.com/volker/prosem_paper.pdf
    - http://www.ppsloan.org/publications/shdering.pdf
    - https://bartwronski.files.wordpress.com/2014/08/bwronski_volumetric_fog_siggraph2014.pdf
*/

struct SphericalHarmonics
{
public:
    constexpr static std::size_t ROWS_COUNT = 4;

    SphericalHarmonics() = default;

    SphericalHarmonics(const std::vector<glm::vec3> &coefs):
        _coefficients(coefs) {}

    operator bool() const { return _coefficients.size() > 0; }
    glm::mat4 get_matrix() const {
        return glm::mat4(glm::vec4(_coefficients[0], 0.0f),
                         glm::vec4(_coefficients[1], 0.0f),
                         glm::vec4(_coefficients[2], 0.0f),
                         glm::vec4(_coefficients[3], 0.0f));
    }

    static SphericalHarmonics load_from_file(const std::string &file);

private:
    std::vector<glm::vec3> _coefficients;
};

/** @brief Images required for Image-based Lighting.

    Image-based lighting is a method of illuminating objects by considering the
    surrounding environment as a singular light source.

    The implementation of image-based lighting involves three components:
    1. Irradiance map: a convoluted environment map that provides diffuse irradiance
       for a specific direction;
    2. Prefiltered environment map: a convoluted cubemap with various mip-levels
       representing different roughness levels, leading to softer specular reflections;
    3. BRDF integration map: a texture that contains scaling and bias parameters for surface
       Fresnel response.

    These maps are typically created from HDR environment images (in equirectangular or other
    formats).

    for further reading:

    https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
    https://learnopengl.com/PBR/IBL/Diffuse-irradiance
    https://github.com/ux3d/glTF-Sample-Environments
    https://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
*/

struct IBL
{
    IBL() = default;

    // irradiance - coefficients for spherical harmonics
    // brdfLUT - a texture that stores scale and bias for the surface Fresnel response
    // specular - a convoluted cubemap with multiple mip levels,
    // representing varying roughness levels, leading to softer
    // specular reflections
    IBL(rf::SphericalHarmonics irradiance,
        cudarf::TextureResource &&brdfLUT,
        const cudarf::CubeMap &specular) :
        irradiance(irradiance),
        brdfLUT(std::move(brdfLUT)),
        specular(specular),
        mipLevels(9)
    {}

    IBL(IBL &&other) = default;
    IBL & operator=(IBL &&other) = default;

    IBL(IBL &other) = delete;
    IBL & operator=(IBL &other) = delete;

    operator bool() const { return brdfLUT.view().textureObject; }

    glm::mat4 get_sh_matrix() const {return irradiance.get_matrix();}

    rf::SphericalHarmonics irradiance;
    cudarf::TextureResource brdfLUT;
    cudarf::CubeMap specular;
    std::size_t mipLevels;
};

rf::IBL load_ibl(const std::string &path_prefix, cudaStream_t cuStream);

} // namespace rf

#endif
