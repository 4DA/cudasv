 /* Code in this file is loosely based on KhronosGroup GLTF-SampleViewer
 *  ( https://github.com/KhronosGroup/glTF-Sample-Viewer )
 *
 * That uses Apache License 2.0
 */

/*******************************************************************************
 * Copyright [UX3D](https://ux3d.io/)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.

 *******************************************************************************/

#ifndef CUDARF_FRAGMENT_INL
#define CUDARF_FRAGMENT_INL

struct MaterialData
{
    float perceptualRoughness;      // roughness value, as authored by the model creator (input to shader)
    float3 f0;                        // full reflectance color (n incidence angle)

    float alphaRoughness;           // roughness mapped to a more linear change in the roughness (proposed by [2])
    float3 albedoColor;

    float3 f90;                       // reflectance color at grazing angle
    float metallic;

    float3 n;
    float3 baseColor; // getBaseColor()
    float alpha;
};


// compute irradiance from spherical harmonics
__device__
cudarf::ColorRGB compute_spherical(const glm::mat4 &coefs,
                                   const cudarf::Vec3f &N,
                                   const cudarf::ColorRGB &albedo)
{
    return albedo * to_vec3f(glm::max(glm::vec3(coefs * glm::vec4(1.0f, N.y, N.z, N.x)), 0.0f));
}

__device__ MaterialData get_metallic_roughness_data(const cudarf::Material &mat, float f0_ior)
{
    MaterialData data;
    data.metallic            = mat.metallic;
    data.perceptualRoughness = mat.roughness;
    data.baseColor.x         = mat.baseColor.x;
    data.baseColor.y         = mat.baseColor.y;
    data.baseColor.z         = mat.baseColor.z;
    data.alpha               = mat.baseColor.w;

    // r: occlusion, g: roughness, b: metallic
    // if (material.hasMetallicRoughnessMap) {
    //     vec4 sample = texture(...)
    //     data.perceptualRoughness *= sample.g;
    //     data.metallic *= sample.b;
    // }

    // Achromatic f0 based on IOR.
    float3 f0 = make_float3(f0_ior, f0_ior, f0_ior);

    data.albedoColor = my::lerp(data.baseColor * (make_float3(1.0) - f0),  make_float3(0.0), data.metallic);
    data.f0 = my::lerp(f0, data.baseColor, data.metallic);

    return data;
}

// https://github.com/KhronosGroup/glTF/blob/master/extensions/2.0/Khronos/KHR_lights_punctual/README.md#range-property
__device__ float get_range_attenuation(float range, float distance)
{
    if (range <= 0.0f) {
        // negative range means unlimited
        return 1.0f;
    }
    return max(min(1.0f - pow(distance / range, 4.0f), 1.0f), 0.0f) / pow(distance, 2.0f);
}

// fresnel reflectance term
__device__ float3 F_Schlick(float3 f0, float3 f90, float VdotH)
{
    return f0 + (f90 - f0) * pow(clamp(1.0f - VdotH, 0.0f, 1.0f), 5.0f);
}

__device__ float3 BRDF_lambertian(float3 f0, float3 f90, float3 diffuseColor, float VdotH)
{
    // float M_PI = 3.141592653589793;
    float pi = 3.141592653589793;

    // see https://seblagarde.wordpress.com/2012/01/08/pi-or-not-to-pi-in-game-lighting-equation/
    return (1.0f - F_Schlick(f0, f90, VdotH)) * (diffuseColor / pi);
}


// Smith Joint GGX
// Note: Vis = G / (4 * NdotL * NdotV)
// see Eric Heitz. 2014. Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs. Journal of Computer Graphics Techniques, 3
// see Real-Time Rendering. Page 331 to 336.
// see https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf/geometricshadowing(specularg)
__device__ float V_GGX(float NdotL, float NdotV, float alphaRoughness)
{
    float alphaRoughness2 = alphaRoughness * alphaRoughness;

    float GGXV = NdotL * sqrt(NdotV * NdotV * (1.0 - alphaRoughness2) + alphaRoughness2);
    float GGXL = NdotV * sqrt(NdotL * NdotL * (1.0 - alphaRoughness2) + alphaRoughness2);

    float GGX = GGXV + GGXL;

    if (GGX > 0.0) {
        return 0.5 / GGX;
    }

    return 0.0;
}

// The following equation(s) model the distribution of microfacet normals across the area being drawn (aka D())
// Implementation from "Average Irregularity Representation of a Roughened Surface for Ray Reflection" by T. S. Trowbridge, and K. P. Reitz
// Follows the distribution function recommended in the SIGGRAPH 2013 course notes from EPIC Games [1], Equation 3.
__device__ float D_GGX(float NdotH, float alphaRoughness)
{
    float alphaRoughness2 = alphaRoughness * alphaRoughness;
    float f = (NdotH * NdotH) * (alphaRoughness2 - 1.0) + 1.0;
    return alphaRoughness2 / (M_PI * f * f);
}


//  https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#acknowledgments AppendixB
__device__ float3 BRDF_specularGGX(float3 f0, float3 f90, float alphaRoughness, float VdotH, float NdotL, float NdotV, float NdotH)
{
    float3 F = F_Schlick(f0, f90, VdotH);
    float Vis = V_GGX(NdotL, NdotV, alphaRoughness);
    float D = D_GGX(NdotH, alphaRoughness);

    return F * Vis * D;
}

#ifdef CUDARF_ENABLE_PUNCTUAL_LIGHTS
__device__ __inline__
void compute_point_lights(const cudarf::rast::PipeParams *pipe,
                          const cudarf::rast::Fragment &frag,
                          const MaterialData &materialData,
                          cudarf::ColorRGB &comp_diffuse,
                          cudarf::ColorRGB &comp_specular)
{
    float3 v = normalize(pipe->camera - frag.pos_global);

    for (int i = 0; i < pipe->lightCount; i++) {
        const cudarf::CUDARFLight &light = pipe->lights[i];

        float3 fragToLight = light.position - frag.pos_global;

        float3 l = normalize(fragToLight);   // direction from surface point to light
        float3 h = normalize(l + v);          // direction of the vector between l and v, called halfway vector
        float NdotL = clamp(dot(materialData.n, l), 0.0f, 1.0f);
        float NdotV = clamp(dot(materialData.n, v), 0.0f, 1.0f);
        float VdotH = clamp(dot(v, h), 0.0f, 1.0f);
        float NdotH = clamp(dot(materialData.n, h), 0.0f, 1.0f);

        float attenuation = get_range_attenuation(light.range, length(fragToLight));
        float3 intensity = make_float3(attenuation) * light.intensity;

        if (NdotL > 0.0 || NdotV > 0.0) {
            // for reference:
            // https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#implementation

            comp_diffuse += intensity * NdotL *
                BRDF_lambertian(materialData.f0, materialData.f90, materialData.albedoColor, VdotH);

            comp_specular += intensity * NdotL *
                BRDF_specularGGX(materialData.f0, materialData.f90, materialData.alphaRoughness, VdotH, NdotL, NdotV, NdotH);
        }
    }
}
#endif


__device__ __inline__
cudarf::ColorRGB get_ibl_radiance_ggx(const cudarf::rast::PipeParams *pipe,
                                      glm::vec3 n,
                                      glm::vec3 v,
                                      float NdotV,
                                      float perceptualRoughness,
                                      glm::vec3 specularColor)
{
    int mipCount = pipe->specular.mipCount;
    assert(mipCount);

    float lod = clamp(perceptualRoughness * float(mipCount), 0.0, float(mipCount));
    glm::vec3 reflection = glm::normalize(glm::reflect(-v, n));

    glm::vec2 brdfSamplePoint = glm::clamp(glm::vec2(NdotV, perceptualRoughness), glm::vec2(0.0, 0.0), glm::vec2(1.0, 1.0));
    glm::vec2 brdf = glm::vec2(to_glm(tex2D<float4>(pipe->brdfLUT, brdfSamplePoint.x, brdfSamplePoint.y)));

    glm::vec4 specularSample = to_glm(sampleCube(pipe->specular, reflection.x, reflection.y, reflection.z, lod));
    glm::vec3 specularLight = glm::vec3(specularSample);

// #ifndef USE_HDR
//     specularLight = cudarf::shading::srgb_to_linear(specularLight);
// #endif

    return to_vec3f(specularLight * (specularColor * brdf.x + brdf.y));
    // return make_color(brdfSamplePoint.x, brdfSamplePoint.y, 0.0f);
}

template<bool TTexturingEnabled, bool TClearcoatEnabled>
__device__ float4 compute_color_pbr(const cudarf::rast::PipeParams *pipe, const cudarf::rast::Fragment &frag)
{
    // The default index of refraction of 1.5 yields a dielectric normal incidence reflectance of 0.04.
    // const float ior = 1.5;
    const float f0_ior = 0.04;
    const auto ClearcoatF0 = make_float3(0.04f, 0.04f, 0.04f);
    const auto ClearcoatF90 = make_float3(1.0f, 1.0f, 1.0f);

    cudarf::Material mat = pipe->materials[frag.materialId];

    if (TTexturingEnabled && mat.albedoTex.textureObject) {
        glm::vec3 uv(frag.tex.x, frag.tex.y, 0.0f);
        uv = uv * mat.albedoTex.uvTransform;

        float4 texVal = tex2DLod<float4>(mat.albedoTex.textureObject, uv.x, uv.y, frag.albedoLod);
        float3 rgb = to_vec3f(texVal);
        rgb = cudarf::shading::srgb_to_linear(rgb);
        mat.baseColor.x *= rgb.x;
        mat.baseColor.y *= rgb.y;
        mat.baseColor.z *= rgb.z;
        mat.baseColor.w *= texVal.w;
    }

    MaterialData materialData = get_metallic_roughness_data(mat, f0_ior);

    const float3 &n = frag.normal;

    // Perceptual roughness is converted to material roughness by squaring the
    // perceptual value, as is standard practice.
    materialData.alphaRoughness = materialData.perceptualRoughness * materialData.perceptualRoughness;

    // Compute reflectance.
    float reflectance = max(max(materialData.f0.x, materialData.f0.y), materialData.f0.z);

    // Anything less than 2% is physically impossible and is instead considered
    // to be shadowing. Compare to "Real-Time-Rendering" 4th editon on page 325.
    materialData.f90 = make_float3(clamp(reflectance * 50.0f, 0.0f, 1.0f));
    materialData.n = n;

    cudarf::ColorRGB comp_diffuse =
        compute_spherical(pipe->sphericalHarmonics,
                          to_vec3f(cudarf::shading::vehicle_frame_to_cubemap_frame(to_glm(n))),
                          materialData.albedoColor);

    float3 emissive = mat.emissive;

    if (TTexturingEnabled && mat.emissiveTex.textureObject) {
        glm::vec3 uv(frag.tex.x, frag.tex.y, 0.0f);
        uv = uv * mat.emissiveTex.uvTransform;
        emissive *= cudarf::shading::srgb_to_linear(
            to_vec3f(tex2D<float4>(mat.emissiveTex.textureObject, uv.x, uv.y)));
    }

    float3 v = normalize(pipe->camera - frag.pos_global);
    float NdotV = clamp(dot(n, v), 0.0f, 1.0f);

    cudarf::ColorRGB comp_specular = get_ibl_radiance_ggx(pipe,
                                                          cudarf::shading::vehicle_frame_to_cubemap_frame(to_glm(n)),
                                                          cudarf::shading::vehicle_frame_to_cubemap_frame(to_glm(v)),
                                                          NdotV,
                                                          materialData.perceptualRoughness,
                                                          to_glm(materialData.f0));

#ifdef CUDARF_ENABLE_PUNCTUAL_LIGHTS
    compute_point_lights(pipe, frag, materialData, comp_diffuse, comp_specular);
#endif

    cudarf::Vec3f outColor;

    // TODO: using clearcoat increases register usage, consider executing in separate kernel
    if (TClearcoatEnabled && mat.clearcoatFactor > 0.0f) {
        cudarf::ColorRGB comp_clearcoat = get_ibl_radiance_ggx(pipe,
                                                               cudarf::shading::vehicle_frame_to_cubemap_frame(to_glm(n)),
                                                               cudarf::shading::vehicle_frame_to_cubemap_frame(to_glm(v)),
                                                               NdotV,
                                                               mat.clearcoatRoughness,
                                                               to_glm(ClearcoatF0));

        float clearcoatFactor = mat.clearcoatFactor;
        cudarf::Vec3f clearcoatFresnel = F_Schlick(ClearcoatF0, ClearcoatF90, NdotV);

        outColor = (comp_specular + comp_diffuse + emissive) *
            (1.0 - clearcoatFactor * clearcoatFresnel) + comp_clearcoat * clearcoatFactor;

    } else {
        outColor = comp_specular + comp_diffuse + emissive;
    }

    return make_float4(outColor, materialData.alpha);
}

template<bool TTexturingEnabled>
__device__
cudarf::Color compute_color_flat(const cudarf::rast::PipeParams *pipe, const cudarf::rast::Fragment &frag)
{
    const cudarf::Material &material = pipe->materials[frag.materialId];
    cudarf::Color texCol = make_color(1.0f, 1.0f, 1.0f, 1.0f);

    if (TTexturingEnabled && material.albedoTex.textureObject) {
        glm::vec3 uv(frag.tex.x, frag.tex.y, 0.0f);

        // uncomment if KHR_TEXTURE_TRANSFORM is used on flat material
        // uv = uv * material.albedoTex.uvTransform;

        texCol = tex2DLod<float4>(material.albedoTex.textureObject, uv.x, uv.y, frag.albedoLod);
    }

    return frag.vertexColor * material.baseColor * texCol;
}

__device__ __inline__
glm::vec4 make_clip_pos_for_deriv(const float2 &screenPos,
                                  float invW,
                                  const glm::vec2 &windowSize)
{
    float w = 1.0f / invW;
    glm::vec2 ndc = 2.0f * glm::vec2(screenPos.x, screenPos.y) / windowSize - glm::vec2(1.0f, 1.0f);
    return glm::vec4(ndc * w, 0.0f, w);
}

template<bool TTexturingEnabled>
__device__ __inline__
cudarf::Vec3f compute_shading_bary(const cudarf::rast::PipeParams *pipe,
                                   const cudarf::rast::Triangle &tri,
                                   int x,
                                   int y,
                                   float &albedoLod)
{
    cudarf::Vec2f fragWindow = make_vec2f(x + 0.5f, y + 0.5f);
    cudarf::Vec3f baryAffine = compute_bary_affine2(tri, fragWindow);
    albedoLod = 0.0f;

#ifdef CUDARF_FORCE_AFFINE_BARYCENTRICS
    return baryAffine;
#else
    if constexpr (!TTexturingEnabled) {
        return compute_bary_persp(baryAffine, tri.w_rcp);
    } else {
        glm::vec2 windowSize((float)pipe->windowWidth, (float)pipe->windowHeight);

        glm::vec4 P0 = make_clip_pos_for_deriv(tri.sP0, tri.w_rcp.x, windowSize);
        glm::vec4 P1 = make_clip_pos_for_deriv(tri.sP1, tri.w_rcp.y, windowSize);
        glm::vec4 P2 = make_clip_pos_for_deriv(tri.sP2, tri.w_rcp.z, windowSize);
        glm::vec2 ndc = 2.0f * glm::vec2(x + 0.5f, y + 0.5f) / windowSize - glm::vec2(1.0f, 1.0f);

        cudarf::Barycentric bary = compute_bary_persp_deriv(P0, P1, P2, ndc, windowSize);

        const cudarf::Material &material = pipe->materials[tri.materialId];
        if (material.albedoTex.textureObject && material.albedoTex.mipLevels > 1) {
            float2 duv_dx = interp(bary.ddx, tri.tex);
            float2 duv_dy = interp(bary.ddy, tri.tex);
            float2 texDim = make_float2(material.albedoTex.width,
                                        material.albedoTex.height);

            float2 dTdx = duv_dx * texDim;
            float2 dTdy = duv_dy * texDim;
            float rho = max(length(dTdx), length(dTdy));
            float lod = log2f(max(rho, 1e-8f));
            float maxLod = float(material.albedoTex.mipLevels - 1);
            albedoLod = clamp(lod, 0.0f, maxLod);
        }

        return make_vec3f(bary.lambda.x, bary.lambda.y, bary.lambda.z);
    }
#endif
}

template<cudarf::ShaderType TShaderType, bool TTexturingEnabled>
__device__ __inline__
void compute_fragment(const cudarf::Material &material,
                      const cudarf::rast::Triangle &tri,
                      const cudarf::Vec3f &baryPersp,
                      cudarf::rast::Fragment &frag)
{
    if constexpr (TShaderType == cudarf::SHADER_TYPE_UNLIT) {
        cudarf::Color color[3] = {tri.col[0], tri.col[1], tri.col[2]};
        frag.vertexColor = interpolate(baryPersp, color);
    } else if constexpr(TShaderType == cudarf::SHADER_TYPE_PBR) {
        frag.pos_global = interp(baryPersp, tri.v_world);
        frag.normal = normalize(interp(baryPersp, tri.normal));

        if (material.normalTex.textureObject) {
            cudarf::Vec3f T = normalize(interp(baryPersp, tri.tangent));
            frag.tangent = normalize(T - frag.normal * dot(frag.normal, T));   // Gram-Schmidt

            float sgns[3] = {
                (tri.bitan_sgn & 1 ? 1.0f : -1.0f),
                (tri.bitan_sgn & 2 ? 1.0f : -1.0f),
                (tri.bitan_sgn & 4 ? 1.0f : -1.0f)
            };

            float sign = (interp(baryPersp, sgns) >= 0.0f) ? 1.0f : -1.0f;

            frag.bitangent = normalize(cross(frag.normal, frag.tangent)) * sign;
        }

        if (tri.isBackFacing) {
            frag.normal = -frag.normal;

            if (material.normalTex.textureObject) {
                frag.tangent = -frag.tangent;
                frag.bitangent = -frag.bitangent;
            }
        }
    }

    if constexpr(TTexturingEnabled) {
        frag.tex = interp(baryPersp, tri.tex);
    }

#ifdef WITH_TAA
    frag.pos_ss_hist = interp(baryPersp, tri.v_ss_hist);
#endif

    frag.materialId = tri.materialId;
}

template<cudarf::ShaderType TShaderType, bool TTexturingEnabled, bool TClearcoatEnabled>
__device__ __inline__
cudarf::Color shade_fragment(const cudarf::rast::PipeParams *pipe,
                             const cudarf::rast::Triangle &tri,
                             int x,
                             int y,
                             cudarf::rast::Fragment &frag)
{
    float albedoLod;
    cudarf::Vec3f baryPersp =
        compute_shading_bary<TTexturingEnabled>(pipe, tri, x, y, albedoLod);
    const cudarf::Material &material = pipe->materials[tri.materialId];

    compute_fragment<TShaderType, TTexturingEnabled>(material, tri, baryPersp, frag);

    frag.albedoLod = albedoLod;

    if constexpr (TShaderType == cudarf::SHADER_TYPE_UNLIT) {
        return compute_color_flat<TTexturingEnabled>(pipe, frag);
    } else {
        return compute_color_pbr<TTexturingEnabled, TClearcoatEnabled>(pipe, frag);
    }
}

#endif
