#include <vector>

#include <spdlog/spdlog.h>

#include <engine/engine.hpp>
#include <rf/renderer/gltf_loader.hpp>
#include <rf/renderer/utils.hpp>

#include "underlay.hpp"
#include "rf/renderer/cudarf/texture.hpp"

using namespace engine;

const std::string underlayCompoName = "Underlay";

rf::NaiveMeshPtr Underlay::generate_underlay_mesh(const Config *config)
{
    const vehicle::VehicleDimensions *vehicle_config =
        &config->vehicle_config;
    const CalibrationConfig *calibration =
        &config->calibration_config;

    rf::NaiveMeshPtr mesh = std::make_shared<rf::NaiveMesh>();

    float length = vehicle_config->length;
    float width = vehicle_config->width;
    float z_offset = 50.0f;
    float tesselation_step = 100.0f;
    int Ny = 1;
    int Nx = 1;

    glm::vec4 underlayRect = glm::vec4(-(length / 2.0f + underlay_config->blind_rear),
                               -(width / 2.0f + underlay_config->blind_right),
                               length + underlay_config->blind_front +
                                       underlay_config->blind_rear,
                               width + underlay_config->blind_left +
                                       underlay_config->blind_right);

    Nx = underlayRect.z / tesselation_step;
    Ny = underlayRect.w / tesselation_step;

    float step_x = underlayRect.z / Nx;
    float step_y = underlayRect.w / Ny;

    float step_u = -1.0 / Ny;
    float step_v = 1.0 / Nx;

    float p0_x = underlayRect.x;
    float p0_y = underlayRect.y;
    float t0_u;
    float t0_v;

    for (int i = 0; i < Nx; i++) {

        p0_x = underlayRect.x + i * step_x;
        t0_v = i * step_v;

        for (int j = 0; j < Ny; j++) {

            p0_y = underlayRect.y + j * step_y;

            mesh->vertices.push_back(cudarf::Vec3f(p0_x,          p0_y, z_offset));
            mesh->vertices.push_back(cudarf::Vec3f(p0_x + step_x, p0_y, z_offset));
            mesh->vertices.push_back(cudarf::Vec3f(p0_x + step_x, p0_y + step_y, z_offset));
            mesh->vertices.push_back(cudarf::Vec3f(p0_x + step_x, p0_y + step_y, z_offset));
            mesh->vertices.push_back(cudarf::Vec3f(p0_x,          p0_y + step_y, z_offset));
            mesh->vertices.push_back(cudarf::Vec3f(p0_x,          p0_y, z_offset));

            t0_u = 1.0f + j * step_u;

            mesh->texcoords.push_back(cudarf::Vec2f(t0_u,          t0_v));
            mesh->texcoords.push_back(cudarf::Vec2f(t0_u,          t0_v + step_v));
            mesh->texcoords.push_back(cudarf::Vec2f(t0_u + step_u, t0_v + step_v));
            mesh->texcoords.push_back(cudarf::Vec2f(t0_u + step_u, t0_v + step_v));
            mesh->texcoords.push_back(cudarf::Vec2f(t0_u + step_u, t0_v));
            mesh->texcoords.push_back(cudarf::Vec2f(t0_u,          t0_v));
        }
    }

    return mesh;
}

struct MaterialWithUnderlayTexture {
    cudarf::TextureResource texture;
    cudarf::Material material;
};

int Underlay::init(cudarf::pipe::Ctx *desc, rf::Scene &scene, const Config *config, cudaStream_t cuStream)
{
    this->underlay_config = &config->overlays_config.underlay_config;

    underlay_image_description = rf::load_image(underlay_config->texture_path, false, true);
    auto loaded = cudarf::create_cuda_texture(underlay_image_description, cudaAddressModeClamp, 1, std::nullopt, cuStream);

    if (!loaded) {
        SPDLOG_ERROR("can not load underlay image: {}",
                     underlay_config->texture_path);
        return -1;
    }
    assert(loaded);

    auto materialWithTex = std::make_shared<MaterialWithUnderlayTexture>();
    auto &underlayTexture = materialWithTex->texture;

    underlayTexture = std::move(*loaded);

    rf::NaiveMeshPtr mesh = generate_underlay_mesh(config);

    // use shared_ptr aliasing constructor for superblock to hold aggregate
    // object, but provide the narrow cudarf::Material pointer
    material = std::shared_ptr<cudarf::Material>(
        materialWithTex, &materialWithTex->material);

    material->isTranslucent = true;
    material->baseColor = make_float4(1.0, 1.0, 1.0, 1.0);
    material->emissive = make_float3(0.0, 0.0, 0.0);
    material->metallic = 0.0f;
    material->roughness = 1.0f;
    material->type = cudarf::SHADER_TYPE_UNLIT;
    material->albedoTex = underlayTexture.view();
    material->isDoubleSided = false;

    loader::add_naive_mesh(desc,
                           underlayCompoName,
                           mesh,
                           material,
                           rf::TRSTransform(),
                           scene,
                           *scene.get_root(),
                           cuStream);

    scene.add_material("underlay::shadowMat",  material);

    return 0;
}
