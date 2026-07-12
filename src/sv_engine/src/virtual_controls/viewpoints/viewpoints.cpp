#include <vector>

#include <spdlog/spdlog.h>

#include <rf/renderer/primitive_component.hpp>
#include <rf/camera_control/rotator.hpp>
#include <rf/camera_control/viewpoint_animation.hpp>
#include <rf/renderer/gltf_loader.hpp>

#include "viewpoints.hpp"

// identity rotation of camera: forward = +Z, up = +Y
// generate rotation rotation to "to"
static glm::quat make_rotation(const glm::vec3 &pos, const glm::vec3 &to, const glm::vec3 &worldUp) {
    const glm::vec3 ORIGINAL_FORWARD(0.0f, 0.0f, 1.0f);
    const glm::vec3 ORIGINAL_UP(0.0f, 1.0f, 0.0f);

    return rf::get_look_at_trs(pos,
                               to,
                               worldUp,
                               ORIGINAL_FORWARD,
                               ORIGINAL_UP).rotation;
}

static float get_prefab_alpha_multiplier(const VirtualControlConfig *config,
                                         const std::string &componentName)
{
    for (uint32_t index = 0; index < config->alpha_multipliers_count; ++index) {
        const ViewpointControlAlphaMultiplier &alpha = config->alpha_multipliers[index];
        if (componentName == std::string("control::") + alpha.component) {
            return alpha.value;
        }
    }

    return 1.0f;
}

static void apply_prefab_alpha_multiplier(const rf::PrimitiveComponent &source,
                                          float alphaMultiplier)
{
    if (alphaMultiplier == 1.0f) {
        return;
    }

    std::set<std::shared_ptr<cudarf::Material>> adjustedMaterials;
    for (const auto &primitive : source.get_primitives()) {
        if (primitive->cudarfMaterial && adjustedMaterials.insert(primitive->cudarfMaterial).second) {
            primitive->cudarfMaterial->baseColor.w *= alphaMultiplier;
        }
    }
}

static bool viewpoint_control_inserter(const VirtualControlConfig *config,
                                       std::vector<rf::PrimitiveComponent> &components,
                                       rf::Scene &scene,
                                       const std::vector<rf::Viewpoint> &viewpoints,
                                       std::vector<rf::PrimitiveComponent *> &viewpointCompos,
                                       rf::SceneComponent *parent)
{
    for (uint32_t index = 0; index < config->controls_count; index++) {
        const ViewpointControlIconSettings *control =
            &config->controls[index];

        for (uint32_t comp_idx = 0; comp_idx < components.size(); comp_idx++)
        {
            rf::PrimitiveComponent &compo = components[comp_idx];

            SPDLOG_INFO("prim_component [name = {}, toLocal='{}'", compo.name.c_str(), compo.toLocal.to_string().c_str());

            std::string public_component_name = std::string(control->component);
            std::string component_name = "control::" + public_component_name;

            if (compo.name == component_name) {

                rf::PrimitiveComponent *newCompo = nullptr;
                uint32_t viewpoint = control->viewpoint;
                bool frontFace = false;
                glm::vec3 pos;
                std::string name = std::string("control::") + public_component_name +
                    std::to_string(index) + "_" +
                    std::to_string(viewpoint);

                if (control->is_billboard) {
                    frontFace = true;
                }

                pos.x = control->position[0];
                pos.y = control->position[1];
                pos.z = control->position[2];

                if (viewpoint < viewpoints.size())
                {
                    const rf::Viewpoint &v = viewpoints[viewpoint];

                    glm::quat rotation(1.0f, 0.0f, 0.0f, 0.0f);

                    if (control->is_viewpoint_oriented) {
                        if (v.is_elliptic()) {
                            glm::vec3 worldUp = rf::EllipticRotator::get_world_up(v.get_elliptic().coord);
                            rotation = make_rotation(pos, glm::vec3(0.0f), worldUp);
                        } else {
                            glm::vec3 worldUp;
                            // if camera is to be oriented strictly vertically than pick up
                            // +X as its new up
                            if (glm::length(glm::vec2(v.get_look_at().lookAt - v.get_look_at().origin)) <=
                                2.0f * std::numeric_limits<float>::epsilon())
                            {
                                worldUp = glm::vec3(1.0f, 0.0f, 0.0f);
                            } else {
                                worldUp = glm::vec3(0.0f, 0.0f, 1.0f);
                            }

                            rotation = make_rotation(pos, v.get_look_at().lookAt, worldUp);
                        }
                    } else {
                        glm::vec3 worldUp = glm::vec3(0.0f, 0.0f, 1.0f);

                        glm::vec3 lookAt = glm::vec3(control->look_at[0],
                                                     control->look_at[1],
                                                     control->look_at[2]);

                        rotation = make_rotation(pos, lookAt, worldUp);
                    }

                    newCompo = scene.add_primitive_component(name,
                                                              rf::TRSTransform(
                                                                  pos / 1000.0f,
                                                                  frontFace ? compo.toLocal.rotation : rotation,
                                                                  compo.toLocal.scale),
                                                              parent,
                                                              true,
                                                              frontFace);
                } else {
                    newCompo = scene.add_primitive_component(name,
                                                              rf::TRSTransform(
                                                                  pos / 1000.0f,
                                                                  compo.toLocal.rotation,
                                                                  compo.toLocal.scale),
                                                              parent,
                                                              true,
                                                              frontFace);
                }

                SPDLOG_INFO("add component {}, {}, scale: ({:f}, {:f}, {:f})", public_component_name.c_str(), name.c_str(), compo.toLocal.scale.x, compo.toLocal.scale.y, compo.toLocal.scale.z);

                if (newCompo) {
                    newCompo->setPrimitiveGroupFrom(compo);
                    viewpointCompos.push_back(newCompo);
                }

                break;
            } else {
                if (comp_idx == (components.size() - 1)) {
                    SPDLOG_ERROR("Can't find control-{}, name = {}", index, component_name.c_str());
                    return false;
                }
            }
        }
    }

    return true;
}

int engine::sv_viewpoint_controls_init(
    cudarf::pipe::Ctx &desc,
    std::vector<rf::PrimitiveComponent *> &viewpoint_controls,
    const VirtualControlConfig *config,
    World &world,
    const std::vector<rf::Viewpoint> &viewpoints,
    cudaStream_t cuStream)
{
    std::vector<rf::PrimitiveComponent *> controls;
    std::vector<rf::PrimitiveComponent> components;

    world.control_root() =
        world.scene().add_scene_component(
            controlCompoName,
            rf::TRSTransform(glm::vec3(0.0f, 0.0f, 0.0f),
                               glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
                               glm::vec3(1000.0f)),
            world.scene().get_root());

    rf::AnimationMap unused;

    std::set<std::string> controlNames;

    for (uint32_t index = 0; index < config->controls_count; index++) {
        const ViewpointControlIconSettings *control = &config->controls[index];
        controlNames.insert("control::" + std::string(control->component));
    }

    bool res = loader::load_gltf_model(
        &desc,
        std::string(config->model_path),
        world.scene(),
        world.control_root(),
        "control::",
        unused,
        world.control_model(),
        [&components, &controlNames, config]
        (const rf::PrimitiveComponent &compo, rf::Scene &scene) {
            // if template component is related to viewpoint
            // control, then return true, so that loader
            // doesn't add it as is to scene
            if (controlNames.count(compo.name)) {
                apply_prefab_alpha_multiplier(
                    compo,
                    get_prefab_alpha_multiplier(config, compo.name));
                components.push_back(compo);
                return true;
            } else {
                return false;
            }
        },
        cuStream);

    if (!res) {
        SPDLOG_ERROR("Failed to load viewpoint controls models");
        return -1;
    }
    else {
        viewpoint_control_inserter(config, components, world.scene(),
                                   viewpoints, controls,
                                   world.control_root());
        viewpoint_controls = controls;

        return 0;
    }

    return 0;
}
