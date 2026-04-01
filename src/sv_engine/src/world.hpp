#ifndef SV_WORLD_HPP
#define SV_WORLD_HPP

#include <memory>

#include <engine/camera_config.hpp>
#include <engine/engine.hpp>

#include <rf/renderer/gltf_common.hpp>

#include <overlays/underlay/underlay.hpp>
#include <overlays/vehicle/vehicle_controller.hpp>

#include <rf/renderer/surround_view/projector.hpp>

/// root component name for vehicle
///
extern const std::string vehicleCompoName;

/// root component name for viewpoint controls
extern const std::string controlCompoName;

namespace engine
{

class World {
public:
    World();

    int init(cudarf::pipe::Ctx *desc, const engine::Config *config, cudaStream_t cuStream);

    rf::Scene &scene();
    const rf::Scene &scene() const;

    rf::SceneComponent *&vehicle_component();
    tinygltf::Model &vehicle_model();
    rf::AnimationMap &animations();

    rf::SceneComponent *&control_root();
    tinygltf::Model &control_model();

    VehicleModelController &vehicle_controller();
    Underlay &underlay();

    rf::surround_view::Projector &frame_projector();
    const rf::surround_view::Projector &frame_projector() const;

    std::array<cudaTextureObject_t, camera::CAMERAS_COUNT> &camera_textures();
    const std::array<cudaTextureObject_t, camera::CAMERAS_COUNT> &camera_textures() const;

private:
    struct SceneRuntime
    {
        std::unique_ptr<rf::Scene> scene;
        rf::SceneComponent *vehicleComponent = nullptr;
        tinygltf::Model vehicleModel;
        rf::AnimationMap animations;

        rf::SceneComponent *controlRoot = nullptr;
        tinygltf::Model controlModel;
    };

    struct OverlayRuntime
    {
        std::unique_ptr<VehicleModelController> vehicleController;
        std::unique_ptr<Underlay> underlay;
    };

    struct FrameProjectionRuntime
    {
        rf::surround_view::Projector projector;
        std::array<cudaTextureObject_t, camera::CAMERAS_COUNT> cameraTextures = {};
    };

    SceneRuntime sceneRuntime;
    OverlayRuntime overlayRuntime;
    FrameProjectionRuntime frameProjectionRuntime;
};

}

#endif /* SV_WORLD_HPP */
