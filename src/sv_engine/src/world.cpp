#include <spdlog/spdlog.h>

#include <world.hpp>

#include <rf/renderer/gltf_loader.hpp>

using namespace engine;

const std::string vehicleCompoName = "VehicleRoot";
const std::string controlCompoName = "ControlRoot";

World::World() {
    sceneRuntime.scene = std::make_unique<rf::Scene>();
}

int World::init(cudarf::pipe::Ctx *desc, const Config *config, cudaStream_t cuStream)
{
    const OverlaysConfig *overlays_config = &config->overlays_config;
    const VehicleModelConfig *vehicle_model_config = &overlays_config->vehicle_config;

    if (overlays_config->renderer_config.use_ibl)
    {
        SPDLOG_INFO("Loading IBL from path: {}", vehicle_model_config->ibl_path);
        scene().set_ibl(rf::load_ibl(std::string(vehicle_model_config->ibl_path), cuStream));
    }

    overlayRuntime.vehicleController = std::make_unique<VehicleModelController>();
    if (overlayRuntime.vehicleController->init(&scene(),
                                               "vehicle::",
                                               vehicle_model_config))
    {
        SPDLOG_ERROR("Failed to init vehicle model controller");
        return -1;
    }

    overlayRuntime.underlay = std::make_unique<Underlay>();

    if (overlayRuntime.underlay->init(desc, scene(), config, cuStream)) {
        SPDLOG_ERROR("Failed to init underlay");
        return -1;
    }

    return 0;
}

rf::Scene &World::scene()
{
    return *sceneRuntime.scene;
}

const rf::Scene &World::scene() const
{
    return *sceneRuntime.scene;
}

rf::SceneComponent *&World::vehicle_component()
{
    return sceneRuntime.vehicleComponent;
}

tinygltf::Model &World::vehicle_model()
{
    return sceneRuntime.vehicleModel;
}

rf::AnimationMap &World::animations()
{
    return sceneRuntime.animations;
}

rf::SceneComponent *&World::control_root()
{
    return sceneRuntime.controlRoot;
}

tinygltf::Model &World::control_model()
{
    return sceneRuntime.controlModel;
}

VehicleModelController &World::vehicle_controller()
{
    return *overlayRuntime.vehicleController;
}

Underlay &World::underlay()
{
    return *overlayRuntime.underlay;
}

rf::surround_view::Projector &World::frame_projector()
{
    return frameProjectionRuntime.projector;
}

const rf::surround_view::Projector &World::frame_projector() const
{
    return frameProjectionRuntime.projector;
}

std::array<cudaTextureObject_t, camera::CAMERAS_COUNT> &World::camera_textures()
{
    return frameProjectionRuntime.cameraTextures;
}

const std::array<cudaTextureObject_t, camera::CAMERAS_COUNT> &World::camera_textures() const
{
    return frameProjectionRuntime.cameraTextures;
}
