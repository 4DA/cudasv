#pragma once

#include <array>
#include <memory>
#include <optional>
#include <vector>

#include <engine/engine.hpp>

#include <rf/renderer/animation.hpp>
#include <rf/renderer/cudarf/cudarf.hpp>
#include <rf/renderer/cudarf/array_surface.hpp>
#include <rf/renderer/cudarf/output.hpp>
#include <rf/renderer/cudarf/draw_list_renderer.hpp>
#include <rf/renderer/cudarf/stream.hpp>

#include <vehicle/vehicle_state.hpp>
#include <views/view_3d.hpp>
#include <world.hpp>

namespace engine
{

struct EngineImpl
{
    std::unique_ptr<World> world;
    std::unique_ptr<view::View3D> view_3d;

    engine::VehicleState vehicleState;

    std::array<cudarf::Stream, SV_MAX_OUTPUTS> cudaOutputStreams;
    std::array<std::unique_ptr<cudarf::CudaOutput>, SV_MAX_OUTPUTS> cudaOutput;
    std::array<std::unique_ptr<cudarf::pipe::Ctx>, SV_MAX_OUTPUTS> cuda_rasterizers;
#ifdef WITH_TAA
    std::array<std::optional<cudarf::memory::ArraySurface>, SV_MAX_OUTPUTS> meshGpuOutputResources;
#endif
    std::array<cudarf::Framebuffer, SV_MAX_OUTPUTS> meshGpuOutputs{};

    std::shared_ptr<cudarf::profiling::Events> frameTimeDB = nullptr;
    std::array<std::shared_ptr<cudarf::profiling::Events>, SV_MAX_OUTPUTS> outputRenderTimeDB;

    std::vector<tinygltf::Model> testScenarioModels;
    std::vector<rf::AnimationMap> testScenarioAnimations;

    unsigned int frameCounter = 0;
};

static constexpr float VehicleScaleFactor = 1000.0f;
static constexpr unsigned int DefaultTileQueueLimit = 2000;

} // namespace engine
