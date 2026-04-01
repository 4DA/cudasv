#pragma once

#include <array>
#include <memory>

#include <engine/engine.hpp>

#include <rf/renderer/cudarf/cudarf.hpp>
#include <rf/renderer/cudarf/output.hpp>
#include <rf/renderer/cudarf/draw_list_renderer.hpp>

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

    std::array<std::unique_ptr<cudarf::CudaOutput>, SV_MAX_OUTPUTS> cudaOutput;
    std::array<std::unique_ptr<cudarf::pipe::Ctx>, SV_MAX_OUTPUTS> cuda_rasterizers;
    std::array<cudarf::Framebuffer, SV_MAX_OUTPUTS> mesh_gpu_outputs;
    std::array<cudaStream_t, SV_MAX_OUTPUTS> cudaOutputStreams;

    std::shared_ptr<cudarf::profiling::Events> frameTimeDB = nullptr;
    std::array<std::shared_ptr<cudarf::profiling::Events>, SV_MAX_OUTPUTS> outputRenderTimeDB;

    unsigned int frameCounter = 0;
};

static constexpr float VehicleScaleFactor = 1000.0f;
static constexpr unsigned int DefaultTileQueueLimit = 2000;

} // namespace engine
