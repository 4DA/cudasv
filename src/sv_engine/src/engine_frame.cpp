#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/printf.h>

#include <engine/engine.hpp>

#include <rf/renderer/cuda_helpers.hpp>

#include "engine_internal.hpp"

using namespace engine;

namespace
{

#ifdef WITH_TAA
void configure_default_taa(cudarf::pipe::Ctx *rasterizer)
{
    auto &taa = rasterizer->TAA;
    taa.pattern = cudarf::TAA_Pattern::Halton;
    taa.scale = 1.0f;
    taa.feedback = 0.70f;
    taa.velocityThreshold = 0.1f;
}

#endif

} // namespace

engine::Error Engine::update_vehicle_state(const vehicle::CANSignals *vehicle_signals)
{
    OverlaysConfig *overlays_config = &config.overlays_config;
    VehicleModelConfig *vehicle_model_config = &overlays_config->vehicle_config;
    vehicle::VehicleDimensions *vehicle_config = &config.vehicle_config;

    sv_vehicle_state_handle_signals(&_impl->vehicleState,
                                    vehicle_config,
                                    vehicle_model_config,
                                    vehicle_signals);

    return OK;
}

engine::Error Engine::pre_process(const videoio::RuntimeFramePacket4Cam &frame_packet)
{
    const videoio::FrameSet<camera::CAMERAS_TOTAL> &frames_set = frame_packet.frames;

    for (bool is_valid : frame_packet.valid_cameras) {
        if (!is_valid) {
            SPDLOG_ERROR("Current runtime requires all render-bridge cameras to be valid in each frame packet");
            return ERROR;
        }
    }

    OverlaysConfig *overlays_config = &config.overlays_config;
    vehicle::VehicleDimensions *vehicle_config = &config.vehicle_config;
    VehicleModelConfig *vehicle_model_config = &overlays_config->vehicle_config;

    sv_vehicle_state_update(&_impl->vehicleState,
                            vehicle_config,
                            vehicle_model_config);

    _impl->world->vehicle_controller().update(&_impl->vehicleState);

    if (_impl->world->frame_projector().prevFrameSeq != frames_set.frameseq) {
        _impl->world->camera_textures() =
            _impl->world->frame_projector().load_rgb(
                {
                    frames_set.data[camera::CAMERA_RIGHT],
                    frames_set.data[camera::CAMERA_LEFT],
                    frames_set.data[camera::CAMERA_FRONT],
                    frames_set.data[camera::CAMERA_REAR]
                },
                0,
                frames_set.width,
                frames_set.height,
                frames_set.stride,
                _impl->cudaOutputStreams[0]);
    }

    _impl->world->frame_projector().prevFrameSeq = frames_set.frameseq;

    rf::Scene &scene = _impl->world->scene();
    assert(scene.get_root());
    scene.get_root()->update_transforms(nullptr);

    return OK;
}

engine::Error Engine::process(const videoio::RuntimeFramePacket4Cam &frame_packet,
                              void *output_buffer,
                              unsigned long long cuda_str,
                              uint32_t width,
                              uint32_t height,
                              float clear_color[4],
                              const Output &output,
                              int output_index)
{
    const videoio::FrameSet<camera::CAMERAS_TOTAL> &frames_set = frame_packet.frames;

    (void)output_buffer;
    (void)cuda_str;
    (void)clear_color;

    cudarf::pipe::Ctx *cuda_rasterizer = _impl->cuda_rasterizers[output_index].get();
    cudarf::CudaOutput *cuda_output = _impl->cudaOutput[output_index].get();
    cudaStream_t cudaStream = _impl->cudaOutputStreams[0];
    cudarf::Framebuffer meshGpuOutput = _impl->mesh_gpu_outputs[output_index];

    if constexpr (CUDARF_ENABLE_CUDA_PROFILING) {
        if (_impl->frameCounter == 0) {
            _impl->frameTimeDB->clear_history();
        }
        _impl->frameTimeDB->clear();
    }

#ifdef WITH_TAA
    configure_default_taa(cuda_rasterizer);
#endif

    if (!_impl->view_3d) {
        return engine::Error::ERROR;
    }

    _impl->view_3d->update_camera();

    const rf::VirtualCamera &virtualCamera = *_impl->view_3d->virtual_camera();
    const rf::Scene &scene = _impl->world->scene();
    const rf::IBL &ibl = scene.get_ibl();

    std::vector<cudarf::CUDARFLight> lightList;

    for (auto compIt: scene.get_lights()) {
        rf::PointLightComponent &comp = *compIt.second;
        auto trans = comp.toWorld.translation;

        cudarf::CUDARFLight light {
            .intensity = comp.intensity,
            .position = make_float3(trans.x, trans.y, trans.z),
            .range = 10000.0
        };

        lightList.push_back(light);
    }

    assert(ibl.specular);

    auto camera_translation = make_float3(virtualCamera.transform.translation.x,
                                          virtualCamera.transform.translation.y,
                                          virtualCamera.transform.translation.z);

    cudarf::PBRParams pbrCommon{camera_translation, virtualCamera.exposure,
        lightList, ibl.get_sh_matrix(), ibl.brdfLUT, ibl.specular};

    assert(pbrCommon.specular);

    cudarf::pipe::begin_frame(cuda_rasterizer,
                              virtualCamera,
                              pbrCommon,
                              _impl->view_3d->postprocess_pipe()->historyUniforms(),
                              _impl->frameCounter,
                              cudaStream);

    bool surroundViewRendered = false;

    for (int viewIndex = 0; viewIndex < output.views_count; ++viewIndex) {
        const int active_view = output.active_view_ids[viewIndex];

        switch (active_view) {
        case view::SV_VIEW_3D:
            if (_impl->view_3d) {
                view::View3D *view = _impl->view_3d.get();

                view->compose(frames_set,
                              cuda_output->d_output,
                              meshGpuOutput,
                              width,
                              height,
                              _impl->vehicleState.current_state.steering_angle.value,
                              {cudaStream},
                              *_impl->frameTimeDB,
                              _impl->frameCounter);

                surroundViewRendered = true;
            }
            break;
        default:
            SPDLOG_ERROR("{}", fmt::sprintf("Unknown view type %d", active_view));
            assert(false);
            return ERROR;
        }
    }

    if constexpr (CUDARF_ENABLE_CUDA_PROFILING) {
        SPDLOG_INFO("{}", fmt::sprintf("frame counter: %u", _impl->frameCounter));
        _impl->frameTimeDB->show();
    }

    CUDA_CHK(cudaStreamSynchronize(cudaStream));
    cuda_output->present(cuda_output->d_output);

    if (surroundViewRendered) {
        _impl->frameCounter++;
    }

    return OK;
}
