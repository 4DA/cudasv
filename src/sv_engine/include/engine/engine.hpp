#ifndef SV_ENGINE_H
#define SV_ENGINE_H

#include <engine/camera_config.hpp>
#include <engine/vehicle_data.hpp>

#include <engine/overlays_config.hpp>
#include <engine/views_config.hpp>

#include <engine/frame_packet.hpp>
#include <engine/video_source.hpp>

inline constexpr unsigned int SV_MAX_OUTPUTS = 3;

namespace engine
{

enum Error
{
    ERROR = -1,
    OK = 0,
    MEMORY_ERROR,
    BAD_PARAMETER,
};

struct CalibrationConfig
{
    camera::CameraConfig camera_cfg[camera::CAMERAS_COUNT];
};

struct OutputConfig
{
    int display_width;
    int display_height;
};

struct Output
{
    static constexpr unsigned int SV_MAX_VIEWS_PER_OUTPUT = 8;
    OutputConfig config;

    int active;

    int views_count;

    view::ViewID active_view_ids[SV_MAX_VIEWS_PER_OUTPUT];

    std::string name;
};


struct OutputSet
{
    Output outputs[SV_MAX_OUTPUTS];
};

struct Config
{
    // Calibration config
    CalibrationConfig calibration_config;
    // Number of outputs (used at initialization)
    int outputs_number;
    // Outputs configuration (used at initialization)
    OutputConfig outputs[SV_MAX_OUTPUTS];
    // Vehicle parameters config
    vehicle::VehicleDimensions vehicle_config;
    // Views config
    view::ViewsConfig views_config;
    // Overlays config
    OverlaysConfig overlays_config;

};

enum InputEventType
{
    TAP,
    PAN,
    DOUBLE_TAP,
    PINCH_ZOOM,
};

struct InputEvent
{
    InputEventType type;
    float x;     // NDC space
    float y;     // NDC space
    float scale; // for zoom
};

struct EngineImpl;

struct Engine
{
    EngineImpl *_impl;
    Config config;

    Engine();

    ~Engine();

    engine::Error init();

    engine::Error update_vehicle_state(const vehicle::CANSignals *vehicle_signals);

    engine::Error pre_process(const videoio::RuntimeFramePacket4Cam &frame_packet);

    engine::Error process(const videoio::RuntimeFramePacket4Cam &frame_packet,
                               void* output_buffer,
                               unsigned long long cuda_str,
                               uint32_t width,
                               uint32_t height,
                               float clear_color[4],
                               const Output &output,
                               int output_index);

    engine::Error view_animate(int view, int viewpoint);

    engine::Error get_viewpoint(int view, int *viewpoint);

    engine::Error input_event(const Output *output, InputEvent *event);

    engine::Error get_animation_status(int view,
                                       int *is_active,
                                       unsigned int *timeleft_ms);
};

}
#endif
