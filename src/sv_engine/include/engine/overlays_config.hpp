#ifndef OVERLAYS_CONFIG_HPP
#define OVERLAYS_CONFIG_HPP

#include <string>

#include <engine/vehicle_data.hpp>

#define SV_MAX_STRING (255)
#define SV_MAX_VIEWPOINTS (15)

enum SV3DOverlayID
{
    SV_3D_CAR_MODEL,
    SV_3D_VIRTUAL_CONTROLS,
    SV_3D_OVERLAYS_MAX
};

struct ColorParameter
{
    float rgba[4];
};

struct WheelsConfig
{
    std::string front_left_steering;
    std::string front_right_steering;
    std::string rear_left_steering;
    std::string rear_right_steering;

    std::string front_left_spinning;
    std::string front_right_spinning;
    std::string rear_left_spinning;
    std::string rear_right_spinning;
};

// 3D vehicle model overlay config
struct VehicleModelConfig
{
    // Path to model file
    std::string model_path;

    // black underlay to hide camera stitching
    std::string underlay_texture_path;

    // Path to IBL map containing directory
    std::string ibl_path;

    // Path to model file
    char overlay_path[SV_MAX_STRING];

    // wheel names
    WheelsConfig wheelsConfig;

    // Use per-wheel velocity for wheels spinning animation or pulses
    int use_velocity_for_wheels_animation;
};

struct ViewpointControlIconSettings
{
    // corresponding 3D view viewpoint
    int viewpoint;

    // Name of component in virtual controls model file
    std::string component;

    // Position of virtual control in 3D scene
    float position[3];

    // Draw as always front face
    int is_billboard;

    // Orient in space based on viewpoint
    int is_viewpoint_oriented;

    // Look at point for orientation if is_viewpoint_oriented is 0
    float look_at[3];
} ;

struct VirtualControlConfig
{
    // Enabled (1) or disabled (0)
    int enabled;

    // Path to model file
    std::string model_path;

    // used control number
    uint32_t controls_count;

    // Single control settings
    ViewpointControlIconSettings controls[SV_MAX_VIEWPOINTS];
};

// car bottom shadow config
struct CarUnderlayConfig
{
    // Front border of underlay in mm
    float blind_front;

    // Rear border of underlay in mm
    float blind_rear;

    // Right border of underlay in mm
    float blind_right;

    // Left border of underlay in mm
    float blind_left;

    // Underlay color texture file name (priority over color)
    std::string texture_path;
};

struct RendererConfig
{
    int use_ibl;
} ;

struct OverlaysConfig
{
    // 3D renderer config
    RendererConfig renderer_config;

    // Underlay/bottom shadow config
    CarUnderlayConfig underlay_config;

    // 3D vehicle model config
    VehicleModelConfig vehicle_config;

    // 3D virtual controls config
    VirtualControlConfig controls_config;
};

#endif
