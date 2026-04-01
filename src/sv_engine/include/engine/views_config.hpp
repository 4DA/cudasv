#ifndef SV_VIEWS_CONFIG_HPP
#define SV_VIEWS_CONFIG_HPP

#include <engine/overlays_config.hpp>

namespace engine
{
namespace view
{

// SV view identifiers
enum ViewID
{
    SV_VIEW_3D,
    SV_VIEWS_MAX
};

// Top view rectangle dimensions
struct TopViewRect
{
    // Top border (vehicle front) in millimeters
    float top;
    // Bottom border (vehicle rear) in millimeters
    float bottom;
    // Left border (vehicle left) in millimeters
    float left;
    // Right border (vehicle right) in millimeters
    float right;
};

// Virtual camera settings using Cartesian coordinates
struct VirtualCamera
{
    // Translation/position point
    float position[3];
    // Look-at camera direction view point (used if is_look_at is true)
    float look_at[3];
    // Camera up vector (used if is_look_at is true)
    float up[3];
    // Vertical field-of-view in degrees
    float vfov;
    // Far plane clipping distance for 3D camera in millimeters
    float z_far;
    // Near plane clipping distance for 3D camera in millimeters
    float z_near;
};

// Virtual 3D camera settings using spherical coordinates
struct SphericalViewpoint
{
    // Polar angle in degrees
    float polar;
    // Azimuthal angle in degrees
    float azimuthal;
};

// Ellipsoid trajectory parameters for spherical-based 3D camera
struct Ellipsoid
{
    // Ellipsoid trajectory parameters in millimeters for spherical-based 3D camera
    float X, Y, Z;
};

// Camera 3D viewpoint boundaries
struct ViewpointBoundary
{
    // View angle minimum boundary
    SphericalViewpoint angle_min;
    // View angle maximum boundary
    SphericalViewpoint angle_max;
    // Minimum translation boundaries
    Ellipsoid boundary_min;
    // Maximum translation boundaries
    Ellipsoid boundary_max;
};

enum PoseMode
{
    POSE_MODE_ORBITAL,
    POSE_MODE_LOOK_AT,
};

// 3D camera viewpoint navigation behavior
enum NavigationMode
{
    NAVIGATION_MODE_ORBITAL,
    NAVIGATION_MODE_TOPVIEW,
    NAVIGATION_MODE_LOOK_AROUND,
    NAVIGATION_MODE_STATIC,
};

// 3D camera viewpoint settings
struct Viewpoint3D
{
    // How the viewpoint pose is represented.
    PoseMode poseMode;
    // Ellipsoid trajectory settings
    Ellipsoid rotator;
    // Spherical coordinates (used with orbital pose mode)
    SphericalViewpoint spherical;
    // Translation and angles boundaries
    ViewpointBoundary boundary;
    // Cartesian camera definition (used with look-at pose mode)
    VirtualCamera camera;
    // How interactive navigation updates the viewpoint
    NavigationMode navigationMode;
};


// 3D view screen fade settings for hiding occlusions/camera frame end
struct SkyFade3D
{
    // Enabled (1) or disabled (0)
    int enabled;
    // Fading cylinder radius in millimeters
    float radius_mm;
    // Fading cylinder starting height in millimeters
    float start_mm;
    // Fading cylinder gradient sector in millimeters
    float gradient_mm;
    // Fade color in RGBA format
    ColorParameter color;
};

// 3D view configuration settings
struct ViewConfig3D
{
    // Enabled (1) or disabled (0)
    int enabled;
    // Initial 3D viewpoint settings (used at initialization only)
    Viewpoint3D viewpoint;
    // Count of 3D viewpoint presets
    uint32_t viewpoints_count;
    // 3D viewpoint presets (used at initialization only)
    Viewpoint3D viewpoint_presets[SV_MAX_VIEWPOINTS];
    // Camera viewpoint animation duration in milliseconds
    unsigned int duration_ms;
    // Minimum zoom value for 3D camera viewpoint with orbital controller
    float min_rotator_scale;
    // Maximum zoom value for 3D camera viewpoint with orbital controller
    float max_rotator_scale;
    // Boundaries (front, rear, left, right in millimeters) for 3D camera viewpoint with top view controller
    TopViewRect topview_limits;
    // Exposure light value for 3D scene for GLTF-based overlays
    float exposure;
    // Settings for sky fade in 3D scene
    SkyFade3D sky_fade;
};

// CE SV Engine views configuration settings
struct ViewsConfig
{
    // 3D view configuration (SV_VIEW_3D)
    ViewConfig3D view_3d;
};

} // namespace view

} // namespace engine

#endif
