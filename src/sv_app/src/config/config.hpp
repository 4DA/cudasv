#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <algorithm>
#include <sstream>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <engine/engine.hpp>
#include <nlohmann/json.hpp>

int load_config(engine::Config *config, std::string path);


NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ColorParameter,
                                   rgba)

namespace engine
{
namespace view
{
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TopViewRect,
                                   top,
                                   bottom,
                                   left,
                                   right)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SphericalViewpoint,
                                   polar,
                                   azimuthal)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Ellipsoid,
                                   X, Y, Z)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ViewpointBoundary,
                                   angle_min, angle_max,
                                   boundary_min, boundary_max)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SkyFade3D,
                                   enabled,
                                   radius_mm,
                                   start_mm,
                                   gradient_mm,
                                   color)

inline PoseMode pose_mode_from_string(const std::string &value)
{
    if (value == "orbital") {
        return POSE_MODE_ORBITAL;
    }
    if (value == "look_at") {
        return POSE_MODE_LOOK_AT;
    }

    throw std::runtime_error("unsupported pose_mode: " + value);
}

inline std::string pose_mode_to_string(PoseMode value)
{
    switch (value) {
    case POSE_MODE_ORBITAL:
        return "orbital";
    case POSE_MODE_LOOK_AT:
        return "look_at";
    }

    throw std::runtime_error("unknown pose mode");
}

inline NavigationMode navigation_mode_from_string(const std::string &value)
{
    if (value == "orbital") {
        return NAVIGATION_MODE_ORBITAL;
    }
    if (value == "topview") {
        return NAVIGATION_MODE_TOPVIEW;
    }
    if (value == "look_around") {
        return NAVIGATION_MODE_LOOK_AROUND;
    }
    if (value == "static") {
        return NAVIGATION_MODE_STATIC;
    }

    throw std::runtime_error("unsupported navigation_mode: " + value);
}

inline std::string navigation_mode_to_string(NavigationMode value)
{
    switch (value) {
    case NAVIGATION_MODE_ORBITAL:
        return "orbital";
    case NAVIGATION_MODE_TOPVIEW:
        return "topview";
    case NAVIGATION_MODE_LOOK_AROUND:
        return "look_around";
    case NAVIGATION_MODE_STATIC:
        return "static";
    }

    throw std::runtime_error("unknown navigation mode");
}

inline void from_json(const nlohmann::json &j, VirtualCamera &camera)
{
    camera = {};

    if (j.contains("position")) {
        const auto &value = j.at("position");
        for (std::size_t i = 0; i < 3; ++i) {
            camera.position[i] = value.at(i).get<float>();
        }
    }
    if (j.contains("look_at")) {
        const auto &value = j.at("look_at");
        for (std::size_t i = 0; i < 3; ++i) {
            camera.look_at[i] = value.at(i).get<float>();
        }
    }
    if (j.contains("up")) {
        const auto &value = j.at("up");
        for (std::size_t i = 0; i < 3; ++i) {
            camera.up[i] = value.at(i).get<float>();
        }
    }

    camera.vfov = j.at("vfov").get<float>();
    camera.z_far = j.at("z_far").get<float>();
    camera.z_near = j.at("z_near").get<float>();
}

inline void to_json(nlohmann::json &j, const VirtualCamera &camera)
{
    j = nlohmann::json{
        {"position", {camera.position[0], camera.position[1], camera.position[2]}},
        {"look_at", {camera.look_at[0], camera.look_at[1], camera.look_at[2]}},
        {"up", {camera.up[0], camera.up[1], camera.up[2]}},
        {"vfov", camera.vfov},
        {"z_far", camera.z_far},
        {"z_near", camera.z_near},
    };
}

inline void from_json(const nlohmann::json &j, Viewpoint3D &viewpoint)
{
    viewpoint = {};

    viewpoint.poseMode =
        pose_mode_from_string(j.at("pose_mode").get<std::string>());
    viewpoint.navigationMode =
        navigation_mode_from_string(j.at("navigation_mode").get<std::string>());

    if (j.contains("rotator")) {
        viewpoint.rotator = j.at("rotator").get<Ellipsoid>();
    }
    if (j.contains("spherical")) {
        viewpoint.spherical = j.at("spherical").get<SphericalViewpoint>();
    }
    if (j.contains("boundary")) {
        viewpoint.boundary = j.at("boundary").get<ViewpointBoundary>();
    }
    if (j.contains("camera")) {
        viewpoint.camera = j.at("camera").get<VirtualCamera>();
    }
}

inline void to_json(nlohmann::json &j, const Viewpoint3D &viewpoint)
{
    j = nlohmann::json{
        {"pose_mode", pose_mode_to_string(viewpoint.poseMode)},
        {"navigation_mode", navigation_mode_to_string(viewpoint.navigationMode)},
        {"rotator", viewpoint.rotator},
        {"spherical", viewpoint.spherical},
        {"boundary", viewpoint.boundary},
        {"camera", viewpoint.camera},
    };
}

inline void from_json(const nlohmann::json &j, ViewConfig3D &config)
{
    config = {};
    config.enabled = j.at("enabled").get<int>();
    config.viewpoint = j.at("viewpoint").get<Viewpoint3D>();
    config.viewpoints_count = j.at("viewpoints_count").get<uint32_t>();

    const auto presets = j.at("viewpoint_presets").get<std::vector<Viewpoint3D>>();
    const auto copy_count = std::min<std::size_t>(presets.size(), SV_MAX_VIEWPOINTS);
    for (std::size_t i = 0; i < copy_count; ++i) {
        config.viewpoint_presets[i] = presets[i];
    }

    config.duration_ms = j.at("duration_ms").get<unsigned int>();
    config.min_rotator_scale = j.at("min_rotator_scale").get<float>();
    config.max_rotator_scale = j.at("max_rotator_scale").get<float>();
    config.topview_limits = j.at("topview_limits").get<TopViewRect>();
    config.exposure = j.at("exposure").get<float>();
    config.sky_fade = j.at("sky_fade").get<SkyFade3D>();
}

inline void to_json(nlohmann::json &j, const ViewConfig3D &config)
{
    std::vector<Viewpoint3D> presets(config.viewpoint_presets,
                                     config.viewpoint_presets + config.viewpoints_count);
    j = nlohmann::json{
        {"enabled", config.enabled},
        {"viewpoint", config.viewpoint},
        {"viewpoints_count", config.viewpoints_count},
        {"viewpoint_presets", presets},
        {"duration_ms", config.duration_ms},
        {"min_rotator_scale", config.min_rotator_scale},
        {"max_rotator_scale", config.max_rotator_scale},
        {"topview_limits", config.topview_limits},
        {"exposure", config.exposure},
        {"sky_fade", config.sky_fade},
    };
}

inline void from_json(const nlohmann::json &j, ViewsConfig &config)
{
    config.view_3d = j.at("view_3d").get<ViewConfig3D>();
}

inline void to_json(nlohmann::json &j, const ViewsConfig &config)
{
    j = nlohmann::json{{"view_3d", config.view_3d}};
}
}
}

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(RendererConfig,
                                   use_ibl)


NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(CarUnderlayConfig,
                                   blind_front,
                                   blind_rear,
                                   blind_left,
                                   blind_right,
                                   texture_path)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(WheelsConfig,
                                   front_left_steering, front_right_steering,
                                   rear_left_steering, rear_right_steering,
                                   front_left_spinning, front_right_spinning,
                                   rear_left_spinning, rear_right_spinning)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(VehicleModelConfig,
                                   model_path,
                                   wheelsConfig,
                                   ibl_path,
                                   use_velocity_for_wheels_animation)

inline void from_json(const nlohmann::json &j, ViewpointControlIconSettings &settings)
{
    settings = {};
    settings.viewpoint = j.at("viewpoint").get<int>();
    settings.component = j.at("component").get<std::string>();

    const auto &position = j.at("position");
    for (std::size_t i = 0; i < 3; ++i) {
        settings.position[i] = position.at(i).get<float>();
    }

    settings.is_billboard = j.at("is_billboard").get<int>();
    settings.is_viewpoint_oriented = j.value("is_viewpoint_oriented", 0);

    if (j.contains("look_at")) {
        const auto &look_at = j.at("look_at");
        for (std::size_t i = 0; i < 3; ++i) {
            settings.look_at[i] = look_at.at(i).get<float>();
        }
    }
}

inline void to_json(nlohmann::json &j, const ViewpointControlIconSettings &settings)
{
    j = nlohmann::json{
        {"viewpoint", settings.viewpoint},
        {"component", settings.component},
        {"position", {settings.position[0], settings.position[1], settings.position[2]}},
        {"is_billboard", settings.is_billboard},
        {"is_viewpoint_oriented", settings.is_viewpoint_oriented},
        {"look_at", {settings.look_at[0], settings.look_at[1], settings.look_at[2]}},
    };
}

inline void from_json(const nlohmann::json &j, VirtualControlConfig &config)
{
    config = {};
    config.enabled = j.at("enabled").get<int>();
    config.model_path = j.at("model_path").get<std::string>();
    config.controls_count = j.at("controls_count").get<uint32_t>();

    const auto controls = j.at("controls").get<std::vector<ViewpointControlIconSettings>>();
    const auto copy_count = std::min<std::size_t>(controls.size(), SV_MAX_VIEWPOINTS);
    for (std::size_t i = 0; i < copy_count; ++i) {
        config.controls[i] = controls[i];
    }
}

inline void to_json(nlohmann::json &j, const VirtualControlConfig &config)
{
    std::vector<ViewpointControlIconSettings> controls(config.controls,
                                                       config.controls + config.controls_count);
    j = nlohmann::json{
        {"enabled", config.enabled},
        {"model_path", config.model_path},
        {"controls_count", config.controls_count},
        {"controls", controls},
    };
}

inline void from_json(const nlohmann::json &j, OverlaysConfig &config)
{
    config.renderer_config = j.at("renderer_config").get<RendererConfig>();
    config.underlay_config = j.at("underlay_config").get<CarUnderlayConfig>();
    config.vehicle_config = j.at("vehicle_config").get<VehicleModelConfig>();
    config.controls_config = j.at("controls_config").get<VirtualControlConfig>();
}

inline void to_json(nlohmann::json &j, const OverlaysConfig &config)
{
    j = nlohmann::json{
        {"renderer_config", config.renderer_config},
        {"underlay_config", config.underlay_config},
        {"vehicle_config", config.vehicle_config},
        {"controls_config", config.controls_config},
    };
}

namespace vehicle
{
inline void from_json(const nlohmann::json &j, VehicleDimensions &vehicle)
{
    vehicle = {};
    vehicle.length = j.at("length").get<float>();
    vehicle.width = j.at("width").get<float>();
    vehicle.max_pulse = j.at("max_pulse").get<uint32_t>();
    vehicle.steering_ratio = j.at("steering_ratio").get<float>();
    vehicle.wheel_diameter_mm = j.at("wheel_diameter_mm").get<float>();
}

inline void to_json(nlohmann::json &j, const VehicleDimensions &vehicle)
{
    j = nlohmann::json{
        {"length", vehicle.length},
        {"width", vehicle.width},
        {"max_pulse", vehicle.max_pulse},
        {"steering_ratio", vehicle.steering_ratio},
        {"wheel_diameter_mm", vehicle.wheel_diameter_mm},
    };
}
}

namespace engine
{

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(OutputConfig,
                                   display_width,
                                   display_height)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Output,
                                   config,
                                   active,
                                   views_count,
                                   active_view_ids,
                                   name)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(OutputSet,
                                   outputs)
}

template <typename T>
int loadJsonConfig(T &result, const std::string& filename, const std::string& key = "")
{
    std::ifstream in(filename);
    if (!in.is_open()) {
        std::cout << "Can not open " << filename << "\n";
        return -1;
    }

    try {
        nlohmann::json json;
        in >> json;
        if (key != "") {
            result = json[key].get<T>();
        } else {
            result = json.get<T>();
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return -1;
    }

    return 0;
}

#endif  // CONFIG_HPP
