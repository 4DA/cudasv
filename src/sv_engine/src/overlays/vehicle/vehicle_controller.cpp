#include <spdlog/spdlog.h>

#include <vector>

#include <rf/renderer/primitive_component.hpp>

#include "vehicle_controller.hpp"

#ifndef NDEBUG

#include <string>
#include <vector>
#include <sstream>
#include <iterator>

void dump_config(const WheelGroupConfig &c)
{
    std::ostringstream out;

    out << "steering: ";
    std::copy(std::begin(c.steeringParts), std::end(c.steeringParts),
              std::ostream_iterator<std::string>(out, ", "));

    out << "\nspinning: ";
    std::copy(std::begin(c.spinningParts), std::end(c.spinningParts),
              std::ostream_iterator<std::string>(out, ", "));

    SPDLOG_INFO("Wheele names: {}", out.str());
}
#endif

WheelComponents::WheelComponents()
{
    for (int i = 0; i < VEHICLE_OVERLAY_WHEELS_NUM; i++) {
        steeringCompos[i] = nullptr;
        spinningCompos[i] = nullptr;
    }
}

WheelComponents::WheelComponents(const WheelGroupConfig &parts_names, const rf::Scene &scene)
{
    for (int i = 0; i < VEHICLE_OVERLAY_WHEELS_NUM; i++) {
        steeringCompos[i] = get_wheel_compo(parts_names.steeringParts[i], scene);
        spinningCompos[i] = get_wheel_compo(parts_names.spinningParts[i], scene);
    }
}


rf::SceneComponent * WheelComponents::get_wheel_compo(const std::string &name,
                                                   const rf::Scene &scene)
{
    if (name.empty()) {
        return nullptr;
    }

    rf::SceneComponent * result = scene.get(name);
    if (!result) {
        SPDLOG_ERROR("Wheel Component `{}` not found", name.c_str());
        return nullptr;
    }

    return result;
}

void WheelComponents::set_spin_angle(rf::SceneComponent *compo, float rotation_angle)
{
    const glm::vec3 MODEL_RIGHT = glm::vec3(0.0f, -1.0f, 0.0f);
    glm::quat spin_rotation = glm::angleAxis(rotation_angle, MODEL_RIGHT);
    compo->toLocal.rotation = spin_rotation;
}

void WheelComponents::set_steering_angle(rf::SceneComponent *compo, float steering_angle)
{
    const glm::vec3 MODEL_UP = glm::vec3(0.0f, 0.0f, 1.0f);
    glm::quat steering_rotation = glm::angleAxis(steering_angle, MODEL_UP);
    compo->toLocal.rotation = steering_rotation;
}

void WheelComponents::set_wheel_orientation(float rotation_angles[VEHICLE_OVERLAY_WHEELS_NUM],
                                     float steering_angles[VEHICLE_OVERLAY_WHEELS_NUM])
{
    for (int i = 0; i < VEHICLE_OVERLAY_WHEELS_NUM; i++) {
        if (steeringCompos[i]) {
            set_steering_angle(steeringCompos[i], steering_angles[i]);
        }

        if (spinningCompos[i]) {
            set_spin_angle(spinningCompos[i], rotation_angles[i]);
        }
    }
}

int VehicleModelController::init(rf::Scene *scene,
                                 const std::string &namePrefix,
                                 const VehicleModelConfig *config)
{
    this->scene = scene;
    this->config = config;
    this->namePrefix = namePrefix;

    wheelGroupConfig.steeringParts[0] = namePrefix +
        config->wheelsConfig.front_left_steering;
    wheelGroupConfig.steeringParts[1] = namePrefix +
        config->wheelsConfig.front_right_steering;
    wheelGroupConfig.steeringParts[2] = namePrefix +
        config->wheelsConfig.rear_left_steering;
    wheelGroupConfig.steeringParts[3] = namePrefix +
        config->wheelsConfig.rear_right_steering;

    wheelGroupConfig.spinningParts[0] = namePrefix +
        config->wheelsConfig.front_left_spinning;
    wheelGroupConfig.spinningParts[1] = namePrefix +
        config->wheelsConfig.front_right_spinning;
    wheelGroupConfig.spinningParts[2] = namePrefix +
        config->wheelsConfig.rear_left_spinning;
    wheelGroupConfig.spinningParts[3] = namePrefix +
        config->wheelsConfig.rear_right_spinning;

#ifndef NDEBUG
        dump_config(wheelGroupConfig);
#endif

    wheels = std::make_unique<WheelComponents>(wheelGroupConfig, *scene);

    return 0;
}

int VehicleModelController::update(const engine::VehicleState *vehicleState)
{
    unsigned int WHEEL_MAX = VEHICLE_OVERLAY_WHEELS_NUM;
    float rotation_angles[VEHICLE_OVERLAY_WHEELS_NUM];
    float steering_angles[VEHICLE_OVERLAY_WHEELS_NUM];

    for (int i = 0; i < vehicle::WHEELS_MAX; i++) {

        rotation_angles[i] =
            glm::radians(vehicleState->wheel_rotation[i].wheel_rotation_angle);
        steering_angles[i] =
            glm::radians(vehicleState->wheel_rotation[i].yaw_angle);
    }

    wheels->set_wheel_orientation(rotation_angles, steering_angles);
    return 0;
}
