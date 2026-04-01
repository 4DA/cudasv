#ifndef SV_VEHICLE_CONTROLLER_HPP
#define SV_VEHICLE_CONTROLLER_HPP

#include <engine/engine.hpp>

#include <vehicle/vehicle_state.hpp>

#include <rf/renderer/trs_transform.hpp>
#include <rf/renderer/animation.hpp>

#include <rf/overlays/vehicle/vehicle.hpp>

// WheelGroupConfig specifies the names of the wheels,

// "steering parts" exhibit a nonzero yaw angle when the steering angle is
// active, and "spinning parts" rotate around their axis when the vehicle is in
// motion. The arrangement follows this order: front_left, front_right,
// rear_left, rear_right (up to VEHICLE_OVERLAY_WHEELS_NUM parts)
struct WheelGroupConfig
{
    std::string steeringParts[VEHICLE_OVERLAY_WHEELS_NUM];
    std::string spinningParts[VEHICLE_OVERLAY_WHEELS_NUM];
};

// WheelGroup contains all the necessary data to animate wheels of the vehicle
// Steering and spinning for the front wheels [left, right]
// Spinning for the rear wheels [left, right]
class WheelComponents
{
public:
    WheelComponents();

    WheelComponents(const WheelGroupConfig &names, const rf::Scene &scene);

    // Adjust steering angles for front wheels and enable spinning for all wheels
    void set_wheel_orientation(float rotation_angles[VEHICLE_OVERLAY_WHEELS_NUM],
                               float steering_angles[VEHICLE_OVERLAY_WHEELS_NUM]);


private:
    rf::SceneComponent * get_wheel_compo(const std::string &name, const rf::Scene &scene);
    void set_spin_angle(rf::SceneComponent *compo, float rotation_angle);
    void set_steering_angle(rf::SceneComponent *compo, float steering_angle);

    rf::SceneComponent *steeringCompos[VEHICLE_OVERLAY_WHEELS_NUM];
    rf::SceneComponent *spinningCompos[VEHICLE_OVERLAY_WHEELS_NUM];
};

class VehicleModelController
{
public:
    int init(rf::Scene *scene,
             const std::string &prefix,
             const VehicleModelConfig *config);

    int update(const engine::VehicleState *vehicleState);

private:
    std::string namePrefix;
    const VehicleModelConfig *config;
    rf::Scene *scene;
    WheelGroupConfig wheelGroupConfig;
    std::unique_ptr<WheelComponents> wheels;
};

#endif  // SV_VEHICLE_CONTROLLER_HPP
