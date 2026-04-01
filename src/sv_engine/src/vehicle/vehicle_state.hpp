#ifndef SV_VEHICLE_STATE_HPP
#define SV_VEHICLE_STATE_HPP

#include <engine/engine.hpp>

namespace engine
{

typedef struct
{
    /* Flag used to avoid delta calculation at system start */
    int             initialized;
    /* Last received value of wheel pulse counter */
    int16_t         count;
    /* Previous received value of wheel pulse counter */
    int16_t         prev_count;
    /* Previous delta value without filtering */
    int16_t         prev_delta;
    /* Previous delta value with filtering */
    float           prev_filtered_delta;
    // rotation angle in degrees
    float           wheel_rotation_angle;
    // wheel yaw angle (depends on steering angle)
    float           yaw_angle;

} sv_wheel_state_t;

struct VehicleState
{
    vehicle::CANSignals current_state;

    float wheel_angles[vehicle::WHEELS_MAX];

    sv_wheel_state_t wheel_rotation[vehicle::WHEELS_MAX];

    //
    uint64_t        timestamp_ns;

};


int sv_vehicle_state_update(
    engine::VehicleState *state,
    const vehicle::VehicleDimensions *vehicle_config,
    const VehicleModelConfig *vehicle_model_config);

int sv_vehicle_state_handle_signals(
    engine::VehicleState *state,
    const vehicle::VehicleDimensions *vehicle_config,
    const VehicleModelConfig *vehicle_model_config,
    const vehicle::CANSignals *vehicle_signals);

} // namespace engine

#endif
