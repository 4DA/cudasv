#include <cmath>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/printf.h>

#include <rf/renderer/glm_common.hpp>

#include <engine/engine.hpp>

#include <sv_gl_common.hpp>
#include "vehicle_state.hpp"

using namespace vehicle;

namespace engine
{

static inline float wheel_position(float speed, float deltaTime,
                                   float prevPosition, float diameter_mm,
                                   float smoothing_ratio)
{
    float position = prevPosition;

    float delta = (float)deltaTime / 1000.0f;

    // kph to m/s
    float F = (speed * 0.277778 * delta) / (6.28318530718 * diameter_mm / 1000.0f);

    F = F * 360.0;

    /* ratio to achieve better visual effect */
    position += F / smoothing_ratio;

    position = fmodf(position, 360.0f);

    if (position < 0)
    {
        position += 360.0f;
    }

    return position;
}

static void update_wheel_rotation_angle(engine::VehicleState *state,
                                        uint32_t count,
                                        int wheel,
                                        float ratio,
                                        int16_t delta_limit,
                                        uint16_t dimension_mask)
{
    int delta;
    sv_wheel_state_t *rotation = &state->wheel_rotation[wheel];

    rotation->count = count;

    /* calculate raw pulse delta given the CAN input */
    delta = (rotation->count - rotation->prev_count) & dimension_mask;
    rotation->prev_count = rotation->count;

    if (!rotation->initialized)
    {
        delta = 0;
        rotation->initialized = 1;
    }

    if (state->current_state.moving_type.value == BACKWARD)
    {
        delta = -delta;
    }

    if (abs(rotation->prev_delta - abs(delta)) > delta_limit)
    {
        /* This happens when car acceleration is large enough, and */
        /* also when track gets changed and there is discontinuity in CAN data */

        rotation->wheel_rotation_angle += delta * 360.0 / ratio;
        rotation->prev_filtered_delta = delta;
    }
    else
    {
        /* apply 1st order filter */
        float filtered_delta = rotation->prev_filtered_delta * 0.75f + delta * (1.0f - 0.75f);

        rotation->prev_filtered_delta = filtered_delta;

        rotation->wheel_rotation_angle += filtered_delta * 360.0 / ratio;
    }

    rotation->prev_delta = abs(delta);

    /* make sure the angle is whithin 0..360 degrees */
    rotation->wheel_rotation_angle = fmod(rotation->wheel_rotation_angle, 360.0);
}


int sv_vehicle_state_update(
    engine::VehicleState *state,
    const VehicleDimensions *vehicle_config,
    const VehicleModelConfig *vehicle_model_config)
{
    int i;
    uint64_t current_time = sv_get_timestamp();

    if (state->timestamp_ns > 0)
    {
        if (vehicle_model_config->use_velocity_for_wheels_animation)
        {
            float deltaTime_ms =
                (current_time / 1000000.0f -
                 state->timestamp_ns / 1000000.0f);

            for (i = 0; i < WHEELS_MAX; i++)
            {
                float sign = 1.0f;

                switch (state->current_state.wheel_movement[i].value)
                {
                case FORWARD:
                    sign = 1.0f;
                    break;
                case BACKWARD:
                    sign = -1.0f;
                    break;
                case UNKNOWN:
                case STATIC:
                default:
                    sign = 0.0f;
                    break;
                }

                state->wheel_rotation[i].wheel_rotation_angle =
                    wheel_position(sign * state->current_state.wheel_velocity[i].value,
                                   deltaTime_ms,
                                   state->wheel_rotation[i].wheel_rotation_angle,
                                   vehicle_config->wheel_diameter_mm,
                                   1.0f);
            }
        }
    }

    state->wheel_rotation[WHEEL_FRONT_LEFT].yaw_angle =
        state->current_state.steering_angle.value /
        vehicle_config->steering_ratio;
    state->wheel_rotation[WHEEL_FRONT_RIGHT].yaw_angle =
        state->current_state.steering_angle.value /
        vehicle_config->steering_ratio;

    state->wheel_rotation[WHEEL_REAR_LEFT].yaw_angle = 0.0f;
    state->wheel_rotation[WHEEL_REAR_RIGHT].yaw_angle = 0.0f;

    state->timestamp_ns = current_time;

    return 0;
}

int sv_vehicle_state_handle_signals(
    engine::VehicleState *state,
    const VehicleDimensions *vehicle_config,
    const VehicleModelConfig *vehicle_model_config,
    const CANSignals *vehicle_signals)
{
    int i;

    for (i = 0; i < WHEELS_MAX; i++)
    {
        if (!vehicle_model_config->use_velocity_for_wheels_animation)
        {
            if (state->current_state.wheel_pulse_count[i].timestamp !=
                vehicle_signals->wheel_pulse_count[i].timestamp)
            {
                // todo: move to config coefficients
                update_wheel_rotation_angle(state,
                                            vehicle_signals->wheel_pulse_count[i].value,
                                            i,
                                            120.0,
                                            1,
                                            (uint32_t)(vehicle_config->max_pulse - 1));
            }
        }
    }

    state->current_state = *vehicle_signals;

    return 0;
}

}
