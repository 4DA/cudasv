#ifndef VEHICLEDATA_HPP
#define VEHICLEDATA_HPP

#include <cstdint>
#include <cstdbool>

#include <engine/vehicle_data.hpp>

namespace vehicle
{

enum WheelID
{
    WHEEL_FRONT_LEFT  = 0,
    WHEEL_FRONT_RIGHT = 1,
    WHEEL_REAR_LEFT   = 2,
    WHEEL_REAR_RIGHT  = 3,
    WHEELS_MAX  = 4
};

enum GearID
{
    GEAR_UNKNOWN = 0,
    GEAR_NEUTRAL = 1,
    GEAR_PARK    = 2,
    GEAR_REVERSE = 3,
    GEAR_DRIVE   = 4,
    GEARS_MAX
};

enum MovementDirection
{
    UNKNOWN = 0,
    FORWARD  = 1,
    BACKWARD = 2,
    STATIC  = 3
};

typedef uint64_t TimestampNS;

template <typename T>
struct CANSignal
{
    TimestampNS timestamp;
    T value;
};

struct CANSignals
{
    CANSignal<float>         yaw_rate;
    CANSignal<float>         long_accel;
    CANSignal<float>         lateral_accel;
    CANSignal<float>         speed_rear_axis;
    CANSignal<float>         steering_angle;
    CANSignal<uint32_t>      moving_type;
    CANSignal<uint32_t>      wheel_pulse_count[WHEELS_MAX];
    CANSignal<float>         wheel_velocity[WHEELS_MAX];
    CANSignal<uint32_t>      wheel_movement[WHEELS_MAX];
    CANSignal<float>         accel[3];
    CANSignal<float>         gyro[3];
    CANSignal<uint32_t>      gear;
    TimestampNS              timestamp;
};

struct VehicleDimensions
{
    float length; // in mm

    float width; // in mm

    float width_with_mirrors; // in mm

    // Path length per one pulse [if pulses are available] (mm)
    float    pulse_size_mm;
    float    pulse_size_mms[WHEELS_MAX];

    float track_front; // Front track width (mm)

    float    track_rear; // Rear track width (mm)

    float wheel_base; // in mm

     // steering angle -> wheel angle ratio
    float    steering_ratio;

    float    pulses_per_meter;

    // Rear wheel axis center in the vehicle coordinate system
    float    rear_axis_pose_xy[2];

    // 0 if not available
    uint32_t max_pulse;

    int wheels_velocity_available;

    float    tow_hitch_position[3];

    int front_wheel_drive;

    float wheel_diameter_mm;

    float tire_width_mm;
};

} // namespace vehicle

#endif
