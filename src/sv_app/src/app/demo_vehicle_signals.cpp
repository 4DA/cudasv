#include "app/demo_vehicle_signals.hpp"

#include <cmath>
#include <ctime>
#include <cstring>

namespace svapp
{

uint64_t monotonic_timestamp_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

void update_demo_vehicle_signals(vehicle::CANSignals &signals)
{
    std::memset(&signals, 0, sizeof(signals));

    const double time_seconds = static_cast<double>(monotonic_timestamp_ns()) / 1000000000.0;
    const uint64_t timestamp = monotonic_timestamp_ns();

    signals.steering_angle.value = 40.0f * sin(0.5f * static_cast<float>(time_seconds));
    signals.moving_type.value = vehicle::BACKWARD;
    signals.steering_angle.timestamp = timestamp;
    signals.timestamp = timestamp;

    for (int wheel = 0; wheel < 4; ++wheel) {
        signals.wheel_velocity[wheel].value = 10.0f;
        signals.wheel_velocity[wheel].timestamp = timestamp;
        signals.wheel_movement[wheel].value = vehicle::BACKWARD;
    }
}

} // namespace svapp
