#pragma once

#include <cstdint>

#include <engine/vehicle_data.hpp>

namespace svapp
{

uint64_t monotonic_timestamp_ns();
void update_demo_vehicle_signals(vehicle::CANSignals &signals);

} // namespace svapp
