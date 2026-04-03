#pragma once

#include <engine/vehicle_data.hpp>

namespace svapp
{

struct VehicleSignalProvider
{
    virtual ~VehicleSignalProvider() = default;

    virtual bool get_next_signals(vehicle::CANSignals &signals) = 0;
};

} // namespace svapp
