#pragma once

#include "app/vehicle_signal_provider.hpp"

namespace svapp
{

class DemoVehicleSignalProvider final : public VehicleSignalProvider
{
public:
    bool get_next_signals(vehicle::CANSignals &signals) override;
};

} // namespace svapp
