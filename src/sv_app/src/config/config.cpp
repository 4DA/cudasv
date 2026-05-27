#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/printf.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <fstream>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

#include <engine/engine.hpp>

#include "config.hpp"

using engine::Config;

struct ConfigFiles
{
    std::string views_config_path;
    std::string overlays_config_path;
    std::string vehicle_config_path;
};

ConfigFiles svConfiguration = {
    .views_config_path = "config/views.json",
    .overlays_config_path = "config/overlays.json",
    .vehicle_config_path = "config/vehicle.json"
};

int load_config(Config *config)
{
    if (loadJsonConfig<vehicle::VehicleDimensions>(
            config->vehicle_config,
            svConfiguration.vehicle_config_path))
    {
        std::cout << "Failed to load vehicle config file \n";
        return -1;
    }

    if (loadJsonConfig<engine::view::ViewsConfig>(
            config->views_config,
            svConfiguration.views_config_path))
    {
        std::cout << "Failed to load views config file \n";
        return -1;
    }


    if (loadJsonConfig<OverlaysConfig>(
            config->overlays_config,
            svConfiguration.overlays_config_path))
    {
        std::cout << "Failed to load overlays config file \n";
        return -1;
    }

    return 0;
}
