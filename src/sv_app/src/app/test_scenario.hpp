#pragma once

#include <string>

#include <engine/overlays_config.hpp>

namespace svapp
{

bool load_test_scenario_config(const std::string &path, TestScenarioConfig &config);

} // namespace svapp
