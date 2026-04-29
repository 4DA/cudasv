#include <filesystem>
#include <fstream>

#include <spdlog/spdlog.h>

#include <nlohmann/json.hpp>

#include "app/test_scenario.hpp"

namespace svapp
{

namespace
{

using nlohmann::json;

bool read_float3(const json &node, const char *label, float out[3])
{
    if (!node.is_array() || node.size() != 3) {
        SPDLOG_ERROR("{} must be an array of 3 numbers", label);
        return false;
    }

    for (int i = 0; i < 3; ++i) {
        out[i] = node.at(i).get<float>();
    }

    return true;
}

std::filesystem::path resolve_relative_path(const std::filesystem::path &baseFile,
                                            const std::string &candidate)
{
    const std::filesystem::path candidatePath(candidate);
    if (candidatePath.is_absolute()) {
        return candidatePath;
    }

    return baseFile.parent_path() / candidatePath;
}

} // namespace

bool load_test_scenario_config(const std::string &path, TestScenarioConfig &config)
{
    std::ifstream input(path);
    if (!input) {
        SPDLOG_ERROR("Failed to open test scenario '{}'", path);
        return false;
    }

    json scenarioJson;
    try {
        input >> scenarioJson;
    } catch (const std::exception &error) {
        SPDLOG_ERROR("Failed to parse test scenario '{}': {}", path, error.what());
        return false;
    }

    if (!scenarioJson.is_object()) {
        SPDLOG_ERROR("Test scenario '{}' must contain a JSON object", path);
        return false;
    }

    TestScenarioConfig parsed = {};
    parsed.enabled = 1;
    parsed.name = scenarioJson.value("name", scenarioJson.value("test_name", std::string("test_scenario")));

    const std::string glbPath =
        scenarioJson.value("glb_path", scenarioJson.value("model_path", std::string()));
    if (glbPath.empty()) {
        SPDLOG_ERROR("Test scenario '{}' must contain 'glb_path'", path);
        return false;
    }

    parsed.glb_path = resolve_relative_path(std::filesystem::path(path), glbPath).lexically_normal().string();

    const json *transformNode = nullptr;
    if (scenarioJson.contains("transform")) {
        transformNode = &scenarioJson.at("transform");
        if (!transformNode->is_object()) {
            SPDLOG_ERROR("Test scenario '{}' field 'transform' must be an object", path);
            return false;
        }
    }

    const json &root = transformNode ? *transformNode : scenarioJson;

    if (root.contains("position") && !read_float3(root.at("position"), "position", parsed.position)) {
        return false;
    }

    if (root.contains("rotation") && !read_float3(root.at("rotation"), "rotation", parsed.rotation)) {
        return false;
    }

    if (root.contains("scale") && !read_float3(root.at("scale"), "scale", parsed.scale)) {
        return false;
    }

    config = parsed;

    SPDLOG_INFO("Loaded test scenario '{}' from '{}'", config.name, path);
    SPDLOG_INFO("Test scenario model path: {}", config.glb_path);

    return true;
}

} // namespace svapp
