#include "renderer/gltf/animation_loader.hpp"

#include <memory>
#include <string>
#include <unordered_map>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/printf.h>

#include "renderer/gltf/accessors.hpp"

namespace
{

const std::string ANIMATION_TARGET_TRANSLATION = "translation";
const std::string ANIMATION_TARGET_ROTATION = "rotation";
const std::string ANIMATION_TARGET_SCALE = "scale";
const std::string ANIMATION_TARGET_WEIGHTS = "weights";

const std::string INTERPOLATION_LINEAR = "LINEAR";
const std::string INTERPOLATION_STEP = "STEP";
const std::string INTERPOLATION_CATMULLROMSPLINE = "CATMULLROMSPLINE";
const std::string INTERPOLATION_CUBICSPLINE = "CUBICSPLINE";

static bool
get_key_frame_input(const tinygltf::Model &model,
                    int accessorID,
                    rf::animation::KeyFrameInput &result)
{
    if (accessorID < 0 || accessorID >= static_cast<int>(model.accessors.size())) {
        SPDLOG_ERROR("Animation input references invalid accessor {}", accessorID);
        return false;
    }

    rf::AttributesAccessor accessor;
    const auto &gltfAccessor = model.accessors[accessorID];
    if (!loader::gltf::init_attributes_accessor(model, accessor, "KeyFrameTimes", gltfAccessor)) {
        return false;
    }

    for (size_t i = 0; i < accessor.count; i++) {
        SPDLOG_DEBUG("{}", fmt::sprintf("accessor %d key frame time: %f",
                                        accessorID, get_attribute_float(&accessor, i)));
        result.times.push_back(get_attribute_float(&accessor, i));
    }

    if (gltfAccessor.minValues.empty() || gltfAccessor.maxValues.empty()) {
        SPDLOG_ERROR("Animation input accessor {} is missing min/max values", accessorID);
        return false;
    }

    result.min = static_cast<float>(gltfAccessor.minValues[0]);
    result.max = static_cast<float>(gltfAccessor.maxValues[0]);

    return true;
}

static std::unique_ptr<rf::animation::Sampler>
create_animation_sampler(const tinygltf::Model &model,
                         const tinygltf::AnimationSampler &gltfSampler)
{
    auto sampler = new rf::animation::Sampler();

    if (gltfSampler.interpolation == INTERPOLATION_LINEAR) {
        sampler->interpolation = rf::animation::Interpolation::Linear;
    } else if (gltfSampler.interpolation == INTERPOLATION_STEP) {
        sampler->interpolation = rf::animation::Interpolation::Step;
    } else if (gltfSampler.interpolation == INTERPOLATION_CATMULLROMSPLINE) {
        sampler->interpolation = rf::animation::Interpolation::CatmullRomSpline;
    } else if (gltfSampler.interpolation == INTERPOLATION_CUBICSPLINE) {
        sampler->interpolation = rf::animation::Interpolation::CubicSpline;
    } else {
        SPDLOG_ERROR("{}", fmt::sprintf("Unknown interpolation type: %s", gltfSampler.interpolation.c_str()));
        return nullptr;
    }

    if (!get_key_frame_input(model, gltfSampler.input, sampler->input)) {
        return nullptr;
    }

    if (gltfSampler.output < 0 || gltfSampler.output >= static_cast<int>(model.accessors.size())) {
        SPDLOG_ERROR("Animation output references invalid accessor {}", gltfSampler.output);
        return nullptr;
    }

    if (!loader::gltf::init_attributes_accessor(model,
                                                sampler->output,
                                                "KeyFrameValues",
                                                model.accessors[gltfSampler.output])) {
        return nullptr;
    }

    return std::unique_ptr<rf::animation::Sampler>(sampler);
}

static std::unique_ptr<rf::animation::Channel>
create_animation_channel(const tinygltf::Model &model,
                         rf::animation::NodeAnimation &animation,
                         const tinygltf::AnimationChannel &gltfChannel,
                         const std::string &prefix)
{
    if (gltfChannel.sampler < 0 ||
        gltfChannel.sampler >= static_cast<int>(animation.samplers.size()) ||
        animation.samplers[gltfChannel.sampler] == nullptr) {
        SPDLOG_ERROR("Animation channel references invalid sampler {}", gltfChannel.sampler);
        return nullptr;
    }

    auto samplerRawPtr = animation.samplers[gltfChannel.sampler].get();
    std::string targetName;
    rf::animation::TargetPath targetPath;

    if (gltfChannel.target_node >= 0) {
        targetName = prefix + model.nodes[gltfChannel.target_node].name;
    }

    if (gltfChannel.target_path == ANIMATION_TARGET_TRANSLATION) {
        targetPath = rf::animation::TargetPath::Translation;
    } else if (gltfChannel.target_path == ANIMATION_TARGET_ROTATION) {
        targetPath = rf::animation::TargetPath::Rotation;
    } else if (gltfChannel.target_path == ANIMATION_TARGET_SCALE) {
        targetPath = rf::animation::TargetPath::Scale;
    } else if (gltfChannel.target_path == ANIMATION_TARGET_WEIGHTS) {
        targetPath = rf::animation::TargetPath::Weights;
    } else {
        SPDLOG_ERROR("{}", fmt::sprintf("Unknown target path: %s", gltfChannel.target_path.c_str()));
        return nullptr;
    }

    return std::unique_ptr<rf::animation::Channel>(
        new rf::animation::Channel{samplerRawPtr, targetName, targetPath});
}

} // namespace

namespace loader::gltf
{

rf::AnimationMap load_animations(const tinygltf::Model &model, const std::string &prefix)
{
    std::unordered_map<std::string, rf::animation::NodeAnimation> result;

    for (const auto &gltfAnimation: model.animations) {
        rf::animation::NodeAnimation animation;
        animation.name = prefix + gltfAnimation.name;
        bool animationValid = true;

        for (const tinygltf::AnimationSampler &gltfSampler: gltfAnimation.samplers) {
            auto sampler = create_animation_sampler(model, gltfSampler);
            if (!sampler) {
                SPDLOG_ERROR("Failed to load sampler for animation '{}'", animation.name);
                animationValid = false;
                break;
            }
            animation.samplers.push_back(std::move(sampler));
        }

        if (!animationValid) {
            continue;
        }

        for (const auto &gltfChannel: gltfAnimation.channels) {
            std::unique_ptr<rf::animation::Channel> channel =
                create_animation_channel(model, animation, gltfChannel, prefix);

            if (!channel) {
                SPDLOG_ERROR("Failed to load channel for animation '{}'", animation.name);
                animationValid = false;
                break;
            }

            SPDLOG_DEBUG("{}", fmt::sprintf("Loaded animation [name=%s] [target=%s], [path=%s]",
                                            animation.name.c_str(),
                                            channel->target_name.c_str(),
                                            gltfChannel.target_path.c_str()));
            animation.channels.push_back(std::move(channel));
        }

        if (!animationValid) {
            continue;
        }

        result.emplace(animation.name, std::move(animation));
    }

    return result;
}

} // namespace loader::gltf
