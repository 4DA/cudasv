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

static rf::animation::KeyFrameInput
get_key_frame_input(const tinygltf::Model &model, int accessorID)
{
    rf::animation::KeyFrameInput result;
    rf::AttributesAccessor accessor;
    const auto &gltfAccessor = model.accessors[accessorID];
    loader::gltf::init_attributes_accessor(model, accessor, "KeyFrameTimes", gltfAccessor);

    for (size_t i = 0; i < accessor.count; i++) {
        SPDLOG_DEBUG("{}", fmt::sprintf("accessor %d key frame time: %f",
                                        accessorID, get_attribute_float(&accessor, i)));
        result.times.push_back(get_attribute_float(&accessor, i));
    }

    result.min = static_cast<float>(gltfAccessor.minValues[0]);
    result.max = static_cast<float>(gltfAccessor.maxValues[0]);

    return result;
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

    sampler->input = get_key_frame_input(model, gltfSampler.input);

    loader::gltf::init_attributes_accessor(model,
                                           sampler->output,
                                           "KeyFrameValues",
                                           model.accessors[gltfSampler.output]);

    return std::unique_ptr<rf::animation::Sampler>(sampler);
}

static std::unique_ptr<rf::animation::Channel>
create_animation_channel(const tinygltf::Model &model,
                         rf::animation::NodeAnimation &animation,
                         const tinygltf::AnimationChannel &gltfChannel,
                         const std::string &prefix)
{
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
        auto &animation = result[prefix + gltfAnimation.name];
        animation.name = prefix + gltfAnimation.name;

        for (const tinygltf::AnimationSampler &gltfSampler: gltfAnimation.samplers) {
            animation.samplers.push_back(create_animation_sampler(model, gltfSampler));
        }

        for (const auto &gltfChannel: gltfAnimation.channels) {
            std::unique_ptr<rf::animation::Channel> channel =
                create_animation_channel(model, animation, gltfChannel, prefix);

            SPDLOG_DEBUG("{}", fmt::sprintf("Loaded animation [name=%s] [target=%s], [path=%s]",
                                            animation.name.c_str(),
                                            channel ? channel->target_name.c_str() : "",
                                            gltfChannel.target_path.c_str()));
            animation.channels.push_back(std::move(channel));
        }
    }

    return result;
}

} // namespace loader::gltf
