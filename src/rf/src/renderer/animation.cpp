#include <spdlog/spdlog.h>

#include "animation.hpp"

using namespace rf;
using namespace animation;

static glm::quat to_glm_quat(const glm::vec4 &in_vec)
{
    //gltf specifies quaternion components are stored in XYZW order
    //glm quat ctor acceps WXYZ
    return glm::quat(in_vec.w, in_vec.x, in_vec.y, in_vec.z);
}

glm::vec3 Channel::get_translation(float time)
{
    if (time < sampler->input.min) {
        return get_attribute3(&sampler->output, 0);
    } else if (time > sampler->input.max) {
        return get_attribute3(&sampler->output, sampler->input.times.size() - 1);
    }

    int keyframeFrom  = -1;
    int keyframeTo  = -1;

    for (size_t i = 0; i < sampler->input.times.size() - 1; i++) {
        if (time >= sampler->input.times[i] && time <= sampler->input.times[i + 1]) {
            keyframeFrom = i;
            keyframeTo = i + 1;
        }
    }

    assert(keyframeFrom >= 0);
    assert(keyframeTo >= 0);

    float t1 = sampler->input.times[keyframeFrom];
    float t2 = sampler->input.times[keyframeTo];

    float interpolation = (time - t1) / (t2 - t1);

    glm::vec3 value1 = get_attribute3(&sampler->output, keyframeFrom);
    glm::vec3 value2 = get_attribute3(&sampler->output, keyframeTo);

    return glm::mix(value1, value2, interpolation);
}

glm::quat Channel::getRotation(float time)
{
    if (time < sampler->input.min) {
        glm::vec4 a = get_attribute4(&sampler->output, 0);
        glm::quat q = to_glm_quat(a);
        return q;
    } else if (time > sampler->input.max) {
        glm::vec4 a = get_attribute4(&sampler->output, sampler->input.times.size() - 1);
        glm::quat q = to_glm_quat(a);
        return q;
    }

    int keyframeFrom  = -1;
    int keyframeTo  = -1;

    for (size_t i = 0; i < sampler->input.times.size() - 1; i++) {
        if (time >= sampler->input.times[i] && time <= sampler->input.times[i + 1]) {
            keyframeFrom = i;
            keyframeTo = i + 1;
        }
    }

    assert(keyframeFrom >= 0);
    assert(keyframeTo >= 0);

    float t1 = sampler->input.times[keyframeFrom];
    float t2 = sampler->input.times[keyframeTo];

    float interpolation = (time - t1) / (t2 - t1);

    glm::vec4 value1 = get_attribute4(&sampler->output, keyframeFrom);
    glm::vec4 value2 = get_attribute4(&sampler->output, keyframeTo);

    glm::quat from;
    glm::quat to;

    from = to_glm_quat(value1);
    to = to_glm_quat(value2);

    return glm::mix(from, to, interpolation);
}

float AnimationState::get_time()
{
    auto now = std::chrono::high_resolution_clock::now();
    float elapsed = std::chrono::duration_cast<std::chrono::microseconds>
        (now - start_timestamp).count() * 1e-6f;

    return elapsed;
}

void AnimationState::update_transforms()
{
    float elapsedTime = get_time();

    bool allChannelsFinished = true;

    for (auto channelIt: channels) {
        auto target = channelIt.first;
        auto channel = channelIt.second;

        float animationTime = elapsedTime;

        if (!play_forward) {
            animationTime = channel->sampler->input.max - elapsedTime;
        }

        switch (channel->targetPath) {
        case TargetPath::Rotation:
        {
            target->toLocal.rotation = channel->getRotation(animationTime);

            SPDLOG_TRACE("anim [name={}, target={}] (t = {:f}) rotation = <{:f}, {:f}, {:f}, {:f}>", name.c_str(), target->name.c_str(), animationTime, target->toLocal.rotation.x, target->toLocal.rotation.y, target->toLocal.rotation.z, target->toLocal.rotation.w);

            break;
        }

        case TargetPath::Translation:
        {
            target->toLocal.translation = channel->get_translation(animationTime);

            SPDLOG_TRACE("anim [name={}, target={}] (t = {:f}) translation = <{:f}, {:f}, {:f}>", name.c_str(), target->name.c_str(), animationTime, target->toLocal.translation.x, target->toLocal.translation.y, target->toLocal.translation.z);

            break;
        }

        case TargetPath::Scale:
        {
            target->toLocal.scale = channel->get_translation(animationTime);

            SPDLOG_TRACE("anim [name={}] (t = {:f}) scale = <{:f}, {:f}, {:f}>", name.c_str(), animationTime, target->toLocal.scale.x, target->toLocal.scale.y, target->toLocal.scale.z);

            break;
        }

        default:
            SPDLOG_ERROR("Unsupported TargetPath ({})", static_cast<int>(channel->targetPath));
            assert(false);
        }

        if (elapsedTime < channel->sampler->input.max) {
             allChannelsFinished = false;
        }
    }

    if (allChannelsFinished) {
        finished = true;
        SPDLOG_INFO("animation [name={}] is over", name.c_str());
    }
}

AnimationState::AnimationState(Scene *scene,
                               const NodeAnimation &modelAnimation,
                               bool play_forward):
    play_forward(play_forward),
    start_timestamp(std::chrono::high_resolution_clock::now()),
    finished(false),
    name(modelAnimation.name)
{
    for (auto &channel: modelAnimation.channels) {
        auto target = scene->get(channel->targetName);
        channels.push_back(CompoChannel(target, channel.get()));
    }
}
