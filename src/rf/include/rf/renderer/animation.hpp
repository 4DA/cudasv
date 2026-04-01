#ifndef RF_MODEL_ANIMATION
#define RF_MODEL_ANIMATION

#include <vector>
#include <chrono>
#include <string>
#include <unordered_map>

#include <rf/renderer/glm_common.hpp>
#include <rf/renderer/scene.hpp>
#include "mesh_geometry.hpp"

namespace rf
{

namespace animation
{
    // Method for interpolation between keys
    enum class Interpolation
    {
        Linear,
        Step,
        CatmullRomSpline,
        CubicSpline
    };

    // Types of scene components properties to animate
    enum class TargetPath
    {
        Translation,
        Rotation,
        Scale,
        Weights
    };

    // array of floating point of values representing linear time in seconds
    struct KeyFrameInput
    {
        float min;
        float max;

        std::vector<float> times;
    };

    struct Sampler
    {
        // Time frames that animation should enumerate as input
        KeyFrameInput input;

        // Accessor storing set of scalars or vectors, representing animated
        // property
        AttributesAccessor output;

        Interpolation interpolation;
    };

    // Channel connects the output values of the key frame animation to a
    // specific scene component in the hierarchy.
    struct Channel
    {
        // Animation sampler used to obtain key frame information
        Sampler *sampler;

        // scene component name to animate
        std::string targetName;

        // which property of scele component to animate
        TargetPath targetPath;

        // get interpolated rotation, corresponding to time
        glm::quat getRotation(float time);

        // get interpolated translation, corresponding to time
        glm::vec3 get_translation(float time);
    };

    // struct that stores data for animation
    struct NodeAnimation
    {
        std::string name;
        std::vector<std::unique_ptr<Channel>> channels;
        std::vector<std::unique_ptr<Sampler>> samplers;
    };

    // struct that stores current state for animation
    struct AnimationState
    {
        using Timestamp = std::chrono::time_point<std::chrono::high_resolution_clock, std::chrono::nanoseconds>;

        float get_time();
        void update_transforms();

        AnimationState(Scene *scene,
                        const NodeAnimation &model_animation,
                        bool play_forward);

    private:
        bool play_forward;
        Timestamp start_timestamp;
        bool finished;
        std::string name;
        using CompoChannel = std::pair<SceneComponent *, Channel *>;
        std::vector<CompoChannel> channels;
    };
};

using AnimationMap = std::unordered_map<std::string, animation::NodeAnimation>;

} // namespace rf

#endif
