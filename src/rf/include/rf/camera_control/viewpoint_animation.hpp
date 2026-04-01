#ifndef RF_CAMERA_VIEWPOINT_ANIMATION_HPP
#define RF_CAMERA_VIEWPOINT_ANIMATION_HPP

#include <chrono>
#include <functional>
#include <memory>
#include <variant>

#include <rf/renderer/trs_transform.hpp>
#include <rf/renderer/glm_common.hpp>
#include <rf/renderer/virtual_camera.hpp>
#include "rotator.hpp"

namespace rf
{
    class ViewpointAnimator;

// Camera animation mode
enum class CameraMovementMode {
    SPHERICAL, // Camera traverses along an ellipsoidal arc
    FREE       // Camera's position and look-at point can be modified independently
};

// Base class for camera animation
struct CameraAnimationBase
{
    enum class Easing {
        Linear, // Movement at a constant speed
        Cubic,  // Cubic easing equation: 3x^2 - 2x^3
        Quintic // Improved Perlin version, has zero first- and second-order derivatives at endpoints
    };

    // Future object that can be utilized to trigger an action upon the completion of the animation
    struct FinishCallback {
        friend ViewpointAnimator;

        FinishCallback(ViewpointAnimator *animator):
            _animator(animator),
            _next(nullptr) {}

        // schedule a callback to be run after animation is completed
        void then(std::function<void()> fn) {
            this->_fn = fn;
        };

        operator bool() const {
            return static_cast<bool>(_fn);
        }

        void invoke() {
            this->_fn();
            this->_fn = std::function<void()>();
        }

    private:
        ViewpointAnimator *_animator;
        FinishCallback *_next;
        std::function<void()> _fn;
    };

    CameraAnimationBase(std::chrono::milliseconds duration,
                        std::chrono::milliseconds start,
                        FinishCallback &completion,
                        float t = 0.0f,
                        Easing easing = Easing::Quintic):
        _duration(duration),
        _start(start),
        _completion(completion),
        _t(t),
        _easing(easing) {}

    // apply easing function to normalized time t
    virtual float ease(float t) final;

    // Check animation status (finished or not, time left in ms)
    virtual bool get_completion_status(unsigned int &timeleft_ms);

    // Adjust the camera position based on the current animation progress
    virtual float update(VirtualCamera &camera,
                         EllipticRotator &rotator,
                         SphericalCoord &camera_pos) = 0;

    // returns true if the animation has finished
    virtual bool is_finished(VirtualCamera &camera) = 0;

    virtual ~CameraAnimationBase() = default;

    std::chrono::milliseconds _duration;
    std::chrono::milliseconds _start;
    CameraAnimationBase::FinishCallback &_completion;
    float _t;
    Easing _easing;
};

struct Viewpoint
{
    struct Elliptic {
        SphericalCoord coord;
        EllipticRotator rotator;
    };

    struct LookAt {
        glm::vec3 origin;
        glm::vec3 lookAt;
        glm::vec3 up;
    };

    bool is_elliptic() const {return std::holds_alternative<Elliptic>(_var);}

    const Elliptic & get_elliptic() const {return std::get<Elliptic>(_var);}

    const LookAt & get_look_at() const {return std::get<LookAt>(_var);}

    Viewpoint(SphericalCoord coord,
              EllipticRotator rotator,
              Projection proj):
        _var(Elliptic{coord, rotator}),
        _projection(proj) {}

    Viewpoint(glm::vec3 origin,
              glm::vec3 lookAt,
              glm::vec3 up,
              Projection proj):
        _var(LookAt{origin, lookAt, up}),
        _projection(proj) {}

    const Projection & get_projection() const { return _projection; }

private:
    std::variant<LookAt, Elliptic> _var;
    Projection _projection;
};

// Camera animation that changes camera position in {position,lookat} parameter space
struct LookatCameraAnimation: public CameraAnimationBase {
    LookatCameraAnimation(const Viewpoint::LookAt &target_vp,
                          const VirtualCamera &camera,
                          Projection to_projection,
                          unsigned int duration,
                          CameraAnimationBase::FinishCallback &completion):

        CameraAnimationBase(std::chrono::milliseconds(duration),
                            std::chrono::duration_cast< std::chrono::milliseconds >
                            (std::chrono::system_clock::now().time_since_epoch()),
                            completion),
        _from(camera.transform),
        _to(get_look_at_trs(target_vp.origin, target_vp.lookAt, target_vp.up)),
        _from_projection(camera.m_projection),
        _to_projection(to_projection) {}

    // update camera position according to current running animation
    virtual float update(VirtualCamera &camera,
                         EllipticRotator &rotator,
                         SphericalCoord &camera_pos) override;

    // @returns true if animation is finished
    virtual bool is_finished(VirtualCamera &camera) override;

private:
    TRSTransform _from;
    TRSTransform _to;
    Projection _from_projection;
    Projection _to_projection;
};

//Camera animation that changes camera position in ellipsoid parameter space
struct EllipticCameraAnimation: public CameraAnimationBase {
    // Adjust the camera's position based on the ongoing animation
    // completion - Future used to schedule an action upon
    // the animation's completion
    EllipticCameraAnimation(SphericalCoord from,
                            SphericalCoord to,
                            EllipticRotator rotatorFrom,
                            EllipticRotator rotatorTo,
                            const VirtualCamera &camera,
                            Projection to_projection,
                            unsigned int duration,
                            CameraAnimationBase::FinishCallback &completion):

        CameraAnimationBase(std::chrono::milliseconds(duration),
                            std::chrono::duration_cast< std::chrono::milliseconds >
                            (std::chrono::system_clock::now().time_since_epoch()),
                            completion),
        _from(from),
        _to(to),
        _rotatorFrom(rotatorFrom),
        _rotatorTo(rotatorTo),
        _from_projection(camera.m_projection),
        _to_projection(to_projection) {}

    // Update the specified virtual camera's position as per the active animation
    // camera - Virtual camera that will be modified
    // rotator - Rotator defining the camera's spatial position
    // camera_pos - Spherical coordinates for the virtual camera
    virtual float update(VirtualCamera &camera,
                         EllipticRotator &,
                         SphericalCoord &camera_pos) override;

    // @returns true if animation is finished
    virtual bool is_finished(VirtualCamera &camera) override;

private:
    SphericalCoord _from;
    SphericalCoord _to;
    EllipticRotator _rotatorFrom;
    EllipticRotator _rotatorTo;
    Projection _from_projection;
    Projection _to_projection;
};

// Class that manages transition between viewpoints
struct ViewpointAnimator
{
    ViewpointAnimator(unsigned int duration = 1000):
        _duration(duration), _finishCB(this) {}

    // Start camera movement towards the specified viewpoint
    CameraAnimationBase::FinishCallback & set_viewpoint(VirtualCamera &camera,
                                                        const SphericalCoord &coord,
                                                        const EllipticRotator &rotator,
                                                        const Viewpoint &v);

    float update(VirtualCamera &camera,
                 SphericalCoord &coord,
                 EllipticRotator &rotator);

    // return status of the animation (whether it has finished or not, time remaining in ms)
    // Parameter - remaining time for animation execution
    // Return true if the animation is currently in progress
    bool is_finished(unsigned int &timeleft_ms);

private:
    CameraMovementMode _camera_mode = CameraMovementMode::SPHERICAL;
    std::unique_ptr<CameraAnimationBase> _anim;
    unsigned int _duration;
    CameraAnimationBase::FinishCallback _finishCB;
};

}

#endif
