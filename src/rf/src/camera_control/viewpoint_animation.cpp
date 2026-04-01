#include <chrono>
#include <memory>
#include <functional>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/printf.h>

#include "viewpoint_animation.hpp"
#include "rotator.hpp"

using namespace std::chrono;

inline static float quintic(float A, float B, float t) {
    t = glm::mix(A, B, t);
    return t * t * t * (t * (t * 6 - 15) + 10);
}

float rf::CameraAnimationBase::ease(float t)
{
    t = glm::clamp(t, 0.0f, 1.0f);

    switch (CameraAnimationBase::_easing) {
    case CameraAnimationBase::Easing::Linear:
        return t;
    case CameraAnimationBase::Easing::Cubic:
        return glm::smoothstep(0.0f, 1.0f, t);
    case CameraAnimationBase::Easing::Quintic:
        return quintic(0.0f, 1.0f, t);
    default:
        SPDLOG_ERROR("{}", fmt::sprintf("unkown easing function"));
        return 1.0f;
    }
}

bool rf::CameraAnimationBase::get_completion_status(unsigned &timeleft_ms)
{
    milliseconds now = duration_cast< milliseconds >(
        system_clock::now().time_since_epoch());

    if (now > _start + _duration) {
        timeleft_ms = 0;
        return false;
    }

    timeleft_ms = (_start + _duration - now).count();

    return true;
}

float rf::LookatCameraAnimation::update(VirtualCamera &camera,
                                        EllipticRotator &rotator,
                                        SphericalCoord &coord)
{
    milliseconds now = duration_cast< milliseconds >(system_clock::now().time_since_epoch());
    _t = static_cast<float>((now - _start).count()) / static_cast<float>(_duration.count());
    _t = ease(_t);

    camera.transform.translation = glm::mix(_from.translation, _to.translation, _t);
    camera.transform.rotation = glm::slerp(_from.rotation, _to.rotation, _t);

    camera.set_projection(Projection::mix(_from_projection, _to_projection, _t));

    SPDLOG_TRACE("{}", fmt::sprintf("camera pos = %f, %f, %f sph:(%f, %f)",
                 camera.transform.translation.x,
                 camera.transform.translation.y,
                 camera.transform.translation.z,
                 coord.polar, coord.azimuthal));
    SPDLOG_TRACE("{}", fmt::sprintf("t = %lf", t));

    if (is_finished(camera)) {
        coord = rotator.get_position(camera);
        SPDLOG_INFO("{}", fmt::sprintf("coord from xyz = <%f, %f>", coord.polar, coord.azimuthal));
    }

    return _t;
}

bool rf::LookatCameraAnimation::is_finished(VirtualCamera &camera)
{
    milliseconds now = duration_cast< milliseconds >(system_clock::now().time_since_epoch());

    if (now > _start + _duration) {
        camera.set_projection(_to_projection);

        if (_completion) {
            _completion.invoke();
        }
        return true;
    }

    return false;
}

float rf::EllipticCameraAnimation::update(VirtualCamera &camera,
                                          EllipticRotator &rotator,
                                          SphericalCoord &coord)
{
    milliseconds now = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
    _t = static_cast<float>((now - _start).count()) / static_cast<float>(_duration.count());
    _t = ease(_t);

    coord = SphericalCoord::interpolate(_from, _to, _t);

    rotator = EllipticRotator::mix(_rotatorFrom, _rotatorTo, _t);
    camera.transform = rotator.get_orientation(coord);
    camera.set_projection(Projection::mix(_from_projection, _to_projection, _t));

    return _t;
}

bool rf::EllipticCameraAnimation::is_finished(VirtualCamera &camera)
{
    milliseconds now = duration_cast< milliseconds >(system_clock::now().time_since_epoch());

    if (now > _start + _duration) {
        camera.set_projection(_to_projection);

        if (_completion) {
            _completion.invoke();
        }

        return true;
    }

    return false;
}

bool rf::ViewpointAnimator::is_finished(unsigned int &timeleft_ms)
{
    if (_anim) {
        return _anim->get_completion_status(timeleft_ms);
    }

    timeleft_ms = 0;
    return false;
}

rf::CameraAnimationBase::FinishCallback &
rf::ViewpointAnimator::set_viewpoint(VirtualCamera &camera,
                                     const SphericalCoord &coord,
                                     const EllipticRotator &rotator,
                                     const Viewpoint &v)
{
    CameraMovementMode desiredMode = v.is_elliptic() == true ?
        CameraMovementMode::SPHERICAL : CameraMovementMode::FREE;

    const Projection &new_projection = v.get_projection();

    if (desiredMode == CameraMovementMode::SPHERICAL) {
        if (_camera_mode == CameraMovementMode::SPHERICAL) {

            assert(v.get_elliptic().rotator.A != 0.0f);

            _anim.reset(new EllipticCameraAnimation(
                            coord, v.get_elliptic().coord,
                            rotator, v.get_elliptic().rotator,
                            camera, new_projection,
                            _duration, _finishCB));
        }
        else {
            TRSTransform xf = rotator.get_orientation(v.get_elliptic().coord);
            Viewpoint new_vp(xf.translation, rotator.center,
                             VirtualCamera::get_up_vector(xf.rotation),
                             v.get_projection());
            _anim.reset(new LookatCameraAnimation(
                new_vp.get_look_at(), camera, new_projection, _duration, _finishCB));
        }
    }
    else {
        _anim.reset(new LookatCameraAnimation(
            v.get_look_at(), camera, new_projection, _duration, _finishCB));
    }

    _camera_mode = desiredMode;

    return _finishCB;
}

float rf::ViewpointAnimator::update(VirtualCamera &camera,
                                    SphericalCoord &coord,
                                    EllipticRotator &rotator)
{
    float ret = 0.0f;

    if (_anim) {
        ret = _anim->update(camera, rotator, coord);

        if (_anim->is_finished(camera)) {
            _anim.reset();
            SPDLOG_INFO("{}", fmt::sprintf("stopped camera animation"));
        }
    }

    return ret;
}
