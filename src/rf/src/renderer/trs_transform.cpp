#include <spdlog/spdlog.h>

#include "trs_transform.hpp"


using namespace rf;

namespace rf
{

void check_rotation_unity(const glm::quat rotation)
{
    if (glm::length(rotation) > 1.0f + 2.0f * std::numeric_limits<float>::epsilon()) {
        SPDLOG_ERROR("not unity rotation quat: <{:f}, {:f}, {:f}, {:f}>, norm = {:f}\n", rotation.w, rotation.x, rotation.y, rotation.z, glm::length(rotation));
    }
}

static bool has_negative_scale(const glm::vec3 &scale)
{
    return scale.x < 0.f || scale.y < 0.f || scale.z < 0.f;
}

}

TRSTransform rf::compose(const TRSTransform &a, const TRSTransform &b)
{
    // negative scale is not supported
    if (has_negative_scale(a.scale) ||
        has_negative_scale(b.scale)) {
        // not supported
        // TODO remove this after model is stabilized
        // assert(false);
    }

    return TRSTransform(b.rotation * (b.scale * a.translation) + b.translation,
                     b.rotation * a.rotation,
                     a.scale * b.scale);
}

glm::vec4 TRSTransform::apply(const glm::vec4 &src) const
{
    // not implemented for other cases
    assert(src.w == 0.f || src.w == 1.f);

	//Transform with:
	//QST(x) = R * S * x * (-R) + T, where R = quaternion, S = scale, T = translation

    glm::vec4 result(glm::rotate(rotation, scale * glm::vec3(src)), 0.f);

	if (src.w == 1.f)
	{
		result += glm::vec4(translation, 1.f);
	}

	return result;
}

glm::mat4 rf::to_matrix(const TRSTransform &t) {
    glm::mat4 result;

    // rotation must be unity quaternion
    // assert(std::abs(glm::length(t.rotation) - 1.0f) <=
    //        2.0f * std::numeric_limits<float>::epsilon());

    result[3][0] = t.translation.x;
    result[3][1] = t.translation.y;
    result[3][2] = t.translation.z;

    const float x2 = t.rotation.x + t.rotation.x;
    const float y2 = t.rotation.y + t.rotation.y;
    const float z2 = t.rotation.z + t.rotation.z;

    const float xx2 = t.rotation.x * x2;
    const float yy2 = t.rotation.y * y2;
    const float zz2 = t.rotation.z * z2;

    result[0][0] = (1.0f - (yy2 + zz2)) * t.scale.x;
    result[1][1] = (1.0f - (xx2 + zz2)) * t.scale.y;
    result[2][2] = (1.0f - (xx2 + yy2)) * t.scale.z;

    const float yz2 = t.rotation.y * z2;
    const float wx2 = t.rotation.w * x2;

    result[2][1] = (yz2 - wx2) * t.scale.z;
    result[1][2] = (yz2 + wx2) * t.scale.y;

    const float xy2 = t.rotation.x * y2;
    const float wz2 = t.rotation.w * z2;

    result[1][0] = (xy2 - wz2) * t.scale.y;
    result[0][1] = (xy2 + wz2) * t.scale.x;

    const float xz2 = t.rotation.x * z2;
    const float wy2 = t.rotation.w * y2;

    result[2][0] = (xz2 + wy2) * t.scale.z;
    result[0][2] = (xz2 - wy2) * t.scale.x;

    result[0][3] = 0.0f;
    result[1][3] = 0.0f;
    result[2][3] = 0.0f;
    result[3][3] = 1.0f;

    return result;
}
