#ifndef RF_TRS_TRANSFORM_HPP
#define RF_TRS_TRANSFORM_HPP

#include <vector>
#include <string>
#include <sstream>

#include <rf/renderer/glm_common.hpp>

namespace rf
{

// check if rotation is unity quaternion
void check_rotation_unity(const glm::quat rotation);

// affine transform stored as {translation, rotation, scale}
struct TRSTransform {
    glm::vec3 translation; // local space translation
    glm::quat rotation;    // local space rotation quaternion
    glm::vec3 scale;       // local space scale

    TRSTransform(glm::vec3 translation = glm::vec3(0.0f, 0.0f, 0.0f),
          glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
          glm::vec3 scale = glm::vec3(1.0f)) : translation(translation), rotation(rotation), scale(scale)
        {
            check_rotation_unity(rotation);
        }

    TRSTransform(const TRSTransform &other)
    {
        this->translation = other.translation;
        this->rotation = other.rotation;
        this->scale = other.scale;
        check_rotation_unity(rotation);
    }

    TRSTransform& operator=(const TRSTransform &other)
    {
        this->translation = other.translation;
        this->rotation = other.rotation;
        this->scale = other.scale;
        check_rotation_unity(rotation);
        return *this;
    }

    glm::vec4 apply(const glm::vec4 &vec) const;

    std::string to_string() const {
        std::stringstream ss;
        ss.precision(2);

        ss << "T: vec3(" << translation.x << ", " << translation.y << ", " << translation.z << "), "
           << "R: quat(" << rotation.w << ", "  << rotation.x << ", " << rotation.y << ", " << rotation.z << "), "
           << "S: vec3(" << scale.x << ", " << scale.y << ", " << scale.z << ")";

        return ss.str();
    }
};

// Compose two transforms A, B, such that C = compose(A, B)(x) = B(A(x)),
// where x is vertex. The operation isn't commutative. One important
// consideration is that resulting scale is applied in x's local space, i.e
// axis rotation is ignored.
TRSTransform compose(const TRSTransform &A, const TRSTransform &B);

glm::mat4 to_matrix(const TRSTransform &t);

}

#endif
