#include <cuda_runtime.h>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/printf.h>


#include "math.hpp"
#include "mesh_geometry.hpp"
#include <rf/renderer/glm_common.hpp>

using namespace rf;

uint32_t get_index(const IndicesAccessor *accessor, size_t number)
{
    assert(accessor);
    assert(number < accessor->count);
    assert(accessor->buffer);

    switch(accessor->index_type)
    {
    case IndexBufferType::UNSIGNED_INT:
        return *reinterpret_cast<const uint32_t*>(accessor->buffer->data +
            sizeof(uint32_t) * number + accessor->byte_offset);
        break;
    case IndexBufferType::UNSIGNED_SHORT:
        return *reinterpret_cast<const uint16_t*>(accessor->buffer->data +
            sizeof(uint16_t) * number + accessor->byte_offset);
        break;
    case IndexBufferType::UNSIGNED_BYTE:
        return *reinterpret_cast<const uint8_t*>(accessor->buffer->data +
            sizeof(uint8_t) * number + accessor->byte_offset);
        break;
    }

    assert(false);
    return -1;
}

template <typename T, T (*CT)(const float *), unsigned int S>
T get_attribute(const AttributesAccessor *accessor, size_t number)
{
    assert(accessor);
    assert(number < accessor->count);
    assert(accessor->type == ComponentType::FLOAT);
    assert(accessor->buffer);
    assert(accessor->num_components == S);

    size_t stride;

    if (accessor->stride) {
        stride = accessor->stride;
    }
    else {
        stride = S * sizeof(float);
    }

    size_t attrib_offset = accessor->byte_offset + stride * number;
    const float *attrib_ptr = reinterpret_cast<const float *>
        (accessor->buffer->data + attrib_offset);

    return CT(attrib_ptr);
}

glm::mat4 rf::get_attributeM4(const AttributesAccessor *accessor, size_t number)
{
    return get_attribute<glm::mat4, glm::make_mat4, 16> (accessor, number);
}

glm::vec4 rf::get_attribute4(const AttributesAccessor *accessor, size_t number)
{
    return get_attribute<glm::vec4, glm::make_vec4, 4> (accessor, number);
}

glm::vec3 rf::get_attribute3(const AttributesAccessor *accessor, size_t number)
{
    return get_attribute<glm::vec3, glm::make_vec3, 3> (accessor, number);
}

glm::vec2 rf::get_attribute2(const AttributesAccessor *accessor, size_t number)
{
    return get_attribute<glm::vec2, glm::make_vec2, 2> (accessor, number);
}

float rf::get_attribute_float(const AttributesAccessor *accessor, size_t number)
{
    return get_attribute<float, [](const float *f) -> float { return *f; }, 1>
        (accessor, number);
}

size_t MeshInfo::compute_key() const
{
    std::size_t seed = 0;
    elegant_pair(seed, normal_component_count);
    elegant_pair(seed, texcoord1_component_count);
    elegant_pair(seed, texcoord2_component_count);
    elegant_pair(seed, tangent_component_count);
    elegant_pair(seed, vertex_color_component_count);
    return seed;
}
