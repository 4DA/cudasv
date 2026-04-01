#include "renderer/gltf/accessors.hpp"

#include <cassert>
#include <memory>

#include <spdlog/spdlog.h>

using rf::AttributesAccessor;
using rf::ComponentType;
using rf::GeometryBuffer;

namespace loader::gltf
{

std::optional<glm::vec2> to_glm_vec2(const std::vector<double> &std_vec)
{
    if (std_vec.size() != 2) {
        return std::nullopt;
    }

    return glm::vec2(static_cast<float>(std_vec[0]), static_cast<float>(std_vec[1]));
}

std::optional<glm::vec3> to_glm_vec3(const std::vector<double> &std_vec)
{
    if (std_vec.size() != 3) {
        return std::nullopt;
    }

    return glm::vec3(static_cast<float>(std_vec[0]),
                     static_cast<float>(std_vec[1]),
                     static_cast<float>(std_vec[2]));
}

std::optional<glm::vec4> to_glm_vec4(const std::vector<double> &std_vec)
{
    if (std_vec.size() != 4) {
        return std::nullopt;
    }

    return glm::vec4(static_cast<float>(std_vec[0]),
                     static_cast<float>(std_vec[1]),
                     static_cast<float>(std_vec[2]),
                     static_cast<float>(std_vec[3]));
}

static GeometryBuffer::Type get_buffer_view_type(const tinygltf::Accessor &accessor,
                                                 const tinygltf::BufferView &buffer_view)
{
    if (buffer_view.target) {
        switch (buffer_view.target)
        {
        case TINYGLTF_TARGET_ARRAY_BUFFER:
            return GeometryBuffer::Type::ArrayBuffer;
        case TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER:
            return GeometryBuffer::Type::ElementBuffer;
        default:
            SPDLOG_ERROR("Unknown buffer_view target: {}. Trying to infer from accessor",
                         buffer_view.target);
            break;
        }
    }

    if (accessor.type == TINYGLTF_TYPE_SCALAR &&
        (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE ||
         accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT ||
         accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)) {
        return GeometryBuffer::Type::ElementBuffer;
    }

    return GeometryBuffer::Type::ArrayBuffer;
}

static std::unique_ptr<GeometryBuffer>
load_geometry_buffer(const tinygltf::Model &model, const tinygltf::Accessor &accessor)
{
    size_t bufferId = accessor.bufferView;
    const tinygltf::BufferView &buffer_view = model.bufferViews[bufferId];
    const tinygltf::Buffer &buffer = model.buffers[buffer_view.buffer];
    GeometryBuffer::Type type = get_buffer_view_type(accessor, buffer_view);

    return std::make_unique<GeometryBuffer>(buffer_view.name,
                                            &(buffer.data[0]) + buffer_view.byteOffset,
                                            buffer_view.byteLength,
                                            type);
}

void init_attributes_accessor(const tinygltf::Model &model,
                              AttributesAccessor &attribPtr,
                              const std::string &name,
                              const tinygltf::Accessor &accessor)
{
    const tinygltf::BufferView &buffer_view = model.bufferViews[accessor.bufferView];

    if (buffer_view.target != 0 && buffer_view.target != TINYGLTF_TARGET_ARRAY_BUFFER) {
        SPDLOG_ERROR("buffer_view (id={}) target is not ELEMENT_ARRAY_BUFFER", accessor.bufferView);
        assert(false);
    }

    ComponentType type;

    switch (accessor.componentType) {
    case TINYGLTF_COMPONENT_TYPE_FLOAT:
        type = ComponentType::FLOAT;
        break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        type = ComponentType::UNSIGNED_BYTE;
        break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        type = ComponentType::UNSIGNED_SHORT;
        break;
    case TINYGLTF_COMPONENT_TYPE_SHORT:
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
    case TINYGLTF_COMPONENT_TYPE_BYTE:
    default:
        SPDLOG_ERROR("Unsupported component type: {}", accessor.type);
        assert(false);
        return;
    }

    size_t compos;

    switch (accessor.type) {
    case TINYGLTF_TYPE_SCALAR:
        compos = 1;
        break;
    case TINYGLTF_TYPE_VEC2:
        compos = 2;
        break;
    case TINYGLTF_TYPE_VEC3:
        compos = 3;
        break;
    case TINYGLTF_TYPE_VEC4:
        compos = 4;
        break;
    case TINYGLTF_TYPE_MAT2:
        compos = 4;
        break;
    case TINYGLTF_TYPE_MAT3:
        compos = 9;
        break;
    case TINYGLTF_TYPE_MAT4:
        compos = 16;
        break;
    default:
        SPDLOG_ERROR("Unsupported element type: {}", accessor.type);
        assert(false);
        return;
    }

    attribPtr.name = name;
    attribPtr.stride = buffer_view.byteStride;
    attribPtr.count = accessor.count;
    attribPtr.num_components = compos;
    attribPtr.type = type;
    attribPtr.byte_offset = accessor.byteOffset;
    attribPtr.buffer = load_geometry_buffer(model, accessor);
}

} // namespace loader::gltf
