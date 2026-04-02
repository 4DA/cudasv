#include "renderer/gltf/accessors.hpp"

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
    if (accessor.bufferView < 0 ||
        accessor.bufferView >= static_cast<int>(model.bufferViews.size())) {
        SPDLOG_ERROR("Accessor references invalid bufferView {}", accessor.bufferView);
        return nullptr;
    }

    const tinygltf::BufferView &buffer_view = model.bufferViews[accessor.bufferView];
    if (buffer_view.buffer < 0 ||
        buffer_view.buffer >= static_cast<int>(model.buffers.size())) {
        SPDLOG_ERROR("BufferView references invalid buffer {}", buffer_view.buffer);
        return nullptr;
    }

    const tinygltf::Buffer &buffer = model.buffers[buffer_view.buffer];
    GeometryBuffer::Type type = get_buffer_view_type(accessor, buffer_view);
    const size_t byteOffset = buffer_view.byteOffset;
    const size_t byteLength = buffer_view.byteLength;

    if (byteOffset > buffer.data.size() ||
        byteLength > buffer.data.size() - byteOffset) {
        SPDLOG_ERROR("BufferView range [{}, {}) exceeds buffer size {}",
                     byteOffset,
                     byteOffset + byteLength,
                     buffer.data.size());
        return nullptr;
    }

    return std::make_unique<GeometryBuffer>(buffer_view.name,
                                            buffer.data.data() + byteOffset,
                                            byteLength,
                                            type);
}

bool init_attributes_accessor(const tinygltf::Model &model,
                              AttributesAccessor &attribPtr,
                              const std::string &name,
                              const tinygltf::Accessor &accessor)
{
    if (accessor.bufferView < 0 ||
        accessor.bufferView >= static_cast<int>(model.bufferViews.size())) {
        SPDLOG_ERROR("Accessor '{}' references invalid bufferView {}", name, accessor.bufferView);
        return false;
    }

    const tinygltf::BufferView &buffer_view = model.bufferViews[accessor.bufferView];

    if (buffer_view.target != 0 && buffer_view.target != TINYGLTF_TARGET_ARRAY_BUFFER) {
        SPDLOG_ERROR("BufferView {} for accessor '{}' has unsupported target {}",
                     accessor.bufferView,
                     name,
                     buffer_view.target);
        return false;
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
        SPDLOG_ERROR("Unsupported component type {} for accessor '{}'",
                     accessor.componentType,
                     name);
        return false;
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
        SPDLOG_ERROR("Unsupported element type {} for accessor '{}'",
                     accessor.type,
                     name);
        return false;
    }

    auto buffer = load_geometry_buffer(model, accessor);
    if (!buffer) {
        return false;
    }

    attribPtr.name = name;
    attribPtr.stride = buffer_view.byteStride;
    attribPtr.count = accessor.count;
    attribPtr.num_components = compos;
    attribPtr.type = type;
    attribPtr.byte_offset = accessor.byteOffset;
    attribPtr.buffer = std::move(buffer);
    return true;
}

} // namespace loader::gltf
