#ifndef RF_GEOMETRY
#define RF_GEOMETRY

#include <string>

#include <rf/renderer/glm_common.hpp>
#include <rf/renderer/cudarf/types.hpp>

namespace rf
{

struct MeshInfo
{
    // number of components in the normal attribute. "0" indicates the attribute is not available
    std::uint8_t normal_component_count = 0;

    // number of components in the first texture coordinate attribute. "0" indicates the attribute is not available
    std::uint8_t texcoord1_component_count = 0;

    // number of components in the second texture coordinate attribute. "0" indicates the attribute is not available
    std::uint8_t texcoord2_component_count = 0;

    // number of components in the tangent attribute. "0" indicates the attribute is not available
    std::uint8_t tangent_component_count = 0;

    // number of components in the vertex color attribute. "0" indicates the attribute is not available
    std::uint8_t vertex_color_component_count = 0;

    //  computed from all properties properties
    mutable std::size_t geometry_key_cached = 0;

    MeshInfo() = default;

    MeshInfo(std::uint8_t normal_component_count,
             std::uint8_t texcoord1_component_count,
             std::uint8_t texcoord2_component_count,
             std::uint8_t tangent_component_count,
             std::uint8_t vertex_color_component_count) :
        normal_component_count(normal_component_count),
        texcoord1_component_count(texcoord1_component_count),
        texcoord2_component_count(texcoord2_component_count),
        tangent_component_count(tangent_component_count),
        vertex_color_component_count(vertex_color_component_count),
        geometry_key_cached(compute_key()) {}

    std::size_t get_key() const {
        if (geometry_key_cached == 0) {
            geometry_key_cached = compute_key();
        } return geometry_key_cached;
    }

    bool operator==(const MeshInfo &) const = default;

private:
    std::size_t compute_key() const;
};


struct GeometryBuffer {
public:
    enum class Type {
        ArrayBuffer = 0,
        ElementBuffer
    };

    std::string name;

    // pointer to buffer in CPU-addressable memory
    const unsigned char *data;

    // length of buffer
    size_t byteLength;

    // vertex attribute or index buffer
    Type type;

    GeometryBuffer() = delete;
    GeometryBuffer(const GeometryBuffer&) = delete;
    GeometryBuffer & operator=(GeometryBuffer const &) = delete;
    ~GeometryBuffer() = default;

    GeometryBuffer(const std::string &name, const unsigned char *data,
        size_t byteLength, Type type):
        name(name), data(data), byteLength(byteLength), type(type){}
};

enum class ComponentType {
    UNSIGNED_BYTE = 0,
    UNSIGNED_SHORT,
    FLOAT
};

enum class IndexBufferType {
    UNSIGNED_BYTE = 0,
    UNSIGNED_SHORT,
    UNSIGNED_INT,
};

enum class PrimitiveMode
{
    TRIANGLES,
    LINES,
};


struct IndicesAccessor {
    // number of elements
    size_t count;

    // index size
    IndexBufferType index_type;

    // byte offset of first element inside index array
    size_t byte_offset;

    // geometry buffer the accessor is looking at
    GeometryBuffer *buffer;
};

struct AttributesAccessor {
    // vertex attribute name
    std::string   name;

    // name of shader attribute
    std::string attribute_name;

    // byte offset between consecutive generic vertex attributes.
    size_t        stride;

    // number of elements
    size_t        count = 0;

    // number of components per generic vertex attribute.
    size_t        num_components;

    // type of each compnent
    ComponentType type;

    // byte offset of first element inside attribute array
    size_t byte_offset;

    // geometry buffer the accessor is looking at
    std::unique_ptr<GeometryBuffer> buffer;
};

struct NaiveMesh {
    std::string name; // internal (non-unique) mesh name
    std::vector<int> idx;
    std::vector<cudarf::Vec3f> vertices;
    std::vector<cudarf::Vec3f> normals;
    std::vector<cudarf::Vec4f> colors;
    std::vector<cudarf::Vec2f> texcoords;

    glm::vec3 vertexMin;
    glm::vec3 vertexMax;

    NaiveMesh():
        vertexMin(glm::vec3(std::numeric_limits<float>::max())),
        vertexMax(glm::vec3(std::numeric_limits<float>::lowest()))
        {}
};

using NaiveMeshPtr = std::shared_ptr<rf::NaiveMesh>;

struct GltfMesh {
    rf::AttributesAccessor vertices;
    rf::AttributesAccessor normals;
    rf::AttributesAccessor texcoords;
    rf::AttributesAccessor tangents;

    std::size_t count;
    uint32_t *indexPtr;
    bool convertIdxTo32;

    //rf::geometry_info info;
};

// get vertex index from indices accessor
uint32_t get_index(const IndicesAccessor *accessor, size_t number);

// get floating point attribute value from accessor
float get_attribute_float(const AttributesAccessor *accessor, size_t number);

// get float2 attribute value from accessor
glm::vec2 get_attribute2(const AttributesAccessor *accessor, size_t number);

// get float3 attribute value from accessor
glm::vec3 get_attribute3(const AttributesAccessor *accessor, size_t number);

// get float4 attribute value from accessor
glm::vec4 get_attribute4(const AttributesAccessor *accessor, size_t number);

// get Mat4x4 attribute value from accessor
glm::mat4 get_attributeM4(const AttributesAccessor *accessor, size_t number);

} //namespace rf

#endif
