#ifndef CUDARF_FRAME_CONVENTIONS_INL
#define CUDARF_FRAME_CONVENTIONS_INL

namespace cudarf
{
namespace shading
{

// The current environment maps are authored with +Z as forward, while the
// renderer's vehicle-space convention uses +X as forward. Keep the conversion
// in one place so the shading code does not carry project-specific names.
static __device__ __forceinline__ glm::vec3 vehicle_frame_to_cubemap_frame(glm::vec3 value)
{
    return glm::vec3(-value.y, value.z, value.x);
}

} // namespace shading
} // namespace cudarf

#endif // CUDARF_FRAME_CONVENTIONS_INL
