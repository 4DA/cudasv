#ifndef CUDARF_COLOR_SPACE_INL
#define CUDARF_COLOR_SPACE_INL

namespace cudarf
{
namespace shading
{

static __device__ __forceinline__ float linear_channel_to_srgb(float channel)
{
    channel = fmaxf(channel, 0.0f);
    return channel < 0.0031308f
        ? 12.92f * channel
        : 1.055f * powf(channel, 1.0f / 2.4f) - 0.055f;
}

static __device__ __forceinline__ float srgb_channel_to_linear(float channel)
{
    channel = fmaxf(channel, 0.0f);
    return channel <= 0.04045f
        ? channel / 12.92f
        : powf((channel + 0.055f) / 1.055f, 2.4f);
}

static __device__ __forceinline__ float3 linear_to_srgb(float3 color)
{
    return make_float3(
        linear_channel_to_srgb(color.x),
        linear_channel_to_srgb(color.y),
        linear_channel_to_srgb(color.z));
}

static __device__ __forceinline__ float3 srgb_to_linear(float3 color)
{
    return make_float3(
        srgb_channel_to_linear(color.x),
        srgb_channel_to_linear(color.y),
        srgb_channel_to_linear(color.z));
}

static __device__ __forceinline__ float4 tone_map(float4 color, float exposure)
{
    return make_float4(linear_to_srgb(make_float3(color.x, color.y, color.z) * exposure), color.w);
}

} // namespace shading
} // namespace cudarf

#endif // CUDARF_COLOR_SPACE_INL
