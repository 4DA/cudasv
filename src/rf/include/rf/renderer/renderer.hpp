#ifndef RF_RENDERER_HPP
#define RF_RENDERER_HPP

#include <unordered_map>
#include <vector>

#include <rf/renderer/scene.hpp>

namespace rf
{

struct ModelLoadFlags {
    bool load_punctual = true;

    // if enabled, fragments, that lie on surface fragment with high curvature
    // will receive less specular light contribution. Specular aliasing will be
    // decreased at expense of some performance and darker on surface edges.
    // increasing MAX_SPEC_CUTOFF will strengthen the effect and decreasing
    // will weaken correspondingly
    bool use_specular_aa = true;


    // During multisampling, if centroid is not present, then the written value
    // can be interpolated to to an arbitrary position within the pixel. This may
    // be the pixel's center, one of the sample locations within the pixel, or an
    // arbitrary location. Most importantly of all, this sample may lie outside of
    // the actual primitive being rendered, since a primitive can cover only part
    // of a pixel's area. If the implementation computes the sample based on the
    // center of the pixel, and the primitive doesn't actually cover the pixel's
    // center (remember: in multisampling, this can still produce a non-zero
    // number of samples), then the interpolated value will be outside of the
    // primitive's borders.

    // TLDR: don't use when MSAA is not enabled

    bool use_centroid_for_msaa = true;

    // keep disabled when debugging shaders, because key isn't changed if you just
    // change shader source
    bool use_binary_shaders = false;

    // catalogue path must exist
    std::string binary_shader_path = "cache";
};

} // namespace rf

#endif
