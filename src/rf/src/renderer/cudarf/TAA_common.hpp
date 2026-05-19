#ifndef TAA_COMMON_HPP
#define TAA_COMMON_HPP

#ifdef WITH_TAA


// ---------------------------------------------------------------------------
// Jitter sample patterns and TAA uniform preparation
// ---------------------------------------------------------------------------

static const float2 Dirty[] = {
    {0.25f,  0.0f},
    {0.75f,  0.25f},
    {0.5f,   0.75f},
    {0.0f,   0.5f}
};

static const float2 Rotated4_Helix[] = {
    {-0.125f, -0.375f},
    { 0.125f,  0.375f},
    { 0.375f, -0.125f},
    {-0.375f,  0.125f}
};

static float2 get_jitter(const cudarf::pipe::Ctx &desc, unsigned int frameCounter)
{
    const auto &params = desc.TAA;
    float2 delta = make_float2(desc.TAA.scale / desc.width,
                               desc.TAA.scale / desc.height);

    switch (params.pattern) {
    case cudarf::TAA_Pattern::Center:
        return make_float2(0.0f, 0.0f);
    case cudarf::TAA_Pattern::Dirty:
        return delta * Dirty[frameCounter % 4];
    case cudarf::TAA_Pattern::Helix:
        return delta * Rotated4_Helix[frameCounter % 4];
    case cudarf::TAA_Pattern::Halton:
        return delta * desc.TAA.pointsHalton[frameCounter % desc.TAA.pointsHalton.size()];
    }
    return make_float2(0.0f, 0.0f);
}

// Returns a jitter-modified copy of clean uniforms for use in the current frame.
static cudarf::CommonUniforms prepare_for_TAA(const cudarf::pipe::Ctx &desc,
                                              unsigned int frameCounter,
                                              unsigned int /*numSamples*/,
                                              const cudarf::CommonUniforms &clean)
{
    auto jitter = get_jitter(desc, frameCounter);
    glm::mat4 jitteredP = clean.P;
    jitteredP[2][0] += jitter.x;
    jitteredP[2][1] += jitter.y;
    return cudarf::make_common(jitteredP, clean.V);
}

#endif

#endif
