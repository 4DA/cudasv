
// temporal antialiasing subroutines, for reference see:
// https://www.gdcvault.com/play/1022970/Temporal-Reprojection-Anti-Aliasing-in
// --
struct ProjectionExtents {
    float scaleX;
    float scaleY;
    float biasX;
    float biasY;
    float2 texelSize;
};

__device__ __inline__ float2 reproject(glm::mat4 VP_prev,
                                       cudarf::Vec3f p_w,
                                       unsigned int width,
                                       unsigned int height)
{
    auto q_clip = VP_prev * glm::vec4(p_w.x, p_w.y, p_w.z, 1.0f);
    cudarf::Vec2f q_uv = 0.5f * make_float2(q_clip.x / q_clip.w, q_clip.y / q_clip.w) + make_float2(0.5f, 0.5f);
    return make_float2(q_uv.x, q_uv.y);
}

__device__ __host__
ProjectionExtents make_extents(float fovY, float aspect, float width, float height, float texelOffsetX, float texelOffsetY)
{
    float extentY = std::tan(0.5f * fovY);
    float extentX = extentY * aspect;

    float texelSizeX = extentX / (0.5f * width);
    float texelSizeY = extentY / (0.5f * height);

    float jitterX = texelSizeX * texelOffsetX;
    float jitterY = texelSizeY * texelOffsetY;

    return ProjectionExtents{extentX, extentY, jitterX, jitterY, make_float2(texelSizeX, texelSizeY)};
}

__device__ __host__
ProjectionExtents make_extents(float fovY, float aspect, float width, float height)
{
    return make_extents(fovY, aspect, width, height, 0.0f, 0.0f);
}

__device__ void sample_min_max(cudaTextureObject_t rasterOut, float2 uv, float2 texelSize, float4 &cmin, float4 &cmax)
{
    float2 du = make_float2(texelSize.x, 0.0f);
    float2 dv = make_float2(0.0f, texelSize.y);

    float4 ctl = fb::tex_sample_4f32(rasterOut, uv - dv - du);
    float4 ctc = fb::tex_sample_4f32(rasterOut, uv - dv);
    float4 ctr = fb::tex_sample_4f32(rasterOut, uv - dv + du);
    float4 cml = fb::tex_sample_4f32(rasterOut, uv - du);
    float4 cmc = fb::tex_sample_4f32(rasterOut, uv);
    float4 cmr = fb::tex_sample_4f32(rasterOut, uv + du);
    float4 cbl = fb::tex_sample_4f32(rasterOut, uv + dv - du);
    float4 cbc = fb::tex_sample_4f32(rasterOut, uv + dv);
    float4 cbr = fb::tex_sample_4f32(rasterOut, uv + dv + du);

    cmin = fminf(ctl, fminf(ctc, fminf(ctr, fminf(cml, fminf(cmc, fminf(cmr, fminf(cbl, fminf(cbc, cbr))))))));
    cmax = fmaxf(ctl, fmaxf(ctc, fmaxf(ctr, fmaxf(cml, fmaxf(cmc, fmaxf(cmr, fmaxf(cbl, fmaxf(cbc, cbr))))))));
}

template<bool TWithVelocity>
__global__
void resolve_TAA(cudarf::ProjectionParams proj,
                 unsigned int width,
                 unsigned int height,
                 cudarf::FBTexture rasterOut,
                 cudarf::FBTexture hist,
                 cudarf::Velocity *velocityTex,
                 const cudarf::CommonUniforms frameUniforms,
                 const cudarf::CommonUniforms histUniforms,
                 cudarf::Framebuffer fb,
                 cudarf::DepthValue *depthBuffer,
                 float kFeedback,
                 float2 jitter)
{
    assert(fb);
    assert(hist);

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int fb_idx = x + (y * width);

    if (x >= width || y >= height) {return;}

    float u = (x+0.5f) / (float)width;
    float v = (y+0.5f) / (float)height;

    auto extents = make_extents(proj.fovY,
                                float(width) / height,
                                width,
                                height,
                                jitter.x,
                                jitter.y);

    float2 repr_txc;

    if (TWithVelocity) {
        repr_txc = make_float2(u, v) - velocityTex[fb_idx] / make_float2(width * 1.0f, height * 1.0f);
    }
    else {
        float fragDepth = depthBuffer[fb_idx];

        float linearDepth = fb::get_linear_depth(fragDepth, proj.near, proj.far);

        auto ray = glm::vec3(
            (2.0f * (x + 0.5) / width - 1.0f) * extents.scaleX,
            (2.0f * (y + 0.5) / height - 1.0f) * extents.scaleY,
            // we use -Z opengl camera convention
            -1.0f);

        glm::vec3 posView = linearDepth * ray;
        auto ws = frameUniforms.V_inv * glm::vec4(posView, 1.0f);

        repr_txc = reproject(histUniforms.PV,
                             make_float3(ws.x, ws.y , ws.z),
                             width,
                             height);
    }

    // to disable reprojection
    // repr_txc.x = u - jitter.x;
    // repr_txc.y = v - jitter.y;

    // current frame must be unjittered
    float4 fragColor = fb::tex_sample_4f32(rasterOut, u - jitter.x, v - jitter.y);

    cudarf::Color sampleHist = fb::tex_sample_4f32(hist, repr_txc.x, repr_txc.y);

    // clamp history sample to neighborhood of current sample
    float4 cmin, cmax;
    sample_min_max(rasterOut,
                   make_float2(u - jitter.x, v - jitter.y),
                   extents.texelSize,
                   cmin,
                   cmax);

    sampleHist = clamp(sampleHist, cmin, cmax);

    cudarf::Color blendColor = (fragColor * (1.0f - kFeedback) + sampleHist * kFeedback);
    fb::store(fb, x, y, blendColor);

}

#ifdef WITH_TAA
#endif
