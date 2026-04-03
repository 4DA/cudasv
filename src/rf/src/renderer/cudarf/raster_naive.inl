template<cudarf::ShaderType TShaderType, bool TWithTexturing>
__device__ __inline__
void compute_fragment(const cudarf::rast::Triangle &tri, const cudarf::Vec3f &bary, cudarf::rast::Fragment &frag)
{
    if (TShaderType == cudarf::SHADER_TYPE_UNLIT) {
        cudarf::Color color[3] = {tri.col[0], tri.col[1], tri.col[2]};
        frag.vertexColor = interpolate(bary, color);
    }
    else if (TShaderType == cudarf::SHADER_TYPE_PBR) {
        // TODO: use bary_persp here
        frag.pos_global = interp(bary, tri.v_world);
        frag.normal = normalize(interp(bary, tri.normal));
    }

    // textures need perspective correct interpolation, not bare linear interpolation
    if (TWithTexturing) {
        float w_rcp = dot(bary, tri.w_rcp);
        const cudarf::Vec3f bary_persp =
            1.0f / w_rcp * bary * make_vec3f(tri.w_rcp.x, tri.w_rcp.y, tri.w_rcp.z);
        frag.tex = interp(bary_persp, tri.tex);
    }

#ifdef WITH_TAA
        frag.pos_ss_hist = interp(bary, tri.v_ss_hist);
#endif

    frag.materialId = tri.materialId;
}

// naive fine rasterization, assumptions:
// - fragment shader doesn't change fragment depth

// with above points we can perform early-Z test and execute fragment shader
// only once (in separate path) for opaque path
// is this case rasterOut is used to store rasterizer output for separate
// shader pass

// for translucent pass shader computation is called for each fragment and
// results are stored out output framebuffer

template<bool TBlendingEnabled, cudarf::ShaderType TShaderType, bool TTexturingEnabled>
__global__
void fine_raster_naive(const cudarf::rast::PipeParams *pipe,
                       cudarf::Framebuffer fb,
                       cudarf::DepthValue *depthBuffer)
{
    assert(fb);

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int outIdx = x + (y * pipe->windowWidth);

    if (x >= pipe->windowWidth || y >= pipe->windowHeight) {return;}

    cudarf::rast::Fragment fragOut;
    bool isCovered = false;
    cudarf::Vec3f baryTop;
    int opaqueTriTop = -1;

    int tileX = x / CUDARF_TILE_SZ;
    int tileY = y / CUDARF_TILE_SZ;

    // TODO: compute on CPU
    int2 tilesInBin = make_int2(pipe->binCtx.binW / CUDARF_TILE_SZ, pipe->binCtx.binH / CUDARF_TILE_SZ);

	int tileId = tileX + tileY * pipe->binCtx.binsX * tilesInBin.x;

    const cudarf::rast::SimpleQueue::Segment &tileSeg = pipe->tileQHeaders[tileId];

    int32_t triCount = min(tileSeg.queueSize, pipe->tileQLimit);

    if (triCount == 0) {return;}

    cudarf::DepthValue fragDepth = depthBuffer[outIdx];
    cudarf::Color fragColor;

    // TODO: dont load framebuffer contents for opaque rendering
    fb::load(fb, x, y, fragColor);

    for (int i = 0; i < triCount; i++) {

        const cudarf::rast::Triangle &tri = pipe->tris[tileSeg.queue[i]];

        // skip triangle if its closest vertex is beyond depth value
        if (tri.zw_min > fragDepth) {continue;}

        cudarf::Vec2f frag = make_vec2f(x + 0.5f, y + 0.5f);

        cudarf::Vec3f bary = compute_bary(tri, frag);

        // DEBUG: visualize triangle edges
        // if (fabs(bary.x) > 0.05 && fabs(bary.y) > 0.05 && fabs(bary.z) > 0.05) {
        //     continue;
        // }

        // TODO: implement DirectX fill rules?
        float zw = dot(bary, tri.zw);

        // clip fragments to the near/far planes (like GL_ZERO_TO_ONE)
        if(zw < 0.0f || zw > 1.0f) {
            continue;
        }

        if (fragDepth <= zw) {continue;}

        // TODO: implement rasterzation rule for edges? conservative raster?
        // discard fragments outside the triangle
        if (!bary_in_bounds(bary)) {continue;}

        // with blending frag shader is invoked on each fragment
        // --
        if (TBlendingEnabled) {
            isCovered = true;

            compute_fragment<TShaderType, TTexturingEnabled>(tri, bary, fragOut);

            bool dstOpaque = (fragColor.w >= 1.0f);

            if (TShaderType == cudarf::SHADER_TYPE_UNLIT) {
                cudarf::Color src = compute_color_flat<TTexturingEnabled>(pipe, fragOut);
                fragColor = make_color(src.w * to_rgb(src), src.w) + (1.0f - src.w) * fragColor;
            }
            // for opaque rendering frag shader is invoked only for topmost fragment
            // --
            else {
                cudarf::Color colPBR = compute_color_pbr<TTexturingEnabled, false>(pipe, fragOut);
                float alpha = colPBR.w;
                fragColor = alpha * make_float4(to_rgb(colPBR), 1.0f) + (1.0f - alpha) * fragColor;
            }

            // we don't want to add alpha to opaque image
            if (dstOpaque) {fragColor.w = 1.0f;}

        } else {
            opaqueTriTop = tileSeg.queue[i];
            baryTop = bary;
            fragDepth = zw;
        }
    }

    if (TBlendingEnabled) {
        if (isCovered) {
            fb::store(fb, x, y, fragColor);
            // depth buffer is not modifed
            // depthBuffer[outIdx] = fragDepth;
        }
    }
    else {
        if (opaqueTriTop != -1) {
            compute_fragment<TShaderType, TTexturingEnabled>(pipe->tris[opaqueTriTop], baryTop, fragOut);
            if (TShaderType == cudarf::SHADER_TYPE_UNLIT) {
                fragColor = compute_color_flat<TTexturingEnabled>(pipe, fragOut);
            } else if (TShaderType == cudarf::SHADER_TYPE_PBR) {
                cudarf::Color linearColor = compute_color_pbr<TTexturingEnabled, true>(pipe, fragOut);
                fragColor = make_float4(linearColor.x, linearColor.y, linearColor.z, 1.0f);
            }

            fragColor.w = 1.0f;
            fb::store(fb, x, y, fragColor);

#ifdef WITH_TAA
            cudarf::Vec2f frag = make_vec2f((x + 0.5f) / pipe->windowWidth, (y + 0.5f) / pipe->windowHeight);
            float2 velocity = make_float2(x + 0.5f, y + 0.5f) - fragOut.pos_ss_hist;
            if (length(velocity) > pipe->taa.velocityThreshold) {
                pipe->taa.velocityTex[outIdx] = velocity;
            }
#endif
        }

        depthBuffer[outIdx] = fragDepth;
    }

}
