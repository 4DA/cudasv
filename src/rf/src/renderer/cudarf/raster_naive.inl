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
                       cudarf::DepthValue *depthBuffer,
                       cudarf::visibuf::GeomOutput *geomFb)
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

        cudarf::Vec3f baryAffine = compute_bary_affine(tri, frag);
#ifdef CUDARF_FORCE_AFFINE_BARYCENTRICS
        cudarf::Vec3f baryPersp = baryAffine;
#else
        cudarf::Vec3f baryPersp = compute_bary_persp(baryAffine, tri.w_rcp);
#endif

        // DEBUG: visualize triangle edges
        // if (fabs(bary.x) > 0.05 && fabs(bary.y) > 0.05 && fabs(bary.z) > 0.05) {
        //     continue;
        // }

        // TODO: implement DirectX fill rules?
        float zw = dot(baryAffine, tri.zw);

        // clip fragments to the near/far planes (like GL_ZERO_TO_ONE)
        if(zw < 0.0f || zw > 1.0f) {
            continue;
        }

        if (fragDepth <= zw) {continue;}

        // TODO: implement rasterzation rule for edges? conservative raster?
        // discard fragments outside the triangle
        if (!bary_in_bounds(baryAffine)) {continue;}

        // with blending frag shader is invoked on each fragment
        // --
        if (TBlendingEnabled) {
            isCovered = true;

            bool dstOpaque = (fragColor.w >= 1.0f);
            cudarf::Color shaded = shade_fragment<TShaderType, TTexturingEnabled, false>(pipe, tri, baryPersp, fragOut);
            float alpha = shaded.w;
            fragColor = alpha * make_float4(to_rgb(shaded), 1.0f) + (1.0f - alpha) * fragColor;

            // we don't want to add alpha to opaque image
            if (dstOpaque) {fragColor.w = 1.0f;}

        } else {
            opaqueTriTop = tileSeg.queue[i];
            baryTop = baryAffine;
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
        // we shade opaque fragments in visibuf pass
        // if (TShadeOpaque) {
        //     fragColor = shade_fragment<TShaderType, TTexturingEnabled, true>(
        //         pipe,
        //         pipe->tris[opaqueTriTop],
        //         baryTop,
        //         fragOut);
        // }

        fragColor.w = 1.0f;

        if (geomFb != nullptr && opaqueTriTop != -1) {
            geomFb[outIdx] = {
                pipe->tris[opaqueTriTop].id,
                pipe->tris[opaqueTriTop].drawPacketId
            };
        }

        depthBuffer[outIdx] = fragDepth;
    }
}
