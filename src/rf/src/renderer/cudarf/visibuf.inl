__global__
void visibuf_count_pixels(const cudarf::rast::PipeParams *pipe,
                          const cudarf::visibuf::GeomOutput *geomFb,
                          cudarf::pipe::Atomics *g_atomics)
{
    assert(geomFb);

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= pipe->windowWidth || y >= pipe->windowHeight) {return;}

    int inIdx = x + (y * pipe->windowWidth);

    cudarf::visibuf::GeomOutput info = geomFb[inIdx];

    // Uncovered pixels are initialized to 0xFFFFFFFF in begin_frame().
    // Reject them before using drawPacketId as an array index.
    if (info.drawPacketId == 0xFFFFFFFFu || info.triId == 0xFFFFFFFFu) {
        return;
    }

    if (info.drawPacketId >= CUDARF_MAX_DRAW_PACKETS) {
        return;
    }

    // fetch material id from {triangle, packet} pair
    // const cudarf::rast::Triangle *tri = pipe->triangles[info.triId];

    uint32_t matId = pipe->drawPacketMaterials[info.drawPacketId];

    if (matId == 0xFFFFFFFFu) {return;}

    assert(matId < CUDARF_MAX_DRAW_PACKETS);
    atomicAdd(&g_atomics->visibuf.materialPixelCount[matId], 1);
}

__global__
void visibuf_build_material_offsets(uint32_t materialCount,
                                    cudarf::pipe::Atomics *g_atomics,
                                    cudarf::visibuf::MaterialOffset *matOffset)

{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= materialCount) {
        return;
    }

    uint32_t S = 0;

    for (uint32_t i = 0; i < materialCount; i++) {
        if (i == idx) {
            matOffset[idx] = {S};
        }

        S += g_atomics->visibuf.materialPixelCount[i];
    }
}

__global__
void visibuf_build_xy_lists(const cudarf::rast::PipeParams *pipe,
                            uint32_t materialCount,
                            const cudarf::visibuf::GeomOutput *geomFb,
                            cudarf::pipe::Atomics *g_atomics,
                            cudarf::visibuf::MaterialOffset *matOffsets,
                            cudarf::visibuf::XYCommand *xyCommands)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= pipe->windowWidth || y >= pipe->windowHeight) {return;}

    int inIdx = x + (y * pipe->windowWidth);

    cudarf::visibuf::GeomOutput info = geomFb[inIdx];

    if (info.drawPacketId == 0xFFFFFFFFu || info.triId == 0xFFFFFFFFu) {
        return;
    }

    if (info.drawPacketId >= CUDARF_MAX_DRAW_PACKETS) {
        return;
    }

    uint32_t matId = pipe->drawPacketMaterials[info.drawPacketId];

    // should not happen for valid drawPacketId
    if(matId == 0xFFFFFFFFu) {
        return;
    }

    uint32_t matOffset = matOffsets[matId].offset;
    uint32_t pixOffset = atomicAdd(&g_atomics->visibuf.xyMaterialOffsets[matId], 1);
    xyCommands[matOffset + pixOffset] = {x, y};
}

__global__
void visibuf_material_pass(const cudarf::rast::PipeParams *pipe,
                           const cudarf::visibuf::GeomOutput *geomFb,
                           cudarf::pipe::Atomics *g_atomics,
                           const cudarf::visibuf::MaterialOffset *matOffsets,
                           const cudarf::visibuf::XYCommand *xyCommands,
                           unsigned int matId,
                           cudarf::Framebuffer fb)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= g_atomics->visibuf.materialPixelCount[matId]) {return;}

    unsigned int xyOffset = matOffsets[matId].offset + i;

    auto [x, y] = xyCommands[xyOffset];

    int inIdx = x + (y * pipe->windowWidth);
    cudarf::visibuf::GeomOutput info = geomFb[inIdx];

    int triId = info.triId;
    assert(triId != 0xFFFFFFFFu);

    const cudarf::rast::Triangle &tri = pipe->tris[triId];

    cudarf::Vec3f baryLambda;

    cudarf::Vec2f fragWindow = make_vec2f(x + 0.5f, y + 0.5f);
    cudarf::Vec3f baryAffine = compute_bary_affine2(tri, fragWindow);

#ifdef CUDARF_FORCE_AFFINE_BARYCENTRICS
    // Match the legacy opaque path exactly by shading from affine screen-space
    // barycentrics instead of the reconstructed perspective-correct basis.
    baryLambda = baryAffine;
#else

    // TODO: to be used when derivatives are needed
    // glm::vec2 windowSize((float)pipe->windowWidth, (float)pipe->windowHeight);

    // // Reconstruct the clip-space inputs expected by compute_barys() from the
    // // stored screen-space vertex positions plus per-vertex 1/w.
    // auto make_clip_pos = [windowSize](const float2 &screenPos, float invW) {
    //     float w = 1.0f / invW;
    //     glm::vec2 ndc = 2.0f * glm::vec2(screenPos.x, screenPos.y) / windowSize - glm::vec2(1.0f, 1.0f);
    //     return glm::vec4(ndc * w, 0.0f, w);
    // };

    // glm::vec4 P0 = make_clip_pos(tri.sP0, tri.w_rcp.x);
    // glm::vec4 P1 = make_clip_pos(tri.sP1, tri.w_rcp.y);
    // glm::vec4 P2 = make_clip_pos(tri.sP2, tri.w_rcp.z);
    // glm::vec2 ndc = 2.0f * glm::vec2(x + 0.5f, y + 0.5f) / windowSize - glm::vec2(1.0f, 1.0f);

    // Barycentric bary = compute_bary_persp_deriv(P0, P1, P2, ndc, windowSize);
    // baryLambda = make_vec3f(bary.lambda.x, bary.lambda.y, bary.lambda.z);

    baryLambda = compute_bary_persp(baryAffine, tri.w_rcp);
#endif

    cudarf::rast::Fragment frag;

    const cudarf::Material &material = pipe->materials[matId];
    bool withTexturing =
        material.albedoTex.textureObject ||
        material.emissiveTex.textureObject ||
        material.occlusionTex.textureObject ||
        material.metRoughTex.textureObject;

    cudarf::Color shadedColor;

    if (material.type == cudarf::SHADER_TYPE_UNLIT) {
        shadedColor = withTexturing
            ? shade_fragment<cudarf::SHADER_TYPE_UNLIT, true, false>(
                pipe, tri, make_vec3f(0.0f, 0.0f, 0.0f), baryLambda, frag)
            : shade_fragment<cudarf::SHADER_TYPE_UNLIT, false, false>(
                pipe, tri, make_vec3f(0.0f, 0.0f, 0.0f), baryLambda, frag);
    }
    else {
        bool withClearcoat = material.clearcoatFactor > 0.0f;
        if (withTexturing) {
            shadedColor = withClearcoat
                ? shade_fragment<cudarf::SHADER_TYPE_PBR, true, true>(
                    pipe, tri, make_vec3f(0.0f, 0.0f, 0.0f), baryLambda, frag)
                : shade_fragment<cudarf::SHADER_TYPE_PBR, true, false>(
                    pipe, tri, make_vec3f(0.0f, 0.0f, 0.0f), baryLambda, frag);
        }
        else {
            shadedColor = withClearcoat
                ? shade_fragment<cudarf::SHADER_TYPE_PBR, false, true>(
                    pipe, tri, make_vec3f(0.0f, 0.0f, 0.0f), baryLambda, frag)
                : shade_fragment<cudarf::SHADER_TYPE_PBR, false, false>(
                    pipe, tri, make_vec3f(0.0f, 0.0f, 0.0f), baryLambda, frag);
        }
    }

    shadedColor.w = 1.0f;

    fb::store(fb, x, y, shadedColor);

    // DEBUG: write barycentric values to framebuffer
    // cudarf::Color debugColor = make_vec4f(baryLambda.x, baryLambda.y, baryLambda.z, 1.0f);
    // fb::store(fb, x, y, debugColor);

#ifdef WITH_TAA
    int outIdx = x + (y * pipe->windowWidth);

    float2 velocity = make_float2(x + 0.5f, y + 0.5f) - frag.pos_ss_hist;
    if (length(velocity) > pipe->taa.velocityThreshold) {
        pipe->taa.velocityTex[outIdx] = velocity;
    }
#endif

}
