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

struct Barycentric
{
    glm::vec3 lambda;
    glm::vec3 ddx;
    glm::vec3 ddy;
};

__device__
glm::vec3 rcp(glm::vec3 src)
{
    return glm::vec3(1.0 / src.x, 1.0 / src.y, 1.0 / src.z);
}


// source:
// https://github.com/ConfettiFX/The-Forge/blob/2d453f376ef278f66f97cbaf36c0d12e4361e275/Examples_3/Aura/src/Shaders/FSL/visibilityBuffer_shade.frag.fsl#L33
//
// Compact math structure:
//
// 1. Project clip-space vertices into NDC:
//      n_i = P_i.xy / P_i.w
//      q_i = 1 / P_i.w
//
// 2. Affine barycentric terms over screen space are linear in x,y, so their
//    gradients are constant across the triangle:
//      d(lambda_i * q_i)/dx, d(lambda_i * q_i)/dy
//
//    This is what ret.ddx / ret.ddy store component-wise for i in {0,1,2}.
//
// 3. The interpolated reciprocal depth is:
//      q(x, y) = sum_i lambda_i(x, y) * q_i
//
//    and the perspective-correct barycentrics are:
//      lambda_i(x, y) = (lambda_i(x, y) * q_i) / q(x, y)
//
// 4. The final derivative projection step converts the raw affine gradients in
//    the lambda/w domain into perspective-correct per-pixel barycentric
//    gradients that can be used for later attribute differentials.

__device__
Barycentric compute_barys(glm::vec4 P0, glm::vec4 P1, glm::vec4 P2, glm::vec2 ndc, glm::vec2 winSize)
{
    Barycentric ret;

    // Perspective-correct interpolation works in 1/w, so convert the clip-space
    // vertex w components into reciprocal form once up front.
    glm::vec3 invW = rcp(glm::vec3(P0.w, P1.w, P2.w));

    // Project the clip-space triangle into NDC. The sample position is also in
    // NDC, so all later barycentric reconstruction happens in this space.
    glm::vec2 ndc0 = glm::vec2(glm::vec2(P0) * invW.x);
    glm::vec2 ndc1 = glm::vec2(glm::vec2(P1) * invW.y);
    glm::vec2 ndc2 = glm::vec2(glm::vec2(P2) * invW.z);

    // Invert the 2x2 screen-space edge matrix once so barycentric gradients can
    // be evaluated from NDC deltas instead of solving the triangle per sample.
    float invDet = 1.0 / glm::determinant(glm::mat2(ndc2 - ndc1, ndc0 - ndc1));

    // These are the per-vertex barycentric gradients multiplied by 1/w, which
    // is the form needed for perspective-correct interpolation.
    ret.ddx = glm::vec3(ndc1.y - ndc2.y, ndc2.y - ndc0.y, ndc0.y - ndc1.y) * invDet * invW;
    ret.ddy = glm::vec3(ndc2.x - ndc1.x, ndc0.x - ndc2.x, ndc1.x - ndc0.x) * invDet * invW;

    // Summing the per-vertex terms gives the screen-space gradient of the
    // interpolated reciprocal w.
    float ddxSum = glm::dot(ret.ddx, glm::vec3(1.0, 1.0, 1.0));
    float ddySum = glm::dot(ret.ddy, glm::vec3(1.0, 1.0, 1.0));

    glm::vec2 deltaVec = ndc - ndc0;

    // Evaluate 1/w at the sample and invert it back to w for the final
    // perspective-correct barycentric reconstruction.
    float interpInvW = invW.x + deltaVec.x*ddxSum + deltaVec.y*ddySum;
    float interpW = 1.0 / interpInvW;

    // Reconstruct the perspective-correct barycentrics at this sample.
    ret.lambda.x = interpW * (invW[0] + deltaVec.x*ret.ddx.x + deltaVec.y*ret.ddy.x);
    ret.lambda.y = interpW * (0.0f    + deltaVec.x*ret.ddx.y + deltaVec.y*ret.ddy.y);
    ret.lambda.z = interpW * (0.0f    + deltaVec.x*ret.ddx.z + deltaVec.y*ret.ddy.z);

    // to compute screen-space barycentrics directly
    // vec2 dndc0 = ndc - ndc0;
    // vec2 dndc1 = ndc - ndc1;
    // vec2 dndc2 = ndc - ndc2;
    // const vec2 lambda_yz = vec2(determinant(mat2(dndc2, dndc0)), determinant(mat2(dndc0, dndc1))) * invDet;
    // const vec3 lambda = vec3(1.f - lambda_yz[0] - lambda_yz[1], lambda_yz);
    // ret.lambda = lambda;

    // Convert the gradients from NDC units into pixel units so later derivative
    // use lines up with screen-space neighborhoods.
    ret.ddx *= (2.0f/winSize.x);
    ret.ddy *= (2.0f/winSize.y);
    ddxSum  *= (2.0f/winSize.x);
    ddySum  *= (2.0f/winSize.y);

    // This part fixes the derivatives error happening for the projected triangles.
    // Instead of calculating the derivatives constantly across the 2D triangle we use a projected version
    // of the gradients, this is more accurate and closely matches GPU raster behavior.
    // Final gradient equation: ddx = (((lambda/w) + ddx) / (w+|ddx|)) - lambda

    // Evaluate neighboring reciprocal-w values used to project the raw
    // barycentric gradients onto the perspective-correct surface.
    float interpW_ddx = 1.0f / (interpInvW + ddxSum);
    float interpW_ddy = 1.0f / (interpInvW + ddySum);

    // Project the raw gradients through perspective so their behavior matches
    // hardware-style post-projection attribute interpolation.
    ret.ddx = interpW_ddx*(ret.lambda*interpInvW + ret.ddx) - ret.lambda;
    ret.ddy = interpW_ddy*(ret.lambda*interpInvW + ret.ddy) - ret.lambda;

    return ret;
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

#ifdef VISIBUF_USE_AFFINE_OPAQUE_INTERPOLATION
    // Match the legacy opaque path exactly by shading from affine screen-space
    // barycentrics instead of the reconstructed perspective-correct basis.
    cudarf::Vec2f fragWindow = make_vec2f(x + 0.5f, y + 0.5f);
    baryLambda = compute_bary(tri, fragWindow);
#else
    glm::vec2 windowSize((float)pipe->windowWidth, (float)pipe->windowHeight);

    // Reconstruct the clip-space inputs expected by compute_barys() from the
    // stored screen-space vertex positions plus per-vertex 1/w.
    auto make_clip_pos = [windowSize](const float2 &screenPos, float invW) {
        float w = 1.0f / invW;
        glm::vec2 ndc = 2.0f * glm::vec2(screenPos.x, screenPos.y) / windowSize - glm::vec2(1.0f, 1.0f);
        return glm::vec4(ndc * w, 0.0f, w);
    };

    glm::vec4 P0 = make_clip_pos(tri.sP0, tri.w_rcp.x);
    glm::vec4 P1 = make_clip_pos(tri.sP1, tri.w_rcp.y);
    glm::vec4 P2 = make_clip_pos(tri.sP2, tri.w_rcp.z);
    glm::vec2 ndc = 2.0f * glm::vec2(x + 0.5f, y + 0.5f) / windowSize - glm::vec2(1.0f, 1.0f);

    Barycentric bary = compute_barys(P0, P1, P2, ndc, windowSize);
    baryLambda = make_vec3f(bary.lambda.x, bary.lambda.y, bary.lambda.z);
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
            ? shade_fragment<cudarf::SHADER_TYPE_UNLIT, true, false>(pipe, tri, baryLambda, frag)
            : shade_fragment<cudarf::SHADER_TYPE_UNLIT, false, false>(pipe, tri, baryLambda, frag);
    }
    else {
        bool withClearcoat = material.clearcoatFactor > 0.0f;
        if (withTexturing) {
            shadedColor = withClearcoat
                ? shade_fragment<cudarf::SHADER_TYPE_PBR, true, true>(pipe, tri, baryLambda, frag)
                : shade_fragment<cudarf::SHADER_TYPE_PBR, true, false>(pipe, tri, baryLambda, frag);
        }
        else {
            shadedColor = withClearcoat
                ? shade_fragment<cudarf::SHADER_TYPE_PBR, false, true>(pipe, tri, baryLambda, frag)
                : shade_fragment<cudarf::SHADER_TYPE_PBR, false, false>(pipe, tri, baryLambda, frag);
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
