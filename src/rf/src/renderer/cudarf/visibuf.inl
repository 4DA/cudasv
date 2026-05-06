__global__
void visibuf_build_xy_lists(const cudarf::rast::PipeParams *pipe,
                            const cudarf::visibuf::GeomOutput *geomFb,
                            cudarf::pipe::Atomics *g_atomics,
                            cudarf::visibuf::XYCommand *xyCommands)
{
    __shared__ int blockFragCount; // block local visible count
    __shared__ int globalOffset;

    int visible = 1;

    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= pipe->windowWidth || y >= pipe->windowHeight) {visible = 0;}

    int inIdx = x + (y * pipe->windowWidth);

    bool isT0 = (threadIdx.x == 0 && threadIdx.y == 0);

    // initialize shared memory
    if (isT0) {
        blockFragCount = 0;
    }

    __syncthreads();

    cudarf::visibuf::GeomOutput info;
    uint32_t matId = 0;

    if (visible) {
        info = geomFb[inIdx];

        if (info.drawPacketId == 0xFFFFFFFFu || info.triId == 0xFFFFFFFFu) {visible = 0;}
        if (info.drawPacketId >= CUDARF_MAX_DRAW_PACKETS) {visible = 0;}

        if (visible) {
            matId = pipe->drawPacketMaterials[info.drawPacketId];
        }

        if(matId == 0xFFFFFFFFu) {visible = 0;}
    }

    int localOffset = -1;

    if (visible) {
        localOffset = atomicAdd(&blockFragCount, visible);
    }

    __syncthreads();

    if (isT0) {
        globalOffset = atomicAdd(&g_atomics->visibuf.totalVisibleFrags, blockFragCount);
    }

    __syncthreads();

    if (visible) {
        uint32_t cmdIdx = globalOffset + localOffset;
        xyCommands[cmdIdx] = {x, y, matId};
    }
}

__global__
void visibuf_material_pass(const cudarf::rast::PipeParams *pipe,
                           const cudarf::visibuf::GeomOutput *geomFb,
                           cudarf::pipe::Atomics *g_atomics,
                           const cudarf::visibuf::XYCommand *xyCommands,
                           cudarf::Framebuffer fb)
{
    unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ty = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int i = tx + (ty * pipe->windowWidth);

    if (tx >= pipe->windowWidth || ty >= pipe->windowHeight) {return;}
    if (i >= g_atomics->visibuf.totalVisibleFrags) {return;}

    auto [x, y, matId] = xyCommands[i];

    int inIdx = x + (y * pipe->windowWidth);
    cudarf::visibuf::GeomOutput info = geomFb[inIdx];

    int triId = info.triId;
    assert(triId != 0xFFFFFFFFu);

    const cudarf::rast::Triangle &tri = pipe->tris[triId];

    const cudarf::Material &material = pipe->materials[matId];

    cudarf::rast::Fragment frag;


    bool withTexturing =
        material.albedoTex.textureObject ||
        material.emissiveTex.textureObject ||
        material.occlusionTex.textureObject ||
        material.metRoughTex.textureObject;

    cudarf::Color shadedColor;

    if (material.type == cudarf::SHADER_TYPE_UNLIT) {
        shadedColor = withTexturing
            ? shade_fragment<cudarf::SHADER_TYPE_UNLIT, true, false>(
                pipe, tri, x, y, frag)
            : shade_fragment<cudarf::SHADER_TYPE_UNLIT, false, false>(
                pipe, tri, x, y, frag);
    }
    else {
        bool withClearcoat = material.clearcoatFactor > 0.0f;
        if (withTexturing) {
            shadedColor = withClearcoat
                ? shade_fragment<cudarf::SHADER_TYPE_PBR, true, true>(
                    pipe, tri, x, y, frag)
                : shade_fragment<cudarf::SHADER_TYPE_PBR, true, false>(
                    pipe, tri, x, y, frag);
        }
        else {
            shadedColor = withClearcoat
                ? shade_fragment<cudarf::SHADER_TYPE_PBR, false, true>(
                    pipe, tri, x, y, frag)
                : shade_fragment<cudarf::SHADER_TYPE_PBR, false, false>(
                    pipe, tri, x, y, frag);
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
