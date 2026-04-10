
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
