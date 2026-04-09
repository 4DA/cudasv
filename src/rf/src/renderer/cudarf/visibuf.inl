
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
