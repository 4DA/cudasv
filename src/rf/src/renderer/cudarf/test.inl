#ifndef CUDARF_TEST_INL
#define CUDARF_TEST_INL

#ifndef NDEBUG

void test_bin_output(const cudarf::pipe::Ctx &desc,
                     const cudarf::rast::PipeScratchContext &scratch,
                     uint32_t numTriangles,
                     const cudarf::pipe::Atomics &pipe_atomics,
                     cudaStream_t cuStream)
{
    const cudarf::rast::BinTilerCtx &ctx = scratch.binCtx;

    const int32_t binTotalL = CUDARF_MAXBINS_SQR * CUDARF_BIN_STREAMS_SIZE;
    const int32_t binSegDataL = desc.internalBufs.maxBinSegs * CUDARF_BIN_SEG_SIZE;

    auto binTotal = std::unique_ptr<int32_t[]>(new int32_t[binTotalL]);
    auto binSegData = std::unique_ptr<int32_t[]>(new int32_t[binSegDataL]);
    auto binSegNext = std::unique_ptr<int32_t[]>(new int32_t[desc.internalBufs.maxBinSegs]);
    auto binSegCount = std::unique_ptr<int32_t[]>(new int32_t[desc.internalBufs.maxBinSegs]);
    auto binFirstSeg = std::unique_ptr<int32_t[]>(new int32_t[binTotalL]);
    auto triBuf = std::unique_ptr<cudarf::rast::Triangle[]>(new cudarf::rast::Triangle[numTriangles]);

    CUDA_CHK(cudaMemcpyAsync(binTotal.get(), ctx.binTotal,
                             binTotalL * sizeof(int32_t),
                             cudaMemcpyDeviceToHost, cuStream));
    CUDA_CHK(cudaMemcpyAsync(binSegData.get(), ctx.binSegData,
                             binSegDataL * sizeof(int32_t),
                             cudaMemcpyDeviceToHost, cuStream));
    CUDA_CHK(cudaMemcpyAsync(binSegNext.get(), ctx.binSegNext,
                             desc.internalBufs.maxBinSegs * sizeof(int32_t),
                             cudaMemcpyDeviceToHost, cuStream));
    CUDA_CHK(cudaMemcpyAsync(binSegCount.get(), ctx.binSegCount,
                             desc.internalBufs.maxBinSegs * sizeof(int32_t),
                             cudaMemcpyDeviceToHost, cuStream));
    CUDA_CHK(cudaMemcpyAsync(binFirstSeg.get(), ctx.binFirstSeg,
                             binTotalL * sizeof(int32_t),
                             cudaMemcpyDeviceToHost, cuStream));
    CUDA_CHK(cudaMemcpyAsync(triBuf.get(), scratch.tris,
                             numTriangles * sizeof(cudarf::rast::Triangle),
                             cudaMemcpyDeviceToHost, cuStream));

    cudaStreamSynchronize(cuStream);

    unsigned int bin_prim_count = 0;

    int SM_count[CUDARF_BIN_STREAMS_SIZE];
    memset(SM_count, 0, CUDARF_BIN_STREAMS_SIZE * sizeof(int));

    int totalCount[CUDARF_BIN_COUNT][CUDARF_BIN_COUNT];
    memset(totalCount, 0, CUDARF_BIN_COUNT * CUDARF_BIN_COUNT * sizeof(int));

    for (int32_t i = 0; i < CUDARF_MAXBINS_SQR; i++) {
        for (int SM = 0; SM < CUDARF_BIN_STREAMS_SIZE; SM++) {
            const int32_t triCount = binTotal[(i << CUDARF_BIN_STREAMS_LOG2) + SM];
            if (triCount > 0) {
                bin_prim_count += triCount;
                SM_count[SM] += triCount;
            }
        }
    }

    for (int SM = 0; SM < CUDARF_BIN_STREAMS_SIZE; SM++) {
        if (SM_count[SM] == 0) { continue; }
        for (int32_t y = CUDARF_BIN_COUNT - 1; y >= 0; y--) {
            for (int32_t x = 0; x < CUDARF_BIN_COUNT; x++) {
                const int bin = x + y * CUDARF_BIN_COUNT;
                totalCount[y][x] += binTotal[(bin << CUDARF_BIN_STREAMS_LOG2) + SM];
            }
        }
    }

    printf("\n");
    printf("bin tiler input triangles: %u, output triangles: %u, subtris: %u, binSegs: %u (max: %d)\n",
           pipe_atomics.subtris_count, bin_prim_count, pipe_atomics.subtris_count,
           pipe_atomics.numBinSegs, desc.internalBufs.maxBinSegs);

    for (int i = 0; i < CUDARF_BIN_STREAMS_SIZE; i++) {
        if (SM_count[i] > 0) {
            printf("SM%2d: %d | ", i, SM_count[i]);
        }
    }
    printf("\n");

    for (int32_t y = CUDARF_BIN_COUNT - 1; y >= 0; y--) {
        for (int32_t x = 0; x < CUDARF_BIN_COUNT; x++) {
            const int tris = totalCount[y][x];
            if (tris > 0) {
                printf(" %4d ", tris);
            } else {
                printf(" .... ");
            }
        }
        printf("\n\n");
    }
    printf("\n");

    int totalOutside = 0;

    for (int binIdx = 0; binIdx < ctx.numBins; binIdx++) {
        const int binX = binIdx % ctx.binsX;
        const int binY = binIdx / ctx.binsY;
        std::vector<int32_t> binTris;

        int32_t streamSeg[CUDARF_BIN_STREAMS_SIZE];
        for (int i = 0; i < CUDARF_BIN_STREAMS_SIZE; i++) {
            streamSeg[i] = binFirstSeg[binIdx * CUDARF_BIN_STREAMS_SIZE + i];
        }

        for (;;) {
            int32_t minStream = -1;
            int32_t minIdx = std::numeric_limits<int32_t>::max();

            for (int i = 0; i < CUDARF_BIN_STREAMS_SIZE; i++) {
                if (streamSeg[i] == -1) { continue; }
                if (binSegData[streamSeg[i] * CUDARF_BIN_SEG_SIZE] < minIdx) {
                    minIdx = binSegData[streamSeg[i] * CUDARF_BIN_SEG_SIZE];
                    minStream = i;
                }
            }

            if (minStream == -1) { break; }

            const int segIdx = streamSeg[minStream];
            streamSeg[minStream] = binSegNext[segIdx];
            for (int i = 0; i < binSegCount[segIdx]; i++) {
                binTris.push_back(binSegData[segIdx * CUDARF_BIN_SEG_SIZE + i]);
            }
        }

        const float2 center = make_float2((binX + 0.5f) * ctx.binW,
                                          (binY + 0.5f) * ctx.binH);
        const float2 half = make_float2(ctx.binW * 0.5f, ctx.binH * 0.5f);

        for (int32_t triIdx: binTris) {
            assert(triIdx >= 0);
            assert(static_cast<uint32_t>(triIdx) < numTriangles);

            const cudarf::rast::Triangle &tri = triBuf[triIdx];
            if (tri.flo.x >= center.x + half.x || tri.flo.y >= center.y + half.y ||
                tri.fhi.x <= center.x - half.x || tri.fhi.y <= center.y - half.y) {
                totalOutside++;
                continue;
            }

            if (disjoint(tri.sP0 - center, tri.sP1 - center, tri.sP2 - center, half)) {
                totalOutside++;
            }
        }
    }

    printf("number of triangles outside bin blocks: %d\n", totalOutside);
}

__global__ void __launch_bounds__(CUDARF_COARSE_WARPS * 32, 1) test_sorting_wrapper(int32_t *values, int SZ)
{
    sort_shared(values, SZ);
}

void test_sorting_driver(unsigned int numMPS)
{
    dim3 blockSize(32, CUDARF_COARSE_WARPS);
    dim3 gridSize(1, 1);

    const unsigned int SZ = 256;
    int32_t testArray[SZ];

    for (unsigned int i = 0; i < SZ; i++) {
        testArray[i] = i;
    }

    std::random_shuffle(std::begin(testArray), std::end(testArray));

    printf("\nsrc array:\n");
    for (unsigned int i = 0; i < SZ; i++) {
        printf("%d ",  testArray[i]);
    }

    int32_t *devArray;
    
    CUDA_CHK(cudaMalloc(&devArray, SZ * sizeof(int32_t)));
    CUDA_CHK(cudaMemcpyAsync(devArray,
                             testArray,
                             SZ * sizeof(int32_t),
                             cudaMemcpyHostToDevice,
                             0));

    test_sorting_wrapper<<<gridSize, blockSize>>>(devArray, SZ);
    CUDA_CHK_KERNEL(0, "test_sorting_driver");

    CUDA_CHK(cudaMemcpyAsync(testArray,
                             devArray,
                             SZ * sizeof(int32_t),
                             cudaMemcpyDeviceToHost,
                             0));

    CUDA_CHK(cudaFree(devArray));

    printf("\ndst array:\n");
    for (unsigned int i = 0; i < SZ; i++) {
        printf("%d ", testArray[i]);
        if (i > 0 && testArray[i-1] > testArray[i]) {
            printf("test_sorting_driver: %d > %d\n",  testArray[i-1], testArray[i]);
        }
    }
    printf("\n");
}

#endif
#endif
