#ifndef CUDARF_TEST_INL
#define CUDARF_TEST_INL

#ifdef TEST_TILER_BIN

void test_bin_output(const cudarf::pipe::Ctx &desc, const cudarf::rast::PipeParams &pipe, const cudarf::pipe::Atomics &pipe_atomics, cudaStream_t cuStream)
{
    
    const cudarf::rast::BinTilerCtx &ctx = pipe.binCtx;

    int32_t binTotalL = CUDARF_MAXBINS_SQR * CUDARF_BIN_STREAMS_SIZE;
    int32_t binSegDataL = desc.internalBufs.maxBinSegs * CUDARF_BIN_SEG_SIZE;

    auto binTotal = std::unique_ptr<int32_t[]> (new int32_t[binTotalL]);
    auto binSegData = std::unique_ptr<int32_t[]> (new int32_t[binSegDataL]);
    auto binSegNext = std::unique_ptr<int32_t[]> (new int32_t[desc.internalBufs.maxBinSegs]);
    auto binSegCount = std::unique_ptr<int32_t[]> (new int32_t[desc.internalBufs.maxBinSegs]);
    auto binFirstSeg = std::unique_ptr<int32_t[]> (new int32_t[binTotalL]);

    std::unique_ptr<cudarf::rast::Triangle[]> triBuf(new cudarf::rast::Triangle[pipe.numTriangles]);


    CUDA_CHK(cudaMemcpyAsync(binTotal.get(), desc.dev_binTotal.get(),
                        binTotalL * sizeof(int32_t), cudaMemcpyDeviceToHost, cuStream));

    CUDA_CHK(cudaMemcpyAsync(binSegData.get(), desc.internalBufs.dev_binSegData,
                        binSegDataL * sizeof(int32_t), cudaMemcpyDeviceToHost, cuStream));

    CUDA_CHK(cudaMemcpyAsync(binSegNext.get(), desc.internalBufs.dev_binSegNext,
                        desc.internalBufs.maxBinSegs * sizeof(int32_t), cudaMemcpyDeviceToHost, cuStream));

    CUDA_CHK(cudaMemcpyAsync(binSegCount.get(), desc.internalBufs.dev_binSegCount,
                        desc.internalBufs.maxBinSegs * sizeof(int32_t), cudaMemcpyDeviceToHost, cuStream));

    CUDA_CHK(cudaMemcpyAsync(binFirstSeg.get(), desc.dev_binFirstSeg.get(),
                        CUDARF_MAXBINS_SQR * CUDARF_BIN_STREAMS_SIZE * sizeof(int32_t),
                        cudaMemcpyDeviceToHost, cuStream));

    CUDA_CHK(cudaMemcpyAsync(triBuf.get(), desc.internalBufs.dev_triangles,
                        pipe.numTriangles * sizeof(cudarf::rast::Triangle), cudaMemcpyDeviceToHost, cuStream));

    cudaStreamSynchronize(cuStream);

    unsigned int bin_prim_count = 0;

    // compute per-SM and total primitive count

    int SM_count[CUDARF_BIN_STREAMS_SIZE];
    memset(SM_count, 0, CUDARF_BIN_STREAMS_SIZE * sizeof(int));

    int totalCount[CUDARF_BIN_COUNT][CUDARF_BIN_COUNT];
    memset(totalCount, 0, CUDARF_BIN_COUNT * CUDARF_BIN_COUNT * sizeof(int));

    for (int32_t i = 0; i < CUDARF_MAXBINS_SQR; i++) {
        for (int SM = 0; SM < CUDARF_BIN_STREAMS_SIZE; SM++) {

            if (binTotal[(i << CUDARF_BIN_STREAMS_LOG2) + SM] > 0) {
                bin_prim_count += binTotal[(i << CUDARF_BIN_STREAMS_LOG2) + SM];
                SM_count[SM] += binTotal[(i << CUDARF_BIN_STREAMS_LOG2) + SM];
            }
        }
    }

    // compute total sum for each bin

    for (int SM = 0; SM < CUDARF_BIN_STREAMS_SIZE; SM++) {
        if (SM_count[SM] == 0) {continue;}
        // printf("bins for SM %d \n", SM);
        for (int32_t i = CUDARF_BIN_COUNT-1; i >= 0; i--) {
            for (int32_t j = 0; j < CUDARF_BIN_COUNT; j++) {
                int bin = (j + i * CUDARF_BIN_COUNT);
                int tris = binTotal[(bin << CUDARF_BIN_STREAMS_LOG2) + SM];
                totalCount[i][j] += tris;
                // if (tris > 0) {
                //     printf(" %4d ", tris);
                // } else {
                //     printf(" .... ");
                // }
            }
            // {printf("\n\n");}
        }
    }
    printf("\n");

    // atomic stats after bin tiler

    printf("bin tiler input triangles: %d, output triangles: %d, subtris: %d, binSegs: %d (max: %d)\n",
           pipe_atomics.subtris_count, bin_prim_count, pipe_atomics.subtris_count, pipe_atomics.numBinSegs,
           desc.internalBufs.maxBinSegs);

    for (int i = 0; i < CUDARF_BIN_STREAMS_SIZE; i++) {
        if (SM_count[i] > 0) {
            printf ("SM%2d: %d | ", i, SM_count[i]);
        }
    }
    printf("\n");

    for (int32_t i = CUDARF_BIN_COUNT-1; i >= 0; i--) {
        for (int32_t j = 0; j < CUDARF_BIN_COUNT; j++) {
            int tris = totalCount[i][j];
            if (tris > 0) {
                printf(" %4d ", tris);
            } else {
                printf(" .... ");
            }
        }
        {printf("\n\n");}
    }
    printf("\n");

    // DEBUG: test to check that bin tiler has processed all triangles
    // will flag if any triangle is culled/clipped
    // CUDA_CHK(cudaMemcpyAsync(&bin_tiler_mask, pipe.dbgbuf,
    //                 triangle_count * sizeof(int32_t), cudaMemcpyDeviceToHost));

    // for (uint i = 0; i < triangle_count; i++) {
    //     if (bin_tiler_mask[i] != 1) {
    //         printf("WARNING triangle %d binned: %d\n", i, bin_tiler_mask[i]);
    //     }
    // }

    int totalOutside = 0;

    for (int binIdx = 0; binIdx < ctx.numBins; binIdx++)
    {
        int binX = binIdx % ctx.binsX;
        int binY = binIdx / ctx.binsY;
        std::vector<int32_t> binTris;

        int32_t streamSeg[CUDARF_BIN_STREAMS_SIZE];
        for (int i = 0; i < CUDARF_BIN_STREAMS_SIZE; i++)
            streamSeg[i] = binFirstSeg[binIdx * CUDARF_BIN_STREAMS_SIZE + i];

        // build sorted triangle list for bin
        while(true) {
            int32_t minStream = -1;
            int32_t minIdx = std::numeric_limits<int32_t>::max();

            // get stream with smallest triangle index
            for (int i = 0; i < CUDARF_BIN_STREAMS_SIZE; i++) {
                if (streamSeg[i] == -1) { continue;}

                if (binSegData[streamSeg[i] * CUDARF_BIN_SEG_SIZE] < minIdx) {
                    minIdx = binSegData[streamSeg[i] * CUDARF_BIN_SEG_SIZE];
                    minStream = i;
                }
            }

            if (minStream == -1) {break;}

            // read all triangles from minStream

            int segIdx = streamSeg[minStream];
            streamSeg[minStream] = binSegNext[segIdx];
            for (int i = 0; i < binSegCount[segIdx]; i++) {
				binTris.push_back(binSegData[segIdx * CUDARF_BIN_SEG_SIZE + i]);
            }
        }

        // check that each triangle intersects the degisnated bin

        for (unsigned int i = 0; i < binTris.size(); i++) {
            const cudarf::rast::Triangle &tri = triBuf[binTris[i]];
            float2 center = (make_float2((binX + 0.5) / ctx.binsX * pipe.windowWidth,
                                         (binY + 0.5) / ctx.binsY * pipe.windowHeight));

            // TODO compute only once
            float2 half = make_float2(pipe.windowWidth / ctx.binsX / 2.0,
                                      pipe.windowHeight / ctx.binsY / 2.0);

            // [0;1] normal ized vertex positions
            float2 vN0 = make_float2((tri.P0.x + 1.0) / 2.0,
                                     (tri.P0.y + 1.0) / 2.0);

            float2 vN1 = make_float2((tri.P1.x + 1.0) / 2.0,
                                     (tri.P1.y + 1.0) / 2.0);

            float2 vN2 = make_float2((tri.P2.x + 1.0) / 2.0,
                                     (tri.P2.y + 1.0) / 2.0);

            float2 flo = vN0 + fminf(make_float2(0.0), fminf(vN1 - vN0, vN2 - vN0));
            float2 fhi = vN0 + fmaxf(make_float2(0.0), fmaxf(vN1 - vN0, vN2 - vN0));

            flo.x = pipe.windowWidth  * flo.x;
            flo.y = pipe.windowHeight * flo.y;
            fhi.x = pipe.windowWidth  * fhi.x;
            fhi.y = pipe.windowHeight * fhi.y;

            float2 fv0 = vN0 * make_float2(pipe.windowWidth, pipe.windowHeight) - center;
            float2 fv1 = vN1 * make_float2(pipe.windowWidth, pipe.windowHeight) - center;
            float2 fv2 = vN2 * make_float2(pipe.windowWidth, pipe.windowHeight) - center;

            // Outside AABB => skip.
            if (flo.x >= center.x + half.x || flo.y >= center.y + half.y ||
                fhi.x <= center.x - half.x || fhi.y <= center.y - half.y)
            {
                // printf("[tiler_bin_test] tri[id:%d] AABB is ouside binsc (%d,%d) | ", binTris[i], binX, binY);
                // dump_triag("triangle data:", tri);
                // printf("\n");
                totalOutside++;
                continue;
            }

            if (disjoint(fv0, fv1, fv2, half))
            {
                totalOutside++;
                continue;
            }
        }
    }

    printf("number of triangles outside 2x2 blocks: %d\n", totalOutside);
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
    
    CUDA_CHK(cudarf_cuda_malloc(&devArray, SZ * sizeof(int32_t)));
    CUDA_CHK(cudaMemcpyAsync(devArray,
                             testArray,
                             SZ * sizeof(int32_t),
                             cudaMemcpyHostToDevice,
                             0));

    test_sorting_wrapper<<<gridSize, blockSize>>>(devArray, SZ);
    CUDA_CHK_ERROR("test_sorting_driver");

    CUDA_CHK(cudaMemcpyAsync(testArray,
                             devArray,
                             SZ * sizeof(int32_t),
                             cudaMemcpyDeviceToHost,
                             0));

    CUDA_CHK(cudarf_cuda_free(devArray));

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
