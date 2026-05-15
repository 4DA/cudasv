#ifndef CUDARF_TILER_COARSE_INL
#define CUDARF_TILER_COARSE_INL

template <class T>
__device__ __inline__ void swp(T *v1, T *v2) {
    int tmp = *v1;
    *v1 = *v2;
    *v2 = tmp;
}

template <class T> __device__ __inline__ void sort_shared(T *ptr, int numItems)
{
    int base = (threadIdx.x + threadIdx.y * blockDim.x) * 2;
    bool isActive = (base < numItems - 1);
    int thrInBlock  = threadIdx.x + threadIdx.y * 32;

    for (int iter = 0; iter < numItems; iter += 2) {
        // every even-indexed element is compared to the next odd
        if (isActive) {
            if (ptr[base + 0] > ptr[base + 1]) {
                swp(&ptr[base + 0], &ptr[base + 1]);
            }
        }
        __syncthreads();

        // every odd-indexed element is compared to the next even
        if (isActive && base < numItems - 2) {
            if (ptr[base + 1] > ptr[base + 2]) {
                swp(&ptr[base + 1], &ptr[base + 2]);
            }
        }
        __syncthreads();
    }
}


/**
   computation scheme
   ------------------
   blockIdx.x  ∈ [0; SMPCount]
   threadIdx.x ∈ [0; 31], so it is individual warp lane (threadIdx.x == ptx_lane_id)
   threadIdx.y ∈ [0; 15], so it is warp index
 */

template<bool TWithBlending>
__global__ __launch_bounds__(CUDARF_COARSE_WARPS * 32, 1)
void tilerCoarse(const cudarf::rast::PipeParams *pipe, cudarf::pipe::Atomics *atomics)
{
    const cudarf::rast::PipeSubmissionContext *sub = pipe->submission;
    const cudarf::rast::PipeScratchContext &scratch = pipe->scratch;
    const cudarf::rast::BinTilerCtx &binCtx = scratch.binCtx;

    // Common
    // --
    __shared__ volatile uint32_t s_workCounter;
    __shared__ volatile uint32_t s_scanTemp [CUDARF_COARSE_WARPS][48]; // 3KB

    // Input
    // --
    __shared__ volatile uint32_t s_binOrder [CUDARF_MAXBINS_SQR]; // 1KB
    __shared__ volatile int32_t s_binStreamCurrSeg  [CUDARF_BIN_STREAMS_SIZE];              // 0KB
    __shared__ volatile int32_t s_binStreamFirstTri [CUDARF_BIN_STREAMS_SIZE];              // 0KB
    __shared__ volatile uint32_t s_binStreamSelectedOfs;
    __shared__ volatile uint32_t s_binStreamSelectedSize;

    __shared__ volatile int32_t s_triQueue          [CUDARF_COARSE_QUEUE_SIZE];             // 4KB
    __shared__ volatile int32_t s_tileStreamCurrOfs [CUDARF_BIN_SQR];                       // 1KB
    __shared__ volatile int32_t s_triQueueWritePos;

    // Output
    // --


    // s_warpEmitMask[i][j][k] == binary 1 if for warp i thread with
    // lane_id=k stores triangle to tile j
    // 16KB, +1 to avoid bank collisions
    __shared__ volatile uint32_t s_warpEmitMask    [CUDARF_COARSE_WARPS][CUDARF_MAX_TILES + 1];

    // s_warpEmitPrefixSum[i][j] = cumulative sum (for tile j) of triangles over warps for
    // warp i
    // 16KB, +1 to avoid bank collisions
    // __shared__ volatile uint32_t s_warpEmitPrefixSum [CUDARF_COARSE_WARPS][CUDARF_MAX_TILES + 1];

    // Pointers
    // --
    const int32_t*              binFirstSeg     = (const int32_t*)binCtx.binFirstSeg;
    const int32_t*              binTotal        = (const int32_t*)binCtx.binTotal;
    const int32_t*              binSegData      = (const int32_t*)binCtx.binSegData;
    const int32_t*              binSegNext      = (const int32_t*)binCtx.binSegNext;
    const int32_t*              binSegCount     = (const int32_t*)binCtx.binSegCount;

    // Constants
    // --

    if (atomics->subtris_count > sub->maxSubtris || atomics->numBinSegs > scratch.binCtx.maxBinSegs)
        return;

    int tileLog     = CUDARF_TILE_LOG2 + CUDARF_SUBPIXEL_LOG2;
    int thrInBlock  = threadIdx.x + threadIdx.y * 32;

    // int emitShift   = CUDARF_BIN_LOG2 * 2 + 5; // We scan ((numEmits <<
    // emitShift) | numAllocs) over tiles.

    // Initialize sharedmem arrays.
    // s_tileEmitPrefixSum[0] = 0;
    // s_tileAllocPrefixSum[0] = 0;
    s_scanTemp[threadIdx.y][threadIdx.x] = 0;

    // sort bins in descending order of triangle count
    // --
    // store bin idx in lower bits and inverted count in upper bits to sort in descending order
    // count = ~s_binOrder[i] >> (CUDARF_MAXBINS_LOG2 * 2)
    // binIdx = s_binOrder[i] & (CUDARF_MAXBINS_SQR - 1)

    // profiling stuff
    // --
    // long long int clockIn;

    for (int binIdx = thrInBlock; binIdx < scratch.binCtx.numBins; binIdx += CUDARF_COARSE_WARPS * 32)
    {
        int count = 0;
        for (int i = 0; i < CUDARF_BIN_STREAMS_SIZE; i++)
            count += binTotal[(binIdx << CUDARF_BIN_STREAMS_LOG2) + i];

        s_binOrder[binIdx] = (~count << (CUDARF_MAXBINS_LOG2 * 2) | binIdx);
    }
    __syncthreads();

    sort_shared(s_binOrder, CUDARF_MAXBINS_SQR);

    int iter = -1;

    // Process each bin by one block.
    for (;;) {
        iter++;

        if (thrInBlock == 0) {
            s_workCounter = atomicAdd(&atomics->coarseCounter, 1);
        }

        __syncthreads();

        int workCounter = s_workCounter;
        if (workCounter >= scratch.binCtx.numBins)
        {
            break;
        }

        uint32_t binOrder = s_binOrder[workCounter];
        bool binEmpty = ((~binOrder >> (CUDARF_MAXBINS_LOG2 * 2)) == 0);
        int binIdx = binOrder & (CUDARF_MAXBINS_SQR - 1); // we don't need bin
                                                        // count

        if (binEmpty)
        {
            break;
        }


        // Initialize input/output streams.

        int triQueueWritePos = 0;
        int triQueueReadPos = 0;

        if (thrInBlock < CUDARF_BIN_STREAMS_SIZE) {
            int segIdx = binFirstSeg[(binIdx << CUDARF_BIN_STREAMS_LOG2) + thrInBlock];
            s_binStreamCurrSeg[thrInBlock] = segIdx;
            s_binStreamFirstTri[thrInBlock] = (segIdx == -1) ? ~0u : binSegData[segIdx << CUDARF_BIN_SEG_LOG2];
        }

        for (int tileInBin = CUDARF_COARSE_WARPS * 32 - 1 - thrInBlock; tileInBin < CUDARF_BIN_SQR; tileInBin += CUDARF_COARSE_WARPS * 32) {
            s_tileStreamCurrOfs[tileInBin] = -CUDARF_TILE_SEG_SIZE;
        }

        // Initialize per-bin state.

        int2 tilesInBin = make_int2(scratch.binCtx.binW / CUDARF_TILE_SZ, scratch.binCtx.binH / CUDARF_TILE_SZ);

        int binY = binIdx / scratch.binCtx.binsX;
        int binX = binIdx - binY * scratch.binCtx.binsX;

        // TODO do we need rasterizer center in (0,0)?
        int originX = binX * binCtx.binW;
        int originY = binY * binCtx.binH;

        int originTileX = originX / CUDARF_TILE_SZ;
        int originTileY = originY / CUDARF_TILE_SZ;

        int originTileXMax = originTileX + binCtx.binW / CUDARF_TILE_SZ;
        int originTileYMax = originTileY + binCtx.binH / CUDARF_TILE_SZ;

        int2 tileLo = tilesInBin * make_int2(binX, binY);
        int2 tileHi = tilesInBin * make_int2(binX + 1, binY + 1) - make_int2(1, 1);

        if (!binEmpty) do
        {
            //------------------------------------------------------------------------
            // Merge.
            //------------------------------------------------------------------------

            // Entire block: Not enough triangles => merge and queue segments.
            // NOTE: The bin exit criterion assumes that we queue more triangles than we actually need.

            while (triQueueWritePos - triQueueReadPos <= CUDARF_COARSE_WARPS * 32) {
                // First warp: Choose the segment with the lowest initial triangle index.

                if (thrInBlock < CUDARF_BIN_STREAMS_SIZE) {
                    // Find the stream with the lowest triangle index.
                    // TODO: consider dumb 16 comparisons, instead of
                    // cooperative scan

                    uint32_t firstTri = s_binStreamFirstTri[thrInBlock];
                    uint32_t t = firstTri;
                    volatile uint32_t* p = &s_scanTemp[0][thrInBlock + 16];

                    #if (CUDARF_BIN_STREAMS_SIZE > 1)
                        p[0] = t, t = ::min(t, p[-1]);
                    #endif
                    #if (CUDARF_BIN_STREAMS_SIZE > 2)
                        p[0] = t, t = ::min(t, p[-2]);
                    #endif
                    #if (CUDARF_BIN_STREAMS_SIZE > 4)
                        p[0] = t, t = ::min(t, p[-4]);
                    #endif
                    #if (CUDARF_BIN_STREAMS_SIZE > 8)
                        p[0] = t, t = ::min(t, p[-8]);
                    #endif
                    #if (CUDARF_BIN_STREAMS_SIZE > 16)
                        p[0] = t, t = ::min(t, p[-16]);
                    #endif
                    p[0] = t;

                    // smallest value is stored in last element of scan domain,
                    // use thread corresponding to bin stream with smallest tri
                    // index to pick up input segment

                    // Consume and broadcast.
                    if (s_scanTemp[0][CUDARF_BIN_STREAMS_SIZE - 1 + 16] == firstTri)
                    {
                        int segIdx = s_binStreamCurrSeg[thrInBlock];
                        s_binStreamSelectedOfs = segIdx << CUDARF_BIN_SEG_LOG2;

                        if (segIdx != -1)
                        {
                            int segSize = binSegCount[segIdx];
                            int segNext = binSegNext[segIdx];
                            s_binStreamSelectedSize = segSize;
                            s_triQueueWritePos = triQueueWritePos + segSize;
                            s_binStreamCurrSeg[thrInBlock] = segNext;
                            s_binStreamFirstTri[thrInBlock] =
                                (segNext == -1) ? ~0u : binSegData[segNext << CUDARF_BIN_SEG_LOG2];

                        }
                    }
                }

                // bin stream with lowest tri idx is chosen
                __syncthreads();
                triQueueWritePos = s_triQueueWritePos;
                int segOfs = s_binStreamSelectedOfs;
                if (segOfs < 0) {break;}                // No more segments => break.

                int segSize = s_binStreamSelectedSize;
                __syncthreads();

                // Fetch triangles into the queue. Right now BIN_SEG_SIZE is
                // 512, so this cycle takes only one iteration

                for (int idxInSeg = CUDARF_COARSE_WARPS * 32 - 1 - thrInBlock;
                     idxInSeg < segSize; idxInSeg += CUDARF_COARSE_WARPS * 32)
                {
                    int32_t triIdx = binSegData[segOfs + idxInSeg];
                    s_triQueue[(triQueueWritePos - segSize + idxInSeg) & (CUDARF_COARSE_QUEUE_SIZE - 1)] = triIdx;
                }
            } // while (triQueueWritePos - triQueueReadPos <= CUDARF_COARSE_WARPS * 32)


            // clear emit masks
            for (int maskIdx = thrInBlock; maskIdx < CUDARF_COARSE_WARPS * CUDARF_BIN_SQR; maskIdx += CUDARF_COARSE_WARPS * 32) {
                s_warpEmitMask[maskIdx >> CUDARF_MAX_TILES_LOG2][maskIdx & (CUDARF_MAX_TILES - 1)] = 0;
            }

            __syncthreads();

            //------------------------------------------------------------------------
            // Raster.
            //------------------------------------------------------------------------

            // Triangle per thread: Read from the queue.
            // TODO: consider order in which triangles are writting to tile
            // queue, they must be written in the order as in bin queue

            int triIdx = -1;
            if (triQueueReadPos + thrInBlock < triQueueWritePos)
                triIdx = s_triQueue[(triQueueReadPos + thrInBlock) & (CUDARF_COARSE_QUEUE_SIZE - 1)];

            int2 tilesInBin = make_int2(scratch.binCtx.binW / CUDARF_TILE_SZ, scratch.binCtx.binH / CUDARF_TILE_SZ);

            cudarf::rast::Triangle tri;

            // uint8_t* currPtr = (uint8_t *)&s_warpEmitMask[threadIdx.y][lox + (loy << CUDARF_BIN_LOG2)];
            // int ptrYInc = CUDARF_BIN_SIZE * 4 - (sizex << 2);
            uint32_t maskBit = 1 << threadIdx.x;

            if (triIdx != -1)
            {
                tri = scratch.tris[triIdx];

                int2 triLo = tile_idx_from_coord(&scratch, tri.flo);
                int2 triHi = tile_idx_from_coord(&scratch, tri.fhi);

                triLo = clamp(triLo, tileLo, tileHi) - tileLo;
                triHi = clamp(triHi, tileLo, tileHi) - tileLo;

                uint32_t maskBit = 1 << threadIdx.x;

                // clockIn = clock64();

                for (int32_t x = triLo.x; x <= triHi.x; x++) {
                    for (int32_t y = triLo.y; y <= triHi.y; y++) {
                        int binTile = x + y * tilesInBin.x;
                        int tileId = x + tileLo.x + (y + tileLo.y) * scratch.binCtx.binsX * tilesInBin.x;
                        int res = is_inside(make_int2(x, y), triLo, triHi);

                        if (res) {
                            if (TWithBlending) {
                                atomicOr((uint32_t *) &s_warpEmitMask[threadIdx.y][binTile], maskBit);
                            }
                            else {
                                cudarf::rast::SimpleQueue::push(scratch.tileQHeaders[tileId], scratch.tileQLimit, triIdx);
                            }

                            // DEBUG: for checking that all triangles were consumed
                            // pipe->dbgbuf[triIdx] += 1;
                         }

                    }
                }

                // long long int diffClock = clock64() - clockIn;
            } // if triIdx != 1

            __syncthreads();

            // currently, CUDARF_MAX_TILES is 256, so only one iteration of this
            // cycle is made

            if (TWithBlending)
            for (int tileInBin = thrInBlock; tileInBin < CUDARF_MAX_TILES; tileInBin += CUDARF_COARSE_WARPS * 32)
            {
                int y = tileInBin / tilesInBin.x;
                int x = tileInBin - tilesInBin.x * y;
                int tileId = x + tileLo.x + (y + tileLo.y) * scratch.binCtx.binsX * tilesInBin.x;

                for (int warp = 0; warp < CUDARF_COARSE_WARPS; warp++) {
                    if (s_warpEmitMask[warp][tileInBin]) {
                        for (int i = 0; i < 32; i++) {
                            if (s_warpEmitMask[warp][tileInBin] & (1 << i)) {
                                int tib = i + warp * 32;
                                if (triQueueReadPos + tib < triQueueWritePos) {
                                    int triIdx = s_triQueue[(triQueueReadPos + tib) & (CUDARF_COARSE_QUEUE_SIZE - 1)];
                                    cudarf::rast::SimpleQueue::push_unprotected(scratch.tileQHeaders[tileId], scratch.tileQLimit, triIdx);
                                }
                            }
                        }
                    }
                }
            }

            // Advance queue read pointer.
            // Queue became empty => bin done.
            triQueueReadPos += CUDARF_COARSE_WARPS * 32;

        } while (triQueueReadPos < triQueueWritePos);
    }
}

#endif
