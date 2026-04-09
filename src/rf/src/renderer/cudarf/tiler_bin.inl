// -*- mode: c++ -*-

/* This file uses code from cudarast renderer hosted on google code:
   https://code.google.com/archive/p/cudaraster/
 
   "High-Performance Software Rasterization on GPUs",
   Samuli Laine and Tero Karras,
   Proc. High-Performance Graphics 2011
   http://www.tml.tkk.fi/~samuli/publications/laine2011hpg_paper.pdf
 
   Original code is licensed under New BSD License
 */   

/*******************************************************************************
 * Copyright (c) 2009-2011, NVIDIA Corporation
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of NVIDIA Corporation nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *******************************************************************************/  

/* Separating axis method
   --
   Each perp(edge) of (T)riangle is potentinal separating axis.
   Given axis `a': [-r, r] = project(AABB, a)
   (p0, p1, p2) = project(T, a)

   If [−r, r] and min(p0, p1, p2 ), max(p0, p1, p2 ) are
   disjoint for axis `a', then `a' is a separating axis
   and T and AABB do not overlap. Because box is at
   origin we can compare relative lengths:
   min(p0, p1, p2) > r || max(p0, p1, p2) < -r
   => `a' is separating
*/

// box is at center of origin
__host__ __device__ bool disjoint(float2 fv0, float2 fv1, float2 fv2, float2 half)
{
    float2 edge01 = fv0 - fv1;
    float2 edge02 = fv0 - fv2;
    float2 edge12 = fv1 - fv2;

    float2 ax01 = normalize(perp(edge01));
    float2 ax02 = normalize(perp(edge02));
    float2 ax12 = normalize(perp(edge12));

    // projection radius of box on e01
    float r01 = abs(ax01.x) * half.x + abs(ax01.y) * half.y;
    float r02 = abs(ax02.x) * half.x + abs(ax02.y) * half.y;
    float r12 = abs(ax12.x) * half.x + abs(ax12.y) * half.y;

    // projections of triangle vertices on axis
    float p0_01 = dot(fv0, ax01);
    float p1_01 = dot(fv1, ax01);
    float p2_01 = dot(fv2, ax01);

    float p0_02 = dot(fv0, ax02);
    float p1_02 = dot(fv1, ax02);
    float p2_02 = dot(fv2, ax02);

    float p0_12 = dot(fv0, ax12);
    float p1_12 = dot(fv1, ax12);
    float p2_12 = dot(fv2, ax12);

    bool sep01 = false, sep02 = false, sep12 = false;
    if (min(p0_01, p2_01) > r01 || max(p0_01, p2_01) < -r01) sep01 = true;
    if (min(p1_02, p2_02) > r02 || max(p1_02, p2_02) < -r02) sep02 = true;
    if (min(p0_12, p1_12) > r12 || max(p0_12, p1_12) < -r12) sep12 = true;

    return (sep01 || sep02 || sep12);

}

/** CUDARF_STAGE: bin tiling

    cudarf::rast::Triangles are binned into 16x16 tiles occupying output window. It follows
    that for current implementation window dimensions must be divisible by
    TILE_SIZE * 16 = 128.

    Memory is allocated dynamically to queues when they are written to
    (requires a global atomic op). To minimize number of allocations queues
    consist of segments which are equally-sized contiguous memory ranges and
    queue is a linked list of references to these segments. Memory allocation
    is necessary when the last segment of queue becomes full.
 */
__global__ void __launch_bounds__(CUDARF_BIN_WARPS * 32, 1)
    tiler_bin(const cudarf::rast::PipeParams *pipe, cudarf::pipe::Atomics *g_atomics, cudarf::rast::Triangle *)
{
    // service array for different broadcast operations
    __shared__ volatile uint32_t s_broadcast [CUDARF_BIN_WARPS + 16];

    // output offset for bin:
    // upper bits store segment number
    // lower 9 bits store offset inside segment(max CUDARF_BIN_SEG_SIZE-1),
    __shared__ volatile int32_t s_outOfs     [CUDARF_MAXBINS_SQR];

    __shared__ volatile int32_t s_outTotal   [CUDARF_MAXBINS_SQR];
    __shared__ volatile int32_t s_overIndex  [CUDARF_MAXBINS_SQR];

    // triangle ring buffer
	__shared__ volatile int32_t s_triBuf     [CUDARF_BIN_WARPS*32*4];

    __shared__ volatile uint32_t s_batchPos;
    __shared__ volatile uint32_t s_bufCount;

    // For each warp and each bin we store 32-bit mask. Each bit corresponds to
    // thread in warp lane whose assigned triangle overlaps that bin
    // (+1 to avoid bank collisions)
	__shared__ volatile int32_t s_outMask [CUDARF_BIN_WARPS][CUDARF_MAXBINS_SQR + 1];

    // cumulative contribution (number of triangles) of each bin across all warps in block
    // (+1 to avoid bank collisions)
	__shared__ volatile int32_t s_outCount [CUDARF_BIN_WARPS][CUDARF_MAXBINS_SQR + 1];

    // total number of overflown threads within block
    __shared__ volatile uint32_t s_overTotal;

    // number of segments allocated within block
    __shared__ volatile uint32_t s_allocBase;

    const cudarf::rast::BinTilerCtx &ctx = pipe->binCtx;

    int32_t*                    binFirstSeg     = (int32_t*)ctx.binFirstSeg;
    int32_t*                    binTotal        = (int32_t*)ctx.binTotal;
    int32_t*                    binSegData      = (int32_t*)ctx.binSegData;
    int32_t*                    binSegNext      = (int32_t*)ctx.binSegNext;
    int32_t*                    binSegCount	    = (int32_t*)ctx.binSegCount;

    const cudarf::rast::Triangle* triangles = pipe->tris;

    int32_t thrInBlock = threadIdx.x + threadIdx.y * 32;
	int32_t batchPos = 0;

    // first 16 elements of s_broadcast are always zero
    // why?
    if (thrInBlock < 16) {
        s_broadcast[thrInBlock] = 0;
    }

	// initialize output linked lists and offsets
    if (thrInBlock < ctx.numBins)
    {
        binFirstSeg[(thrInBlock << CUDARF_BIN_STREAMS_LOG2) + blockIdx.x] = -1;
        s_outOfs[thrInBlock] = -CUDARF_BIN_SEG_SIZE;
        s_outTotal[thrInBlock] = 0;
    }

	// repeat until done
    for(;;) {
        if (thrInBlock == 0) {
            // .. Global memory access
            s_batchPos = atomicAdd(&g_atomics->bin_counter, ctx.binBatchSize);
        }

		__syncthreads();

		batchPos = s_batchPos;

		// all batches done?
        if (batchPos >= pipe->numTriangles) {
			break;
        }

        // per-thread state
        int bufIndex = 0;
        int	bufCount = 0;
        int batchEnd = ::min(batchPos + ctx.binBatchSize, pipe->numTriangles);

        do {

            // read more triangles
            while (bufCount < CUDARF_BIN_WARPS*32 && batchPos < batchEnd) {
                int triIdx = batchPos + thrInBlock;
                int num = 0;
                if (triIdx < batchEnd) {
                    num = pipe->triSubtris[triIdx];
                }

                // cumulative sum of triangle count within each warp,
                uint32_t myIdx =
                    __popc(__ballot_sync(WARP_FULL_MASK, num & 1) & ptx_lanemask_gt());
                // so, if num == 1 in each thread in warp,
                // myIdx[i] = i, where i is lane in warp

                if (ptx_lane_id() == 0) {
                    s_broadcast[threadIdx.y + 16] = myIdx + num;
                }
                __syncthreads();

                // compute cumulative sum of per-warp triangle counts to determine
                // storage position in ring buffer in shared memory
                // use parallel reduction algorithm:
                // https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

                if (thrInBlock < CUDARF_BIN_WARPS)
                {
                    volatile uint32_t* ptr = &s_broadcast[thrInBlock + 16];
                    uint32_t val = *ptr;

#if (CUDARF_BIN_WARPS > 1)
            		val += ptr[-1]; *ptr = val;
#endif
#if (CUDARF_BIN_WARPS > 2)
            		val += ptr[-2]; *ptr = val;
#endif
#if (CUDARF_BIN_WARPS > 4)
            		val += ptr[-4]; *ptr = val;
#endif
#if (CUDARF_BIN_WARPS > 8)
            		val += ptr[-8]; *ptr = val;
#endif
#if (CUDARF_BIN_WARPS > 16)
            		val += ptr[-16]; *ptr = val;
#endif
                    // initially assume that we consume everything

                    // same as TODO1
                    if (thrInBlock == CUDARF_BIN_WARPS - 1) {
                        s_batchPos = batchPos + CUDARF_BIN_WARPS * 32;
                        s_bufCount = bufCount + val;
                    }

                }
                __syncthreads();

                // skip if no subtriangles
                if (num) {
                    // calculate write position for triangle
                    uint32_t pos = bufCount + myIdx + s_broadcast[threadIdx.y + 16 - 1];

                    // only write if entire triangle fits
                    if (pos + num <= ARRAY_SIZE(s_triBuf)) {
                        pos += bufIndex; // adjust for current start position
                        pos &= ARRAY_SIZE(s_triBuf) - 1;
                        s_triBuf[pos] = triIdx;
                    }
                    // this triangle is the first that failed,
                    // overwrite total count and triangle count
                    else if (pos <= ARRAY_SIZE(s_triBuf)) {
                        s_batchPos = batchPos + thrInBlock;
                        s_bufCount = pos;
                    }
                }

                // update triangle counts
                __syncthreads();
                batchPos = s_batchPos;
                bufCount = s_bufCount;
            }

            // make every warp clear its output buffers
            for (int i = threadIdx.x; i < ctx.numBins; i += 32)
                s_outMask[threadIdx.y][i] = 0;

            // .. Rasterization phase: each thread processes one triangle
            // -----------------------------------------------------------------

            // choose our triangle
            // TODO: consider using texture fetch here
            const cudarf::rast::Triangle *triData = nullptr;

            int triIdx = -1;

            if (thrInBlock < bufCount)
            {
                uint32_t triPos = bufIndex + thrInBlock;
                triPos &= ARRAY_SIZE(s_triBuf)-1;

                // find triangle
                triIdx = s_triBuf[triPos];
                triData = &triangles[triIdx];
            }

            // setup bounding box and edge functions, and rasterize
            int32_t lox, loy, hix, hiy;
            if (thrInBlock < bufCount) {
                const cudarf::rast::Triangle &tri = *triData;

                // Avoid deriving bin indices by directly remapping window-space
                // pixel coordinates into the fixed bin grid. That approach can
                // skip valid bins for some output resolutions, so neighboring
                // pixels do not map onto the bin lattice consistently.

                // int2 v0 = make_int2(tri.iP0.x, tri.iP0.y) +
                //     make_int2(pipe->windowWidth,
                //               pipe->windowHeight) * CUDARF_SUBPIXEL_SIZE / 2;

                // int2 d01 = make_int2(tri.iP1.x, tri.iP1.y) -
                //     make_int2(tri.iP0.x, tri.iP0.y);
                // int2 d02 = make_int2(tri.iP2.x, tri.iP2.y) -
                //     make_int2(tri.iP0.x, tri.iP0.y);

                // log2(# of subpixels in bin) = 7(128x128) + 4(16x16)
				// int binLog = CUDARF_BIN_LOG2 + CUDARF_TILE_LOG2 + CUDARF_SUBPIXEL_LOG2;

                // lox = lo.x >> binLog;
                // loy = lo.y >> binLog;
                // hix = hi.x >> binLog;
                // hiy = hi.y >> binLog;
                // -----------------

                bin_idx_from_coord(pipe, tri.flo, lox, loy);
                bin_idx_from_coord(pipe, tri.fhi, hix, hiy);

                // .. determine bin coverage by triangle
                // most common case: triangle covers at most 2x2 bins, that is
                // correct solution when triangle covers 1x2, 2x1 or 1x1 bins

                // each triangle in warp spans one bin
				bool is_1x1 = (hix == lox && hiy == loy);

                // each triangle in warp spans 2x2 bins, including 2x1 or 1x2
                // it might turn out that triangle may not occupy lower left
                // bin of 2x2, but still end up in (lox, loy) output mask, this
                // is not an issue from correctness point of view

                bool is_2x2 = (hix <= lox + 1 && hiy <= loy + 1);

				uint32_t bit = 1 << threadIdx.x;

				if (__all_sync(__activemask(), is_1x1))
				{
                    int32_t binIdx = lox + ctx.binsX * loy;

                    // Each thread in warp writes (likely different) bin index
                    // to broadcast location. Winned value is selected for
                    // storing mask with overlapped triangles. Loop is
                    // continued until thread's bin mask is stored.
                    bool won;
                    do {
                        s_broadcast[threadIdx.y + 16] = binIdx;
                        int selected = s_broadcast[threadIdx.y + 16];
                        won = (selected == binIdx);
                        uint32_t mask = __ballot_sync(__activemask(), won);
                        s_outMask[threadIdx.y][selected] = mask;
                    } while (!won);

                } else if (__all_sync(__activemask(), is_2x2)) {
                    int32_t binIdx = lox + ctx.binsX * loy;

                    atomicOr((uint32_t *) &s_outMask[threadIdx.y][binIdx], bit);

                    if (hix > lox) {atomicOr((uint32_t *) &s_outMask[threadIdx.y][binIdx + 1], bit);}
                    if (hiy > loy) {atomicOr((uint32_t *) &s_outMask[threadIdx.y][binIdx + ctx.binsX], bit);}
                    if (hix > lox && hiy > loy) {
                        atomicOr((uint32_t *) &s_outMask[threadIdx.y][binIdx + ctx.binsX + 1], bit);
                    }
                }
                // else we have general case
                else {
                    // TODO don't iterate over all bins
                    for (int binIdx = 0; binIdx < ctx.numBins; binIdx++)
                    {
                        int binX = binIdx % ctx.binsX;
                        int binY = binIdx / ctx.binsY;

                        float2 center = bin_center_from_idx(pipe, binX, binY);

                        // TODO compute only once
                        float2 half = make_float2(pipe->binCtx.binW / 2.0,
                                                  pipe->binCtx.binH / 2.0);

                        // Outside AABB => skip.
                        if (tri.flo.x >= center.x + half.x || tri.flo.y >= center.y + half.y ||
                            tri.fhi.x <= center.x - half.x || tri.fhi.y <= center.y - half.y)
                        {continue;}

                        if (disjoint(tri.sP0 - center, tri.sP1 - center, tri.sP2 - center, half)) {continue;}

                        atomicOr((uint32_t *) &s_outMask[threadIdx.y][binIdx], bit);
                    }
                }
            }

			s_overTotal = 0; // overflow counter

			// ensure that out masks are done
			__syncthreads();

            // index of current thread (within block) if its bin has overflown
            int overIndex = -1;

            // compute cumulative contribution of each bin across all warps in block

            if (thrInBlock < ctx.numBins) {
                uint8_t *srcPtr = (uint8_t *) &s_outMask[0][thrInBlock];
                uint8_t *dstPtr = (uint8_t *) &s_outCount[0][thrInBlock];
                int total = 0; // eventually will store total number of triangles
                               // assigned to specific bin

                for (int i = 0; i < CUDARF_BIN_WARPS; i++) {
                    total += __popc(*(uint32_t *) srcPtr);
                    *(uint32_t *) dstPtr = total;
                    srcPtr += (CUDARF_MAXBINS_SQR + 1) * sizeof(int32_t);
                    dstPtr += (CUDARF_MAXBINS_SQR + 1) * sizeof(int32_t);
                }

				int ofs = s_outOfs[thrInBlock];

                // For all bins with no more segment capacity compute overflow
                // index. Default ofs value is set to -CUDARF_BIN_SEG_SIZE so that
                // +1 would overflow lower bits and change segment number

				if (((ofs - 1) >> CUDARF_BIN_SEG_LOG2) !=
                    (((ofs - 1) + total) >> CUDARF_BIN_SEG_LOG2))
                {
                    uint32_t mask = __ballot_sync(__activemask(), true);

                    overIndex = __popc(mask & ptx_lanemask_gt());

                    // s_broadcast[warp_id] <- cumulative sum of per-warp overflown threads
                    if (overIndex == 0) {
                        s_broadcast[threadIdx.y + 16] =
                            atomicAdd((uint32_t*)&s_overTotal, __popc(mask));
                    }

                    // TODO: check if whether CUDA ensures that threads will
                    // re-converge after `if(overIndex == 0)' clause
                    __syncwarp(mask);

                    // add per-warp offset to overflow index
                    overIndex += s_broadcast[threadIdx.y + 16];
                    s_overIndex[thrInBlock] = overIndex;
                }
            }

            // sync after s_overTotal is ready
            __syncthreads();

            // .. Computation of output indices for each bin
            // It is important to to calculate indices in such a way that
            // triangle IDs are stored in output queues in same order as they
            // were read in
            // -----------------------------------------------------------------

            // overTotal now contains total number of overflown threads (segments) within block
            uint32_t overTotal = s_overTotal;
            uint32_t allocBase = 0;

            // at least one segment overflowed => allocate segments
            if (overTotal > 0) {

                // allocate memory
                if (thrInBlock == 0) {
                    uint32_t allocBase = atomicAdd(&g_atomics->numBinSegs, overTotal);
                    s_allocBase = (allocBase + overTotal <= ctx.maxBinSegs) ? allocBase : 0;   
                }
                __syncthreads();

                allocBase = s_allocBase;
            }

            // did my bin overflow?
            if (overIndex != -1) {

                // calculate new segment index
                int segIdx = allocBase + overIndex;

                // add to segments linked list
                if (s_outOfs[thrInBlock] < 0) {
                    binFirstSeg[(thrInBlock << CUDARF_BIN_STREAMS_LOG2) + blockIdx.x] = segIdx;
                }
                else {
                    binSegNext[(s_outOfs[thrInBlock] - 1) >> CUDARF_BIN_SEG_LOG2] = segIdx;
                }

                // defaults
                binSegCount[segIdx] = CUDARF_BIN_SEG_SIZE;
                binSegNext [segIdx] = -1;
            }

            // .. concurrent emission -- each warp handles its own triangle
            // --
            if (thrInBlock < bufCount)
            {
                int triPos  = (bufIndex + thrInBlock) & (ARRAY_SIZE(s_triBuf) - 1);
                int currBin = lox + loy * ctx.binsX;
                int skipBin = (hix + 1) + loy * ctx.binsX;
                int endBin  = lox + (hiy + 1) * ctx.binsY;
                // how many tiles to advance to the first tile in next line
                int binYInc = ctx.binsX - (hix - lox + 1);

                // loop over triangle's bins
                // TODO: rewrite using two for loops
                do
                {
                    uint32_t outMask = s_outMask[threadIdx.y][currBin];
                    if (outMask & (1 << threadIdx.x))
                    {
                        int idx = __popc(outMask & ptx_lanemask_gt());
                        if (threadIdx.y > 0)
                            idx += s_outCount[threadIdx.y-1][currBin];

                        int base = s_outOfs[currBin];
                        int free = (-base) & (CUDARF_BIN_SEG_SIZE - 1);

                        if (idx >= free) {
                            idx += ((allocBase + s_overIndex[currBin]) << CUDARF_BIN_SEG_LOG2) - free;
                        }
                        else { idx += base;}

                        // how does binSegData stack with binFirstSeg,
                        // binSegNext, binSegCount etc?
						binSegData[idx] = s_triBuf[triPos];
                    }

                    currBin++;

                    if (currBin == skipBin) {
                        currBin += binYInc;
                        skipBin += ctx.binsX;
                    }
                }
                while (currBin != endBin);
            }

            // -- wait all triangles to finish, then replace overflown segment offsets
			__syncthreads();
            if (thrInBlock < ctx.numBins)
            {
                uint32_t total  = s_outCount[CUDARF_BIN_WARPS - 1][thrInBlock];
                uint32_t oldOfs = s_outOfs[thrInBlock];
                if (overIndex == -1)
                    s_outOfs[thrInBlock] = oldOfs + total;
                else
                {
                    int addr = oldOfs + total;
                    // initial case: (total - 512) & (512 - 1) = total (upper 23
                    // bits are cleared)
                    addr = ((addr - 1) & (CUDARF_BIN_SEG_SIZE - 1)) + 1;
                    addr += (allocBase + overIndex) << CUDARF_BIN_SEG_LOG2;
					s_outOfs[thrInBlock] = addr;
				}
                s_outTotal[thrInBlock] += total;
			}

            // these triangles are now done
            int count = ::min(bufCount, CUDARF_BIN_WARPS * 32);
            bufCount -= count;
            bufIndex += count;
            bufIndex &= ARRAY_SIZE(s_triBuf) - 1;
        } while (bufCount > 0 || batchPos < batchEnd);

        // .. flush all bins output queues by marking the last segments as full, before
        // repeating input phase. This guarantees that merging of per-block
        // queues of per-segment bases, which is enough bc each segment
        // corresponds to single, continuous part of the input

        if (thrInBlock < ctx.numBins)
        {
			int ofs = s_outOfs[thrInBlock];
			if (ofs & (CUDARF_BIN_SEG_SIZE-1))
			{
				int seg = ofs >> CUDARF_BIN_SEG_LOG2;

				binSegCount[seg] = ofs & (CUDARF_BIN_SEG_SIZE-1);

                // Advance the write pointer to the next segment boundary.
                // This finalizes the current partially-filled segment so the
                // next input batch starts appending at offset 0 of a fresh one.
                //
                // Example with CUDARF_BIN_SEG_SIZE = 512:
                //   ofs = 3600                  -> segment 7, offset 16
                //   ofs + (512 - 1) = 4111     -> move just past the boundary
                //   4111 & -512      = 4096    -> clear low offset bits
                //
                // Result: s_outOfs now points to segment 8, offset 0.
                s_outOfs[thrInBlock] = (ofs + CUDARF_BIN_SEG_SIZE - 1) & -CUDARF_BIN_SEG_SIZE;
			}
		}

        // output totals
        if (thrInBlock < ctx.numBins) {
            binTotal[(thrInBlock << CUDARF_BIN_STREAMS_LOG2) + blockIdx.x] = s_outTotal[thrInBlock];
        }
    }
}
