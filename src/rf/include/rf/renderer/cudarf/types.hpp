#ifndef CUDARF_TYPES_HPP
#define CUDARF_TYPES_HPP

#include <cuda_runtime.h>

// constants
// TODO move to constants.hpp

#define ARRAY_SIZE(X) (sizeof(X) / sizeof((X)[0]))

#define WARP_FULL_MASK 0xffffffff

// TODO: should be obtained with deviceQuery
#define THREADS_IN_BLOCK        512
#define WARPS_IN_BLOCK          32

/// number of signed integer bits that can be used for representation of
/// viewport coordinate with subpixel precision. See justification of this
/// number in edge_function description
#define CUDARF_VIEWPORT_BITS 15

// If binSize is 128, rasterizer viewport size is 2048,
// then each bin spans 128x128 pixels

// bin tiler
// --

/// number of warps to run on each SM for tiler bin phase
#define CUDARF_MAXBINS_LOG2         4       // log2(ViewportSize / BinSize)
#define CUDARF_BIN_LOG2             4       // log2(BinSize / TileSize)

#define CUDARF_BIN_WARPS            (THREADS_IN_BLOCK / WARPS_IN_BLOCK)
#define CUDARF_BIN_SEG_LOG2         9       // 32-bit entries.
#define CUDARF_BIN_STREAMS_LOG2     4
#define CUDARF_BIN_STREAMS_SIZE     (1 << CUDARF_BIN_STREAMS_LOG2)
#define CUDARF_BIN_SEG_SIZE         (1 << CUDARF_BIN_SEG_LOG2)
#define CUDARF_MAXBINS_SIZE         (1 << CUDARF_MAXBINS_LOG2)
#define CUDARF_MAXBINS_SQR          (1 << (CUDARF_MAXBINS_LOG2 * 2))
#define CUDARF_BIN_COUNT            (1 << CUDARF_BIN_LOG2)
#define CUDARF_BIN_SQR              (1 << (CUDARF_BIN_LOG2 * 2))

// coarse tiler
// --

#define CUDARF_TILE_LOG2            3       // log2(TileSize / PixelSize)

#define CUDARF_TILE_SZ              (1 << CUDARF_TILE_LOG2)
// log(maximum possible number of tiles in bin)
// TODO: show how this number is derived
#define CUDARF_MAX_TILES_LOG2       8
#define CUDARF_MAX_TILES            (1 << CUDARF_MAX_TILES_LOG2)
#define CUDARF_COARSE_QUEUE_LOG2    10      // Triangles.
#define CUDARF_COARSE_QUEUE_SIZE    (1 << CUDARF_COARSE_QUEUE_LOG2)
#define CUDARF_COARSE_WARPS         16      // power of two
#define CUDARF_TILE_SEG_LOG2        5       // 32-bit entries.
#define CUDARF_TILE_SEG_SIZE        (1 << CUDARF_TILE_SEG_LOG2)

#define CUDARF_MAX_SUBTRIS_COUNT (1 << 24)


/// log2(subpixels inside pixel)
///
#define CUDARF_SUBPIXEL_LOG2  4

#define CUDARF_SUBPIXEL_SIZE (1 << CUDARF_SUBPIXEL_LOG2)

/// log2(rectangular viewport size in pixels)
/// (11)
#define CUDARF_MAXVIEWPORT_LOG2 (CUDARF_VIEWPORT_BITS - CUDARF_SUBPIXEL_LOG2)

/// rectangular viewport size in pixels
/// (2048)
#define CUDARF_VIEWPORT_SIZE (1 << CUDARF_MAXVIEWPORT_LOG2)

/// maximum value of viewport coordinate value
/// (16383)
#define CUDARF_VPCOORD_MAX ((1 << (CUDARF_VIEWPORT_BITS - 1)) - 1)


/// maximum value of viewport coordinate value
/// (-16384)
#define CUDARF_VPCOORD_MIN (-CUDARF_VPCOORD_MAX - 1)


#define CUDARF_VIEWPORT_RANGE (CUDARF_VPCOORD_MAX - CUDARF_VIEWPORT_MIN)

/// todo consider possible overflows like in cudarast
#define CUDARF_DEPTH_MIN 0
#define CUDARF_DEPTH_MAX (1 << 15)
#define CUDARF_DEPTH_RANGE (CUDARF_DEPTH_MAX - CUDARF_DEPTH_MIN)

#define PRIM_ELEMS 3

#define CUDARF_MAX_CONSTMEM_SZ (4 * sizeof(cudarf::rast::PipeParams))

#define cudarf_cuda_malloc cudaMalloc
#define cudarf_cuda_free cudaFree

#define cudarf_cuda_malloc_host cudaMallocHost
#define cudarf_cuda_free_host cudaFreeHost

namespace cudarf
{
struct Material;
using PrimitiveIndex = unsigned int;
using DepthValue = float;
using Color = float4;
using ColorRGB = float3;
using ColorN = uchar4;
using Vec4f = float4;
using Vec3f = float3;
using Vec2f = float2;
using FBTexture = cudaTextureObject_t;
using Velocity = float2;
}

#endif
