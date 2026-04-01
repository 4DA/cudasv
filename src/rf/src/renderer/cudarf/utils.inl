#ifndef CUDA_UTILS_HPP
#define CUDA_UTILS_HPP

#include <cmath>
#include "types.hpp"
#include "vecglm.inl"

static int32_t iclamp(int32_t what, int32_t from, int32_t to)
{
    if (what < from) {return from;}
    if (what > to) {return to;}
    return what;
}

// round up value to next multiple of 2^L
int32_t __host__ __device__ round_up_to_mult_pwr(int32_t val, int L)
{
    return (((val - 1) >> L) + 1) << L;
}

__attribute__((unused)) static __device__ uint
rgba_float_to_int(float4 rgba) {
  rgba.x = __saturatef(rgba.x);  // clamp to [0.0, 1.0]
  rgba.y = __saturatef(rgba.y);
  rgba.z = __saturatef(rgba.z);
  rgba.w = __saturatef(rgba.w);
  return (uint(rgba.w * 255) << 24) | (uint(rgba.z * 255) << 16) |
         (uint(rgba.y * 255) << 8) | uint(rgba.x * 255);
}


//TODO add matrix type and implement matrix multiplication
__attribute__((unused)) __host__ __device__ __inline__ static
cudarf::Vec3f operator*(glm::mat3 mat, cudarf::Vec3f vec) {
    return to_vec3f(mat * to_glm(vec));
}

__attribute__((unused)) __host__ __device__ __inline__ static
cudarf::Vec4f operator*(glm::mat4 mat, cudarf::Vec4f vec) {
    return to_vec4f(mat * to_glm(vec));
}

static __device__ void bin_idx_from_coord(const cudarf::rast::PipeParams *pipe, float2 fragScrn, int32_t &binX, int32_t &binY)
{
    binX = clamp(int32_t(fragScrn.x / pipe->binCtx.binW), 0, CUDARF_BIN_COUNT - 1);
    binY = clamp(int32_t(fragScrn.y / pipe->binCtx.binH), 0, CUDARF_BIN_COUNT - 1);
}

static __device__ float2 bin_center_from_idx(const cudarf::rast::PipeParams *pipe, const int32_t binX, const int32_t binY)
{
    return make_float2((binX + 0.5) * pipe->binCtx.binW,
                       (binY + 0.5) * pipe->binCtx.binH);
}

static __device__ int2 tile_idx_from_coord(const cudarf::rast::PipeParams *pipe, float2 fragScrn)
{
    return make_int2(fragScrn.x / CUDARF_TILE_SZ, fragScrn.y / CUDARF_TILE_SZ);
}

__attribute__((unused)) static __host__ __device__
void dump_vert(const char *pref, const char *post, const cudarf::Vec2f &v)
{
    printf("%s(%.3f, %.3f)%s", pref, v.x, v.y, post);
}

__attribute__((unused)) static __host__ __device__
void dump_vert(const char *pref, const char *post, const int2 &v)
{
    printf("%s(%d, %d)%s", pref, v.x, v.y, post);
}

__attribute__((unused)) static __host__ __device__
void dump_vert(const char *pref, const char *post, const cudarf::Vec3f &v)
{
    printf("%s(%.3f, %.3f, %.3f)%s", pref, v.x, v.y, v.z, post);
}

__attribute__((unused)) static __host__ __device__
void dump_vert(const char *pref, const char *post, const int3 &v)
{
    printf("%s(%d, %d, %d)%s", pref, v.x, v.y, v.z, post);
}

__attribute__((unused)) static __host__ __device__
void dump_vert(const char *pref, const char *post, const cudarf::Vec4f &v)
{
    printf("%s(%.3f, %.3f, %.3f, %.3f)%s", pref, v.x, v.y, v.z, v.w, post);
}

__attribute__((unused)) static __host__ __device__
void dump_vert(const char *pref, const char *post, const int4 &v)
{
    printf("%s(%d, %d, %d, %d)%s", pref, v.x, v.y, v.z, v.w, post);
}

__attribute__((unused)) static __host__ __device__
void dump_vert(const char *pref, const char *post, const uchar4 &v)
{
    printf("%s(%d, %d, %d, %d)%s", pref, v.x, v.y, v.z, v.w, post);
}

__attribute__((unused))
__device__ __host__ static inline void dump_mat3(const char *prefix, const glm::mat3 &mat) {
    printf("%s: ", prefix);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%f ", mat[i][j]);
        }
    }
    printf("\n");
}

__attribute__((unused))
__device__ __host__ static inline void dump_mat4(const char *prefix, const glm::mat4 &mat) {
    printf("%s: ", prefix);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%f ", mat[i][j]);
        }
    }
    printf("\n");
}

__host__ __device__ __attribute__((unused))
static cudarf::rast::AABB get_aabb(cudarf::Vec3f v0, cudarf::Vec3f v1, cudarf::Vec3f v2) {
    cudarf::rast::AABB aabb;
    aabb.min = make_float3(
            min(min(v0.x, v1.x), v2.x),
            min(min(v0.y, v1.y), v2.y),
            min(min(v0.z, v1.z), v2.z));
    aabb.max = make_float3(
            max(max(v0.x, v1.x), v2.x),
            max(max(v0.y, v1.y), v2.y),
            max(max(v0.z, v1.z), v2.z));
    return aabb;
}

__device__ bool overlaps(cudarf::rast::AABB a, cudarf::rast::AABB b){
	bool result;
	if (a.max.x < b.min.x) {
		result = false;
	}
	else if (a.min.x > b.max.x){
		result = false;
	}
	else if (a.max.y < b.min.y){
		result = false;
	}
	else if (a.min.y > b.max.y) {
		result = false;
	}
	else {
		result = true;
	}

	return result;
}

__device__ bool overlaps(float2 lo1, float2 hi1, float2 lo2, float2 hi2)
{
	if (hi1.x < lo2.x) {
		return false;
	}
	else if (lo1.x > hi2.x){
		return false;
	}
	else if (hi1.y < lo2.y){
		return false;
	}
	else if (lo1.y > hi2.y) {
		return false;
	}
	else {
		return true;
	}
}

__device__ bool overlaps(int2 lo1, int2 hi1, int2 lo2, int2 hi2)
{
	if (hi1.x < lo2.x) {
		return false;
	}
	else if (lo1.x > hi2.x){
		return false;
	}
	else if (hi1.y < lo2.y){
		return false;
	}
	else if (lo1.y > hi2.y) {
		return false;
	}
	else {
		return true;
	}
}

__device__ bool is_inside(int2 point, int2 lo2, int2 hi2)
{
	if (point.x < lo2.x) {
		return false;
	}
	else if (point.x > hi2.x){
		return false;
	}
	else if (point.y < lo2.y){
		return false;
	}
	else if (point.y > hi2.y) {
		return false;
	}
	else {
		return true;
	}
}


/** Compute determinant of (A, B, C) extended to homogeneous space

    | Ax Bx Cx |
    | Ay By Cy |
    |  1  1  1 |

    Result is unique for any cyclical permutation and unique up to sign for all
    other permutations.

    Value can be interpeted as double signed area of parallelogram with edges
    BA and CA.

    If sign is positive, then C belongs to the left(+) halfspace of the
    directed edge AB. If sign is negative, then C belongs in right(-) halfspace
    of AB. If value is zero then C lies on AB. So, triangle with vertices in
    CCW order will have positive area.
 */

__device__ __inline__ float edge_function(const cudarf::Vec2f &a, const cudarf::Vec2f &b, const cudarf::Vec2f &c)
{
    return (b.x-a.x) * (c.y-a.y) - (b.y-a.y) * (c.x-a.x);
}

/** Compute determinant of (A, B, C) extended to homogeneous space
   if a, b and c are signed p-bit integers, then maximum abs value of result
   is 2^(2p+1) - 2.

   Namely, if result is desired to fit in 32-bit integer range, then p ≤ (32-2)/2 =
   15 to avoid overflows.
*/

__device__ __inline__ int32_t edge_function(int2 a, int2 b, int2 c)
{
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

// sPX \in [0; screen width] x [0; screen height]
__device__ __inline__
void to_screen_space(const cudarf::rast::PipeParams *pipe, const cudarf::rast::Triangle &tri,
                     cudarf::Vec2f &sP0, cudarf::Vec2f &sP1, cudarf::Vec2f &sP2)
{
    // TODO store in pipe params
    float2 screen = make_float2((float)pipe->windowWidth, (float)pipe->windowHeight);

    // sP0 = 0.5f * screen * (make_float2(tri.P0.x, tri.P0.y) + 1.0f);
    // sP1 = 0.5f * screen * (make_float2(tri.P1.x, tri.P1.y) + 1.0f);
    // sP2 = 0.5f * screen * (make_float2(tri.P2.x, tri.P2.y) + 1.0f);
}


// most straighforward approach: compute barycentric coordinates, given triangle
// vertices in 3DP(NDC) space and raster coordinates
// frag - fragment coordinates in NDC space
__device__ __inline__ cudarf::Vec3f compute_bary(const cudarf::rast::Triangle &tri, const cudarf::Vec2f &frag)
{
    return tri.area_rcp * cudarf::Vec3f {
        edge_function(tri.sP1, tri.sP2, frag),
        edge_function(tri.sP2, tri.sP0, frag),
        edge_function(tri.sP0, tri.sP1, frag)
    };
}

// TBD
int3 plane_equation(const float3 &values, const int2 &v0,
                    const int2 &d1, const int2 &d2,
                    int32_t area)
{
    int3 result = make_int3(0,0,0);
    double t10 = (double) values.y - values.x;
    double t20 = (double) values.z - values.x;

    double xc = (t10 * (double)d2.y - t20 * (double)d1.y) / (double)area;
    double yc = (t20 * (double)d1.x - t10 * (double)d2.x) / (double)area;

    return result;
}

/**
 * Check if a barycentric coordinate is within the boundaries of a triangle.
 */
__host__ __device__ __attribute__((unused))
static bool bary_in_bounds(const cudarf::Vec3f barycentricCoord) {
    return barycentricCoord.x >= 0.0 && barycentricCoord.x <= 1.0 &&
           barycentricCoord.y >= 0.0 && barycentricCoord.y <= 1.0 &&
           barycentricCoord.z >= 0.0 && barycentricCoord.z <= 1.0;
}

/**
 * Check if a barycentric coordinate is within the boundaries of a triangle.
 */
__host__ __device__ __attribute__((unused))
static bool bary_in_bounds(const int3 coord) {
    return coord.x >= 0 && coord.y >= 0 && coord.z >= 0;
}

/** Precompute the affine transform from fragment coordinates to barycentric
    coordinates.

    For each edge function if we treat a, b as known and set p = c as variable,
    we can rearrange terms to the following form:
    Fij(p) = (Pi.y - Pj.y) * p.x + (Pj.x - Pi.x) * p.y + C

    Here for (i,j) = {(0,1), (1,2), (2,0)} we precompute p_x, p_y multipliers as
    well as C and store them as scale and bias in vectorized form.
 */
static __device__ __inline__ void
compute_bary_affine_tr(const cudarf::Vec2f &P0, const cudarf::Vec2f &P1, const cudarf::Vec2f &P2, float area_rcp,
                       cudarf::Vec3f &scale_x, cudarf::Vec3f &scale_y, cudarf::Vec3f &bias)
{
    scale_x = area_rcp * make_vec3f(P1.y - P2.y, P2.y - P0.y, P0.y - P1.y);
    scale_y = area_rcp * make_vec3f(P2.x - P1.x, P0.x - P2.x, P1.x - P0.x);
    bias    = area_rcp * make_vec3f(P1.x * P2.y - P2.x * P1.y,
                                    P2.x * P0.y - P0.x * P2.y,
                                    P0.x * P1.y - P1.x * P0.y);
}

static __device__ bool fuzzy_equal(float x, float y, float margin)
{
    return std::abs(x - y) <= margin;
}

/**
   Return true if fragment passes fill rule.

   Fill rule for edges: fragment is drawn if it is on top edge or
   left edge of triangle.
   --
   In a CCW triangle, a top edge is an edge that is exactly
   horizontal and goes towards the left, i.e. its end point is left
   of its start point;

   In a CCW triangle, a left edge is an edge that goes down,
   i.e. its end point is strictly below its start point.
*/
static __device__ __inline__ bool is_top_left(const cudarf::Vec2f &edge, float w)
{
    // TODO compute reasonable error here
    const float ERROR = 2.0f * FLT_EPSILON;

    if (fuzzy_equal(w, 0.0f, ERROR)) {
        return (fuzzy_equal(edge.y, 0.0f, ERROR) && edge.x < 0.0f) || (edge.y < 0.0f);
    } else {
        return w > 0.0f;
    }
}

static __device__ __inline__ bool is_top_left(const int2 &edge, int w)
{
    if (w == 0) {
        return (edge.y == 0 && edge.x < 0) || (edge.y < 0);
    } else {
        return w > 0;
    }
}


template <typename T>
__device__ __inline__ T min3(const T &a, const T &b, const T &c)
{
    return min(min(a, b), c);
}

template <typename T>
__device__ __inline__ T max3(const T &a, const T &b, const T &c)
{
    return max(max(a, b), c);
}

static __device__ __inline__ int dot(const int3 &a, const int3 &b)
{
    return (a.x * b.x + a.y * b.y + a.z * b.z);
}

////////////////////////////////////////////////////////////////////////////////
// rotate vector 90 degrees CCW
////////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ float2 perp(float2 a)
{
    return make_float2(-a.y, a.x);
}

static __device__ void clip_to_window(const cudarf::Vec2f &in, uint w, uint h, uint &x, uint &y)
{
    cudarf::Vec2f clip = in / 2.0f + make_vec2f(0.5f, 0.5f);
    x = uint(w * clip.x);
    y = uint(h * clip.y);
}

static __device__ float2 clip_to_window(const cudarf::Vec2f &in, uint w, uint h)
{
    cudarf::Vec2f clip = in / 2.0f + make_vec2f(0.5f, 0.5f);
    return make_float2(w * clip.x, h * clip.y);
}

// static void __device__ compute_aabb(const cudarf::rast::Triangle &tri, int2 &lo, int2 &hi)
// {
//     // todo use subpixel shift here

//     // Compute triangle bounding box
//     int min_x = min3(tri.iP0.x, tri.iP1.x, tri.iP2.x);
//     int min_y = min3(tri.iP0.y, tri.iP1.y, tri.iP2.y);
//     int max_x = max3(tri.iP0.x, tri.iP1.x, tri.iP2.x);
//     int max_y = max3(tri.iP0.y, tri.iP1.y, tri.iP2.y);

//     // Clip against screen bounds
//     // min_x = max(min_x, 0);
//     // min_y = max(min_y, 0);
//     // max_x = min(max_x, pipe->windowWidth - 1);
//     // max_y = min(max_y, pipe->windowHeight - 1);

//     lo = make_int2(min_x, min_y);
//     hi = make_int2(max_x, max_y);
// }

void __device__ compute_aabb_screen(const cudarf::rast::PipeParams *pipe, const cudarf::rast::Triangle &tri, float2 &lo, float2 &hi)
{
    // Compute triangle bounding box
    float min_x = min3(tri.sP0.x, tri.sP1.x, tri.sP2.x);
    float min_y = min3(tri.sP0.y, tri.sP1.y, tri.sP2.y);
    float max_x = max3(tri.sP0.x, tri.sP1.x, tri.sP2.x);
    float max_y = max3(tri.sP0.y, tri.sP1.y, tri.sP2.y);

    // Clip against screen bounds
    min_x = max(min_x, 0.0f);
    min_y = max(min_y, 0.0f);
    max_x = min(max_x, (float)pipe->windowWidth);
    max_y = min(max_y, (float)pipe->windowHeight);

    lo = make_float2(min_x, min_y);
    hi = make_float2(max_x, max_y);
}

// convert post-projection Z ∈ [0; 1] to integer viewport range
static __device__ __inline__ int32_t Z_to_fixed(float val)
{
    const float SRC_MIN = 0.0f;
    const float SRC_MAX = 1.0f;
    const float SRC_RANGE = SRC_MAX - SRC_MIN;

    const int DST_MIN = CUDARF_DEPTH_MIN;
    const int DST_RANGE = CUDARF_DEPTH_RANGE;

    return
        static_cast<int32_t>(((val - SRC_MIN) * DST_RANGE) / SRC_RANGE) + DST_MIN;
}

// convert post-projection Z ∈ [0; 1] to integer viewport range
static __device__ __inline__ float Z_from_fixed(int32_t val)
{
    const int SRC_MIN = CUDARF_DEPTH_MIN;
    const int SRC_RANGE = CUDARF_DEPTH_RANGE;

    const float DST_MIN = 0.0f;
    const float DST_MAX = 1.0f;
    const float DST_RANGE = DST_MAX - DST_MIN;

    return (((val - SRC_MIN) * DST_RANGE) / static_cast<float>(SRC_RANGE)) + DST_MIN;
}

static __device__ void raster_to_window(const int2 &in, uint w, uint h, uint &x_out, uint &y_out)
{
    const int SRC_MIN_X = -(w  << (CUDARF_SUBPIXEL_LOG2 - 1)) - 1;
    const int SRC_RANGE_X = w << CUDARF_SUBPIXEL_LOG2;

    const int SRC_MIN_Y = -(h  << (CUDARF_SUBPIXEL_LOG2 - 1)) - 1;
    const int SRC_RANGE_Y = h << CUDARF_SUBPIXEL_LOG2;

    const int DST_MIN_X = 0;
    const int DST_RANGE_X = w;

    const int DST_MIN_Y = 0;
    const int DST_RANGE_Y = h;

    x_out = (((in.x - SRC_MIN_X) * DST_RANGE_X) / SRC_RANGE_X) + DST_MIN_X;
    y_out = (((in.y - SRC_MIN_Y) * DST_RANGE_Y) / SRC_RANGE_Y) + DST_MIN_Y;
}

static __device__ cudarf::Vec2f window_to_clip(uint x, uint y, uint w, uint h)
{
    return 2.0f * make_vec2f(float(x) / w, float(y) / h) - make_vec2f(1.0f, 1.0f);
}

__host__ __device__ __attribute__((unused))
static 
cudarf::Vec4f interpolate(const cudarf::Vec3f bary, const cudarf::Vec4f tri[3])
{
    return bary.x * tri[0] + bary.y * tri[1] + bary.z * tri[2];
}

template <typename T>
__host__ __device__ __attribute__((unused))
static T interp(float3 bary, T val[3])
{
    return bary.x * val[0] + bary.y * val[1] + bary.z * val[2];
}

__attribute__((unused)) __host__ __device__
static float4 to_float4(const cudarf::Vec4f &src)
{
    return make_float4(src.x, src.y, src.z, src.w);
}

__attribute__((unused)) __host__ __device__
static float3 to_float3(const cudarf::Vec3f &src)
{
    return make_float3(src.x, src.y, src.z);
}

__attribute__((unused)) __host__ __device__
static float2 to_float2(const cudarf::Vec2f &src)
{
    return make_float2(src.x, src.y);
}

__attribute__((unused)) __host__ __device__
static float3 to_float3(const cudarf::Vec4f &src)
{
    return make_float3(src.x, src.y, src.z);
}

__attribute__((unused)) __host__ __device__
static cudarf::Color make_color(const cudarf::ColorRGB &rgb, float a)
{
    return make_float4(rgb, a);
}

__attribute__((unused)) __host__ __device__
static cudarf::Color make_color(float r, float g, float b, float a)
{
    return make_float4(r, g, b, a);
}

__attribute__((unused)) __host__ __device__
static cudarf::ColorRGB make_color(float r, float g, float b)
{
    return make_float3(r, g, b);
}

__attribute__((unused)) __host__ __device__
static cudarf::ColorRGB to_rgb(const cudarf::Vec4f &src)
{
    return make_color(src.x, src.y, src.z);
}

// for PTX asm reference see https://docs.nvidia.com/cuda/parallel-thread-execution/index.html

__device__ __inline__ int32_t   ptx_max_max (int32_t a, int32_t b, int32_t c)
{ int32_t v; asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }

__device__ __inline__ int32_t   ptx_min_min (int32_t a, int32_t b, int32_t c)
{ int32_t v; asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt

__device__ __inline__ int32_t   ptx_f32_to_s32_sat (float a)
{ int32_t v; asm("cvt.rni.sat.s32.f32 %0, %1;" : "=r"(v) : "f"(a)); return v; }


__device__ __inline__ uint32_t   ptx_lanemask_lt(void)
{ uint32_t r; asm("mov.u32 %0, %lanemask_lt;" : "=r"(r)); return r; }

__device__ __inline__ uint32_t   ptx_lanemask_gt(void)
{ uint32_t r; asm("mov.u32 %0, %lanemask_gt;" : "=r"(r)); return r; }


inline __device__ unsigned ptx_lane_id() {
  unsigned ret;
  asm volatile("mov.u32 %0, %laneid;" : "=r"(ret));
  return ret;
}

inline __device__ unsigned ptx_warp_id() {
  unsigned ret;
  asm volatile("mov.u32 %0, %warpid;" : "=r"(ret));
  return ret;
}

#endif
