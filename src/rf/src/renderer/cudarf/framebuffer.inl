namespace fb
{
__device__ __inline__ uchar4 to_rgba_norm(const cudarf::Color &input)
{
    return make_uchar4(static_cast<unsigned char> (0.5f + 255.0f * clamp(input.x, 0.0f, 1.0f)),
                       static_cast<unsigned char> (0.5f + 255.0f * clamp(input.y, 0.0f, 1.0f)),
                       static_cast<unsigned char> (0.5f + 255.0f * clamp(input.z, 0.0f, 1.0f)),
                       static_cast<unsigned char> (0.5f + 255.0f * clamp(input.w, 0.0f, 1.0f)));
}

__device__ __inline__ cudarf::Color to_rgba_float(uchar4 input)
{
    return make_float4(input.x / 255.0f, input.y / 255.0f, input.z / 255.0f, input.w / 255.0f);
}

__device__ __inline__ void load(cudaSurfaceObject_t fb, int x, int y, int, cudarf::ColorN &out)
{
    surf2Dread(&out, fb, sizeof(cudarf::ColorN) * x, y);
}

__device__ __inline__ void load(cudaSurfaceObject_t fb, int x, int y, int, cudarf::Color &out)
{
    cudarf::ColorN data;
    surf2Dread(&data, fb, sizeof(cudarf::ColorN) * x, y);
    out = to_rgba_float(data);
}

__device__ __inline__ void load(cudaSurfaceObject_t fb, int x, int y, cudarf::ColorN &out)
{
    surf2Dread(&out, fb, sizeof(cudarf::ColorN) * x, y);
}

__device__ __inline__ void load(cudaSurfaceObject_t fb, int x, int y,  cudarf::Color &out)
{
    cudarf::ColorN data;
    surf2Dread(&data, fb, sizeof(cudarf::ColorN) * x, y);
    out = to_rgba_float(data);
}

__device__ __inline__ void load(cudarf::ColorN *fb, int w, int x, int y, cudarf::ColorN &out)
{
    out = fb[y * w + x];
}

__device__ __inline__ void load(cudarf::ColorN *fb, int w, int x, int y, cudarf::Color &out)
{
    out = to_rgba_float(fb[y * w + x]);
}

__device__ __inline__ void load(cudarf::LinearSurface fb, int x, int y, cudarf::ColorN &out)
{
    out = fb.devPtr[y * fb.w + x];
}

__device__ __inline__ void load(cudarf::LinearSurface fb, int x, int y, cudarf::Color &out)
{
    out = to_rgba_float(fb.devPtr[y * fb.w + x]);
}

__device__ __inline__ void store(cudaSurfaceObject_t fb, int x, int y, const cudarf::ColorN &color)
{
    surf2Dwrite(color, fb, sizeof(cudarf::ColorN) * x, y, cudaBoundaryModeTrap);
}

__device__ __inline__ void store(cudaSurfaceObject_t fb, int x, int y, const cudarf::Color &color)
{
    store(fb, x, y, to_rgba_norm(color));
}

__device__ __inline__ void store(cudarf::ColorN *fb, int w, int x, int y, const cudarf::ColorN &color)
{
    fb[y * w + x] = color;
}

__device__ __inline__ void store(cudarf::ColorN *fb, int w, int x, int y, const cudarf::Color &color)
{
    store(fb, w, x, y, to_rgba_norm(color));
}

__device__ __inline__ void store(cudarf::LinearSurface fb, int x, int y, const cudarf::ColorN &color)
{
    fb.devPtr[y * fb.w + x] = color;
}

__device__ __inline__ void store(cudarf::LinearSurface fb, int x, int y, const cudarf::Color &color)
{
    store(fb, x, y, to_rgba_norm(color));
}

__device__ __inline__ float4 tex_sample_4f32(cudaTextureObject_t tex, float2 uv)
{
    return tex2D<float4>(tex, uv.x, uv.y);
}

__device__ __inline__ float4 tex_sample_4f32(cudaTextureObject_t tex, float u, float v)
{
    return tex2D<float4>(tex, u, v);
}

// http://www.humus.name/temp/Linearize%20depth.txt
__device__ __inline__ float get_linear_depth(float z, float n, float f)
{
    return (2 * n * f) / (f + n - z * (f - n));
}

__device__ __inline__ int get_native_idx(int w, int h, int x, int y)
{
    return x + (h - 1 - y) * w;
}

__device__ __inline__ void load_native(cudarf::ColorN *fb, int w, int h, int x, int y, cudarf::Color &out)
{
    out = to_rgba_float(fb[get_native_idx(w, h, x, y)]);
}

__device__ __inline__
void store_native(uchar4 *out, int w, int h, int x, int y, const cudarf::Color &color)
{
    out[get_native_idx(w, h, x, y)] = to_rgba_norm(color);
}

}
