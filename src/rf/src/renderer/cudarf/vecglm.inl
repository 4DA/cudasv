#ifndef CUDA_VECGLM_HPP
#define CUDA_VECGLM_HPP

__attribute__((unused)) __host__ __device__ __inline__ static
cudarf::Vec2f make_vec2f(float x, float y) {
    return make_float2(x, y);
}

__attribute__((unused)) __host__ __device__ __inline__ static
cudarf::Vec3f make_vec3f(float x, float y, float z) {
    return make_float3(x, y, z);
}

__attribute__((unused)) __host__ __device__ __inline__ static
cudarf::Vec4f make_vec4f(float x, float y, float z, float w) {
    return make_float4(x, y, z, w);
}

__attribute__((unused)) __host__ __device__ __inline__ static
cudarf::Vec4f make_vec4f(cudarf::Vec3f vec, float w) {
    return make_float4(vec.x, vec.y, vec.z, w);
}

__attribute__((unused)) __host__ __device__ __inline__ static
cudarf::Vec3f to_vec3f(cudarf::Vec4f vec) {
    return make_float3(vec.x, vec.y, vec.z);
}

__attribute__((unused)) __host__ __device__ __inline__ static
cudarf::Vec2f to_vec2f(const glm::vec2 &vec) {
    return make_float2(vec.x, vec.y);
}

__attribute__((unused)) __host__ __device__ __inline__ static
cudarf::Vec2f to_vec2f(cudarf::Vec4f vec) {
    return make_float2(vec.x, vec.y);
}

__attribute__((unused)) __host__ __device__ __inline__ static
cudarf::Vec3f to_vec3f(const glm::vec3 &vec) {
    return make_float3(vec.x, vec.y, vec.z);
}

__attribute__((unused)) __host__ __device__ __inline__ static
cudarf::Vec4f to_vec4f(const glm::vec4 &vec) {
    return make_vec4f(vec.x, vec.y, vec.z, vec.w);
}

__attribute__((unused)) __host__ __device__ __inline__ static
glm::vec2 to_glm(const cudarf::Vec2f &vec) {
    return glm::vec2(vec.x, vec.y);
}

__attribute__((unused)) __host__ __device__ __inline__ static
glm::vec3 to_glm(const cudarf::Vec3f& vec) {
    return glm::vec3(vec.x, vec.y, vec.z);
}

__attribute__((unused)) __host__ __device__ __inline__ static
glm::vec4 to_glm(const cudarf::Vec4f& vec) {
    return glm::vec4(vec.x, vec.y, vec.z, vec.w);
}

#endif
