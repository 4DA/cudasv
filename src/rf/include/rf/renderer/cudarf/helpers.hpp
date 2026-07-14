#ifndef CUDARF_HELPERS_HPP
#define CUDARF_HELPERS_HPP

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include <rf/renderer/cuda_helpers.hpp>

// round up value to next 2^L
int32_t __host__ __device__ round_up_to_mult_pwr(int32_t val, int L);

// Free and reallocate a device buffer. Caller is responsible for ensuring the
// old pointer is either null or was previously allocated with cudaMalloc.
template<typename T>
static void reinit_buf(T **dev_ptr, size_t size) {
    CUDA_CHK(cudaFree(*dev_ptr));
    CUDA_CHK(cudaMalloc(dev_ptr, size));
}

#endif
