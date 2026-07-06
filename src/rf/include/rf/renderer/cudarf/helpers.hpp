#ifndef helpers_hpp
#define helpers_hpp

#include <cuda_runtime.h>
#include <stdio.h>
#include <ctype.h>

#ifdef __DRIVER_TYPES_H__
static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorName(error);
}
#endif

#ifdef CUDA_DRIVER_API
// CUDA Driver API errors
static const char *_cudaGetErrorEnum(CUresult error) {
  static char unknown[] = "<unknown>";
  const char *ret = NULL;
  cuGetErrorName(error, &ret);
  return ret ? ret : unknown;
}
#endif

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    // assert(false);
    // exit(EXIT_FAILURE);
  }
}

#define CUDA_CHK(val) check((val), #val, __FILE__, __LINE__)

__attribute__((unused))
static void _checkCUDAErrorHelper(const char *msg, const char *filename, int line) {
#if !defined(NDEBUG)
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (filename) {
        fprintf(stderr, " (%s:%d)", filename, line);
    }
    fprintf(stderr, ": %s: %s(%d)\n", msg, cudaGetErrorString(err), err);
    assert(false);
    // exit(EXIT_FAILURE);
#endif
}

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define CUDA_CHK_ERROR(msg) _checkCUDAErrorHelper(msg, FILENAME, __LINE__)
// #define CUDA_CHK_ERROR(msg) {}

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
