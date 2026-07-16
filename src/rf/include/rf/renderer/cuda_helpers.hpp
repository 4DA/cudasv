#ifndef CUDA_HELPERS_HPP
#define CUDA_HELPERS_HPP

#include <cassert>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#ifdef __DRIVER_TYPES_H__
[[maybe_unused]]
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
    assert(false);
    exit(EXIT_FAILURE);
  }
}

#define CUDA_CHK(val) check((val), #val, __FILE__, __LINE__)

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define CUDA_CHK_ERROR(msg) _checkCUDAErrorHelper(msg, FILENAME, __LINE__)

[[maybe_unused]]
static void _checkCUDAErrorHelper(const char *msg,
                                  const char *filename,
                                  int line) {
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
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
    getchar();
#  endif
    assert(false);
    exit(EXIT_FAILURE);
#endif
}

#endif
