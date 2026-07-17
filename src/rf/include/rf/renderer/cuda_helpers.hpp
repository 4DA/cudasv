#ifndef CUDA_HELPERS_HPP
#define CUDA_HELPERS_HPP

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

namespace cudarf::detail {

inline const char *cuda_error_name(cudaError_t error) noexcept
{
    return cudaGetErrorName(error);
}

#ifdef CUDA_DRIVER_API
inline const char *cuda_error_name(CUresult error) noexcept
{
    const char *name = nullptr;
    cuGetErrorName(error, &name);
    return name ? name : "<unknown>";
}
#endif

template <typename T>
[[noreturn]]
inline void cuda_fail(T result,
                      const char *expression,
                      const char *file,
                      int line) noexcept
{
    std::fprintf(stderr,
                 "CUDA error at %s:%d: %s failed with code=%u (%s)\n",
                 file,
                 line,
                 expression,
                 static_cast<unsigned int>(result),
                 cuda_error_name(result));
    std::fflush(stderr);
    std::abort();
}

template <typename T>
inline void cuda_check(T result,
                       const char *expression,
                       const char *file,
                       int line) noexcept
{
    if (result != 0) {
        cuda_fail(result, expression, file, line);
    }
}

inline void cuda_check_kernel(cudaStream_t stream,
                              const char *kernelName,
                              const char *file,
                              int line) noexcept
{
    cuda_check(cudaGetLastError(), kernelName, file, line);

#ifndef NDEBUG
    cuda_check(cudaStreamSynchronize(stream),
               kernelName,
               file,
               line);
#else
    (void)stream;
#endif
}

} // namespace cudarf::detail

#define CUDA_CHK(val) \
    ::cudarf::detail::cuda_check((val), #val, __FILE__, __LINE__)

#define CUDA_CHK_KERNEL(stream, kernelName) \
    ::cudarf::detail::cuda_check_kernel((stream), (kernelName), __FILE__, __LINE__)

#endif
