#ifndef RF_UTILS_HPP
#define RF_UTILS_HPP

#include <array>
#include <chrono>
#include <vector>

#include <cuda_runtime.h>
#include <rf/renderer/texture.hpp>

#include "glcommon.hpp"

namespace rf
{
using Duration = std::chrono::duration<double, std::milli>;
using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

void log_duration(const std::string &message,
                  const Duration& duration,
                  unsigned int indent = 0);

cudaTextureObject_t create_cuda_texture(rf::Image desc,
                                        cudaTextureAddressMode addressMode,
                                        cudaStream_t cuStream);
} // namespace rf


#endif
