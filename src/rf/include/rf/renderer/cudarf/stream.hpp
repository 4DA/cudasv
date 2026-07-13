#ifndef CUDARF_STREAM_HPP
#define CUDARF_STREAM_HPP

#include <utility>

#include <cuda_runtime.h>

#include <rf/renderer/cuda_helpers.hpp>

namespace cudarf
{
struct Stream {
    Stream() = default;

    explicit Stream(unsigned int flags)
    {
        CUDA_CHK(cudaStreamCreateWithFlags(&_stream, flags));
    }

    Stream(const Stream &) = delete;
    Stream &operator=(const Stream &) = delete;

    Stream(Stream &&other) noexcept:
        _stream(std::exchange(other._stream, nullptr)) {}

    Stream &operator=(Stream &&other) noexcept
    {
        if (this != &other) {
            destroy();
            _stream = std::exchange(other._stream, nullptr);
        }
        return *this;
    }

    ~Stream() {destroy();}

    cudaStream_t get() const {return _stream;}

private:
    void destroy() noexcept
    {
        if (_stream) {
            CUDA_CHK(cudaStreamDestroy(_stream));
            _stream = nullptr;
        }
    }

    cudaStream_t _stream = nullptr;
};
}

#endif
