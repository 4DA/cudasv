#ifndef CUDARF_MEMORY_HPP
#define CUDARF_MEMORY_HPP

#include <cassert>
#include <cstddef>
#include <utility>

#include <cuda_runtime.h>

#include <rf/renderer/cuda_helpers.hpp>

namespace cudarf
{
namespace memory
{
template <class T>
struct DeviceBuffer {
    DeviceBuffer(): _devPtr(nullptr), _byteSize(0), _count(0) {}
    explicit DeviceBuffer(std::size_t count);
    DeviceBuffer(DeviceBuffer &&other) noexcept;
    DeviceBuffer &operator=(DeviceBuffer &&other) noexcept;
    ~DeviceBuffer();

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    T *get() const { return _devPtr; }
    std::size_t size() const { return _byteSize; }
    std::size_t count() const { return _count; }
    explicit operator bool() const { return _devPtr != nullptr; }
    void reset(std::size_t count = 0);

private:
    T *_devPtr;
    std::size_t _byteSize;
    std::size_t _count;
};

template <typename T>
DeviceBuffer<T>::DeviceBuffer(std::size_t count)
{
    _count = count;
    assert(_count > 0);
    _byteSize = _count * sizeof(T);
    CUDA_CHK(cudaMalloc((void **)&_devPtr, _byteSize));
}

template <typename T>
DeviceBuffer<T>::DeviceBuffer(DeviceBuffer &&other) noexcept
{
    _devPtr = std::exchange(other._devPtr, nullptr);
    _byteSize = std::exchange(other._byteSize, 0);
    _count = std::exchange(other._count, 0);
}

template <typename T>
DeviceBuffer<T> &DeviceBuffer<T>::operator=(DeviceBuffer &&other) noexcept
{
    if (this != &other) {
        reset();
        _devPtr = std::exchange(other._devPtr, nullptr);
        _byteSize = std::exchange(other._byteSize, 0);
        _count = std::exchange(other._count, 0);
    }
    return *this;
}

template <typename T>
void DeviceBuffer<T>::reset(std::size_t count)
{
    if (_devPtr) {
        CUDA_CHK(cudaFree(_devPtr));
        _devPtr = nullptr;
    }

    _count = count;
    _byteSize = _count * sizeof(T);
    if (_count) {
        CUDA_CHK(cudaMalloc((void **)&_devPtr, _byteSize));
    }
}

template <typename T>
DeviceBuffer<T>::~DeviceBuffer()
{
    reset();
}

template <class T>
struct PinnedBuffer {
    PinnedBuffer(): _hostPtr(nullptr), _byteSize(0), _count(0) {}
    explicit PinnedBuffer(std::size_t count);
    PinnedBuffer(PinnedBuffer &&other) noexcept;
    PinnedBuffer &operator=(PinnedBuffer &&other) noexcept;
    ~PinnedBuffer();

    PinnedBuffer(const PinnedBuffer &) = delete;
    PinnedBuffer &operator=(const PinnedBuffer &) = delete;

    T *get() const { return _hostPtr; }
    std::size_t size() const { return _byteSize; }
    std::size_t count() const { return _count; }
    explicit operator bool() const { return _hostPtr != nullptr; }
    void reset(std::size_t count = 0);

private:
    T *_hostPtr = nullptr;
    std::size_t _byteSize = 0;
    std::size_t _count = 0;
};

template <typename T>
PinnedBuffer<T>::PinnedBuffer(std::size_t count)
{
    _count = count;
    assert(_count > 0);
    _byteSize = _count * sizeof(T);
    CUDA_CHK(cudaMallocHost((void **)&_hostPtr, _byteSize));
}

template <typename T>
PinnedBuffer<T>::PinnedBuffer(PinnedBuffer &&other) noexcept
{
    _hostPtr = std::exchange(other._hostPtr, nullptr);
    _byteSize = std::exchange(other._byteSize, 0);
    _count = std::exchange(other._count, 0);
}

template <typename T>
PinnedBuffer<T> &PinnedBuffer<T>::operator=(PinnedBuffer &&other) noexcept
{
    if (this != &other) {
        reset();
        _hostPtr = std::exchange(other._hostPtr, nullptr);
        _byteSize = std::exchange(other._byteSize, 0);
        _count = std::exchange(other._count, 0);
    }
    return *this;
}

template <typename T>
void PinnedBuffer<T>::reset(std::size_t count)
{
    if (_hostPtr) {
        CUDA_CHK(cudaFreeHost(_hostPtr));
        _hostPtr = nullptr;
    }

    _count = count;
    _byteSize = _count * sizeof(T);
    if (_count) {
        CUDA_CHK(cudaMallocHost((void **)&_hostPtr, _byteSize));
    }
}

template <typename T>
PinnedBuffer<T>::~PinnedBuffer()
{
    reset();
}

} // namespace memory

} // namespace cudarf

#endif
