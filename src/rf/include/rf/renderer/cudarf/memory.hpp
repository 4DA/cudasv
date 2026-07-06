#ifndef CUDARF_MEMORY_HPP
#define CUDARF_MEMORY_HPP

#include <cassert>
#include <cstddef>
#include <utility>

#include <cuda_runtime.h>

namespace cudarf
{
template <class T>
struct DeviceBuffer {
    DeviceBuffer(): _devPtr(nullptr), _byteSize(0), _count(0) {}
    DeviceBuffer(unsigned int count);
    DeviceBuffer(DeviceBuffer &&other) noexcept;
    DeviceBuffer &operator=(DeviceBuffer &&other) noexcept;
    ~DeviceBuffer();

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    T *get() const { return _devPtr; }
    std::size_t size() const { return _byteSize; }
    unsigned int count() const { return _count; }
    explicit operator bool() const { return _devPtr != nullptr; }
    void reset(unsigned int count = 0);

private:
    T *_devPtr;
    std::size_t _byteSize;
    unsigned int _count;
};

template <typename T>
DeviceBuffer<T>::DeviceBuffer(unsigned int count)
{
    _count = count;
    assert(_count > 0);
    _byteSize = _count * sizeof(T);
    cudaError_t err = cudaMalloc((void **)&_devPtr, _byteSize);
    assert(err == cudaSuccess);
    (void)err;
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
void DeviceBuffer<T>::reset(unsigned int count)
{
    if (_devPtr) {
        cudaError_t err = cudaFree(_devPtr);
        assert(err == cudaSuccess);
        (void)err;
        _devPtr = nullptr;
    }

    _count = count;
    _byteSize = _count * sizeof(T);
    if (_count) {
        cudaError_t err = cudaMalloc((void **)&_devPtr, _byteSize);
        assert(err == cudaSuccess);
        (void)err;
    }
}

template <typename T>
DeviceBuffer<T>::~DeviceBuffer()
{
    reset();
}

}

#endif
