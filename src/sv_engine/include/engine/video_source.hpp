#ifndef VIDEO_SOURCE_HPP
#define VIDEO_SOURCE_HPP

#include <array>
#include <cstdint>
#include <chrono>
#include <cstdint>

namespace videoio
{

template <unsigned int N>
struct FrameSet {
    std::array<uint8_t *, N> data = {nullptr};
    uint64_t timestamp;

    uint32_t width;
    uint32_t height;
    uint32_t stride;

    uint64_t frameseq = -1;

    std::array<void *, N> userdata = {nullptr};
};

template <unsigned int N>
struct VideoSource {
    virtual ~VideoSource() = default;
    virtual bool get_next_frame(FrameSet<N> &) = 0;
    virtual bool release_frame(const FrameSet<N> &) = 0;
};

inline uint64_t monotonic_ts_now() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

}

#endif
