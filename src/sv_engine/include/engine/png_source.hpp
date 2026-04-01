#ifndef PNG_SOURCE_HPP
#define PNG_SOURCE_HPP

#include <array>
#include <memory>
#include <string>

#include <engine/camera_config.hpp>
#include <engine/video_source.hpp>

namespace videoio
{

class PNGSource final: public VideoSource<camera::CAMERAS_TOTAL>
{
public:
    PNGSource();
    ~PNGSource() override;

    PNGSource(const PNGSource &) = delete;
    PNGSource& operator=(const PNGSource &) = delete;

    bool start(std::array<std::string, camera::CAMERAS_TOTAL> sources);
    bool get_next_frame(FrameSet<camera::CAMERAS_TOTAL> &frames) override;
    bool release_frame(const FrameSet<camera::CAMERAS_TOTAL> &frame_set) override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

}

#endif // PNG_SOURCE_HPP
