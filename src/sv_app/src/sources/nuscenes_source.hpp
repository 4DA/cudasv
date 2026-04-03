#ifndef NUSCENES_SOURCE_HPP
#define NUSCENES_SOURCE_HPP

#include <string>

#include <engine/frame_source.hpp>

namespace svapp
{

class NuScenesSource final: public videoio::FrameSource
{
public:
    NuScenesSource(std::string datasetRoot, std::string sequenceId);

    bool open() override;
    const camera::CameraRig& rig() const override;
    const videoio::SourceInfo& info() const override;
    bool get_next_frame(videoio::FramePacket &packet) override;
    bool release_frame(const videoio::FramePacket &packet) override;

private:
    std::string _datasetRoot;
    std::string _sequenceId;
    camera::CameraRig _rig;
    videoio::SourceInfo _info;
};

} // namespace svapp

#endif // NUSCENES_SOURCE_HPP
