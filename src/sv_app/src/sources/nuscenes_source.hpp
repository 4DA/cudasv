#ifndef NUSCENES_SOURCE_HPP
#define NUSCENES_SOURCE_HPP

#include <vector>
#include <string>

#include <engine/frame_source.hpp>
#include <engine/image_frame_loader.hpp>

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
    struct CameraSample
    {
        camera::CameraRole role = camera::CameraRole::Unknown;
        std::string channel_name;
        std::string relative_path;
        std::string calibrated_sensor_token;
        uint32_t width = 0;
        uint32_t height = 0;
        uint64_t timestamp_ns = 0;
    };

    std::string _datasetRoot;
    std::string _dataRoot;
    std::string _sequenceId;
    std::string _versionRoot;
    std::string _resolvedSampleToken;
    std::string _resolvedSceneName;
    bool _opened = false;
    bool _decoded_sample_ready = false;
    uint64_t _frameId = 0;
    camera::CameraRig _rig;
    videoio::SourceInfo _info;
    std::vector<CameraSample> _cameraSamples;
    std::vector<videoio::DecodedImageFrame> _decodedFrames;
};

} // namespace svapp

#endif // NUSCENES_SOURCE_HPP
