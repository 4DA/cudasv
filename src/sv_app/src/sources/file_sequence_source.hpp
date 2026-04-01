#ifndef FILE_SEQUENCE_SOURCE_HPP
#define FILE_SEQUENCE_SOURCE_HPP

#include <array>
#include <string>

#include <engine/frame_source.hpp>
#include <engine/png_source.hpp>

namespace svapp
{

struct FileSequenceSourceConfig
{
    std::string dataset_root;
    std::string sequence_id;
    std::array<std::string, camera::CAMERAS_TOTAL> frame_paths;
    std::string rig_path;
};

class FileSequenceSource final: public videoio::FrameSource
{
public:
    explicit FileSequenceSource(FileSequenceSourceConfig config);

    bool open() override;
    const camera::CameraRig& rig() const override;
    const videoio::SourceInfo& info() const override;
    bool get_next_frame(videoio::FramePacket &packet) override;
    bool release_frame(const videoio::FramePacket &packet) override;

private:
    FileSequenceSourceConfig _config;
    camera::CameraRig _rig;
    videoio::SourceInfo _info;
    videoio::PNGSource _png_source;
    uint64_t _frame_id = 0;
};

} // namespace svapp

#endif // FILE_SEQUENCE_SOURCE_HPP
