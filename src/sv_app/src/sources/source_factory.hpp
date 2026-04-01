#ifndef SOURCE_FACTORY_HPP
#define SOURCE_FACTORY_HPP

#include <array>
#include <memory>
#include <string>

#include <engine/frame_source.hpp>

namespace svapp
{

struct SourceFactoryConfig
{
    videoio::SourceKind source_kind = videoio::SourceKind::FileSequence;
    std::string dataset_root;
    std::string sequence_id;
    std::array<std::string, camera::CAMERAS_TOTAL> frame_paths;
    std::string rig_path;
};

std::unique_ptr<videoio::FrameSource> create_source(const SourceFactoryConfig &config);

} // namespace svapp

#endif // SOURCE_FACTORY_HPP
