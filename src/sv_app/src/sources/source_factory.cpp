#include "sources/source_factory.hpp"

#include <memory>

#include "sources/file_sequence_source.hpp"

namespace svapp
{

std::unique_ptr<videoio::FrameSource> create_source(const SourceFactoryConfig &config)
{
    if (config.source_kind != videoio::SourceKind::FileSequence) {
        return nullptr;
    }

    FileSequenceSourceConfig file_config;
    file_config.dataset_root = config.dataset_root;
    file_config.sequence_id = config.sequence_id;
    file_config.frame_paths = config.frame_paths;
    file_config.rig_path = config.rig_path;

    return std::make_unique<FileSequenceSource>(std::move(file_config));
}

} // namespace svapp
