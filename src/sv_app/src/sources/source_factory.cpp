#include "sources/source_factory.hpp"

#include <memory>

#include <spdlog/spdlog.h>

#include "sources/file_sequence_source.hpp"
#include "sources/nuscenes_source.hpp"

namespace svapp
{

namespace
{

const char *source_kind_name(videoio::SourceKind kind)
{
    switch (kind) {
    case videoio::SourceKind::FileSequence:
        return "file_sequence";
    case videoio::SourceKind::NuScenes:
        return "nuscenes";
    case videoio::SourceKind::Unknown:
    default:
        return "unknown";
    }
}

} // namespace

std::unique_ptr<videoio::FrameSource> create_source(const SourceFactoryConfig &config)
{
    switch (config.source_kind) {
    case videoio::SourceKind::FileSequence: {
        FileSequenceSourceConfig file_config;
        file_config.dataset_root = config.dataset_root;
        file_config.sequence_id = config.sequence_id;
        file_config.frame_paths = config.frame_paths;
        file_config.rig_path = config.rig_path;

        return std::make_unique<FileSequenceSource>(std::move(file_config));
    }
    case videoio::SourceKind::NuScenes:
        return std::make_unique<NuScenesSource>(config.dataset_root, config.sequence_id);
    case videoio::SourceKind::Unknown:
    default:
        SPDLOG_ERROR("Source kind '{}' is not supported",
                     source_kind_name(config.source_kind));
        return nullptr;
    }
}

} // namespace svapp
