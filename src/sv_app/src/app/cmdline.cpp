#include "app/cmdline.hpp"

#include <algorithm>
#include <cstring>

#include <spdlog/spdlog.h>

namespace svapp
{

namespace
{

videoio::SourceKind parse_source_kind(const char *value)
{
    if (!strcmp(value, "file_sequence")) {
        return videoio::SourceKind::FileSequence;
    }

    if (!strcmp(value, "nuscenes")) {
        return videoio::SourceKind::NuScenes;
    }

    return videoio::SourceKind::Unknown;
}

} // namespace

int parse_cmdline(int argc, char **argv, CmdlineOpts &options)
{
    for (int index = 1; index < argc; ++index) {
        if (!strcmp(argv[index], "--frames")) {
            if (index + camera::CAMERAS_TOTAL >= argc) {
                SPDLOG_ERROR("--frames requires exactly {} file arguments",
                             static_cast<int>(camera::CAMERAS_TOTAL));
                return -1;
            }

            options.files[0] = argv[++index];
            options.files[1] = argv[++index];
            options.files[2] = argv[++index];
            options.files[3] = argv[++index];
        } else if (!strcmp(argv[index], "--fps")) {
            if (index + 1 >= argc) {
                SPDLOG_ERROR("--fps requires a value");
                return -1;
            }

            options.fps = atoi(argv[++index]);
        } else if (!strcmp(argv[index], "--rig")) {
            if (index + 1 >= argc) {
                SPDLOG_ERROR("--rig requires a file path");
                return -1;
            }

            options.rig_file = argv[++index];
        } else if (!strcmp(argv[index], "--dump-frame")) {
            if (index + 1 >= argc) {
                SPDLOG_ERROR("--dump-frame requires a file path");
                return -1;
            }

            options.dump_frame_path = argv[++index];
        } else if (!strcmp(argv[index], "--source-kind")) {
            if (index + 1 >= argc) {
                SPDLOG_ERROR("--source-kind requires a value");
                return -1;
            }

            options.source_kind = parse_source_kind(argv[++index]);
            if (options.source_kind == videoio::SourceKind::Unknown) {
                SPDLOG_ERROR("Unsupported --source-kind value");
                return -1;
            }
        } else if (!strcmp(argv[index], "--dataset-root")) {
            if (index + 1 >= argc) {
                SPDLOG_ERROR("--dataset-root requires a value");
                return -1;
            }

            options.dataset_root = argv[++index];
        } else if (!strcmp(argv[index], "--sequence-id")) {
            if (index + 1 >= argc) {
                SPDLOG_ERROR("--sequence-id requires a value");
                return -1;
            }

            options.sequence_id = argv[++index];
        } else if (!strcmp(argv[index], "--width")) {
            if (index + 1 >= argc) {
                SPDLOG_ERROR("--width requires a value");
                return -1;
            }

            options.width = atoi(argv[++index]);
        } else if (!strcmp(argv[index], "--height")) {
            if (index + 1 >= argc) {
                SPDLOG_ERROR("--height requires a value");
                return -1;
            }

            options.height = atoi(argv[++index]);
        }
    }

    return 0;
}

bool has_frame_inputs(const CmdlineOpts &options)
{
    return std::all_of(options.files.begin(), options.files.end(), [](const std::string &path) {
        return !path.empty();
    });
}

bool validate_source_inputs(const CmdlineOpts &options)
{
    switch (options.source_kind) {
    case videoio::SourceKind::FileSequence:
        if (!has_frame_inputs(options)) {
            SPDLOG_ERROR("--frames is mandatory for source-kind=file_sequence and requires exactly {} PNG files",
                         static_cast<int>(camera::CAMERAS_TOTAL));
            return false;
        }
        return true;
    case videoio::SourceKind::NuScenes:
        if (options.dataset_root.empty()) {
            SPDLOG_ERROR("--dataset-root is mandatory for source-kind=nuscenes");
            return false;
        }
        if (options.sequence_id.empty()) {
            SPDLOG_ERROR("--sequence-id is mandatory for source-kind=nuscenes");
            return false;
        }
        return true;
    case videoio::SourceKind::Unknown:
    default:
        SPDLOG_ERROR("Unsupported source configuration");
        return false;
    }
}

engine::OutputSet make_single_output_set(int width, int height)
{
    engine::OutputSet output_set = {};
    output_set.outputs[0].active = 1;
    output_set.outputs[0].views_count = 1;
    output_set.outputs[0].active_view_ids[0] = engine::view::SV_VIEW_3D;
    output_set.outputs[0].config.display_width = width;
    output_set.outputs[0].config.display_height = height;
    output_set.outputs[0].name = "cudasv_main";
    return output_set;
}

} // namespace svapp
