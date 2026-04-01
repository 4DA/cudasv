#include "app/cmdline.hpp"

#include <algorithm>
#include <cstring>

#include <spdlog/spdlog.h>

namespace svapp
{

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
