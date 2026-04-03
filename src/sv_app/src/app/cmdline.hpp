#pragma once

#include <array>
#include <string>

#include <engine/engine.hpp>
#include <engine/frame_source.hpp>

namespace svapp
{

struct CmdlineOpts
{
    videoio::SourceKind source_kind = videoio::SourceKind::FileSequence;
    std::string dataset_root;
    std::string sequence_id;
    std::array<std::string, camera::CAMERAS_TOTAL> files;
    std::string rig_file = "canonical-rig.json";
    int width = 0;
    int height = 0;
    int fps = 1000;
};

int parse_cmdline(int argc, char **argv, CmdlineOpts &options);
bool has_frame_inputs(const CmdlineOpts &options);
engine::OutputSet make_single_output_set(int width, int height);

} // namespace svapp
