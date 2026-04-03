#include "sources/nuscenes_source.hpp"

#include <spdlog/spdlog.h>

namespace svapp
{

NuScenesSource::NuScenesSource(std::string datasetRoot, std::string sequenceId):
    _datasetRoot(std::move(datasetRoot)),
    _sequenceId(std::move(sequenceId))
{
    _info.kind = videoio::SourceKind::NuScenes;
    _info.source_name = "nuscenes";
    _info.dataset_root = _datasetRoot;
    _info.sequence_id = _sequenceId;
    _info.contract.synchronized_samples = true;
    _info.contract.provides_sample_identity = true;
    _info.contract.provides_source_timestamp = true;
    _info.contract.provides_per_camera_timestamps = true;
    _info.contract.provides_ego_pose = true;
}

bool NuScenesSource::open()
{
    SPDLOG_ERROR("NuScenesSource is not implemented yet [dataset_root='{}', sequence_id='{}']",
                 _datasetRoot,
                 _sequenceId);
    return false;
}

const camera::CameraRig& NuScenesSource::rig() const
{
    return _rig;
}

const videoio::SourceInfo& NuScenesSource::info() const
{
    return _info;
}

bool NuScenesSource::get_next_frame(videoio::FramePacket &packet)
{
    (void)packet;
    SPDLOG_ERROR("NuScenesSource::get_next_frame() called before implementation");
    return false;
}

bool NuScenesSource::release_frame(const videoio::FramePacket &packet)
{
    (void)packet;
    return true;
}

} // namespace svapp
