#ifndef CANONICAL_RIG_HPP
#define CANONICAL_RIG_HPP

#include <string>

#include <engine/camera_rig.hpp>

// Canonical rig JSON is the project-owned runtime calibration format.
//
// High-level shape:
// - top-level metadata such as schema_version, rig_name, and coordinate_system
// - a cameras[] array instead of hardcoded per-role top-level objects
// - each camera entry carries:
//   - id
//   - role
//   - projection_model
//   - image_size
//   - intrinsics
//   - distortion
//   - pose_vehicle
//
// Normalization rules:
// - pose_vehicle.rotation is stored as a readable 3x3 row-wise matrix in JSON
// - runtime Extrinsics::R is column-major, so the loader repacks the matrix
// - compact/import-specific calibration encodings should be normalized offline
//   into this format rather than carried into the runtime schema
//
int load_canonical_rig(camera::CameraRig *rig,
                       const std::string &path);

#endif // CANONICAL_RIG_HPP
