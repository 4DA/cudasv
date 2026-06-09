# Camera Rig Debug

This note lists the available facilities for debugging incorrect camera intrinsics,
extrinsics, role mapping, and surround-view projection behavior.

It should be updated whenever a new camera-rig or surround-view debug aid is added.

## Facilities

### 1. Source And Rig Validation Log

Files:

- [main.cpp](src/sv_app/src/main.cpp)
- [source_validation.cpp](src/sv_app/src/sources/source_validation.cpp)

What it shows:

- source kind / name / dataset root / sequence id
- sequence frame count when known
- source render-role order
- each rig camera id / role / projection model / image size
- each rig camera translation in vehicle coordinates
- each rig camera right / up / forward axis in vehicle coordinates
- fixed 4-camera bridge compatibility
- uniform image size expectation for the bridge

What errors it helps catch:

- wrong source camera ordering
- wrong role assignment
- missing required `front/rear/left/right` cameras
- duplicate roles
- inconsistent image sizes across render-bridge cameras
- obviously wrong extrinsics orientation
- row/column confusion in pose interpretation

### 2. Camera-Role Debug View

Files:

- [overlays_config.hpp](src/sv_engine/include/engine/overlays_config.hpp)
- [surround_view_composer.cpp](src/sv_engine/src/views/surround_view_composer.cpp)
- [projector.cu](src/rf/src/renderer/surround_view/projector.cu)

Config:

```json
"surround_view_debug_config": {
  "mode": 1
}
```

What it visualizes:

- front camera region as red
- left camera region as green
- right camera region as blue
- rear camera region as yellow

What errors it helps catch:

- swapped camera mapping
- overlap regions on the wrong side of the vehicle
- clearly wrong front/rear or left/right assignment
- overlap-anchor placement that looks obviously off

Typical bad result:

- if red appears where the right camera should dominate, the role mapping or pose is likely wrong

### 3. Coverage Mask Debug View

Files:

- [overlays_config.hpp](src/sv_engine/include/engine/overlays_config.hpp)
- [projector.cu](src/rf/src/renderer/surround_view/projector.cu)

Config:

```json
"surround_view_debug_config": {
  "mode": 2
}
```

What it visualizes:

- black: no camera produces a valid projection
- blue-ish: one camera is valid
- orange: two cameras are valid
- magenta: three cameras are valid
- white: four or more cameras are valid

What errors it helps catch:

- missing coverage
- overlap regions that are too small or too large
- projection validity failures
- unexpected holes near the vehicle footprint

Important detail:

- this is a projection-validity mask, not an occlusion mask
- it counts how many cameras can validly sample a point
- it does not identify which specific cameras overlap

Typical bad result:

- large black holes where you expect coverage usually indicate bad extrinsics, bad intrinsics, or a bad role/footprint assumption

### 4. Camera-Image UV Reprojection Grid

Files:

- [overlays_config.hpp](src/sv_engine/include/engine/overlays_config.hpp)
- [surround_view_composer.cpp](src/sv_engine/src/views/surround_view_composer.cpp)
- [projector.cu](src/rf/src/renderer/surround_view/projector.cu)

Config:

```json
"surround_view_debug_config": {
  "mode": 3,
  "camera_role": "front"
}
```

Supported `camera_role` values:

- `front`
- `rear`
- `left`
- `right`

What it visualizes:

- the selected camera's sampled image on the stitched surface
- a white grid in normalized camera-image UV space
- black where the selected camera has no valid projection

What errors it helps catch:

- bad camera extrinsics
- wrong projection direction
- bad fisheye intrinsics/distortion interpretation
- wrong role assignment for the selected camera
- sampling that lands in unexpected regions of the stitched surface

Important detail:

- this is a camera-image UV grid, not a world-space ground grid

Typical bad result:

- a badly warped or obviously misplaced UV grid usually means the projection math or camera pose is wrong
- mostly black output for a camera that should see that region indicates invalid projection or incorrect camera selection/mapping

### 5. Configurable Blend Profile

Files:

- [overlays_config.hpp](src/sv_engine/include/engine/overlays_config.hpp)
- [config.hpp](src/sv_app/src/config/config.hpp)
- [surround_view_composer.cpp](src/sv_engine/src/views/surround_view_composer.cpp)
- [projector.cu](src/rf/src/renderer/surround_view/projector.cu)

Config block:

```json
"surround_view_blend_config": {
  "front_left_start_deg": 35.0,
  "front_left_end_deg": 55.0,
  "right_front_start_deg": 295.0,
  "right_front_end_deg": 315.0,
  "rear_right_start_deg": 215.0,
  "rear_right_end_deg": 235.0,
  "left_rear_start_deg": 125.0,
  "left_rear_end_deg": 145.0,
  "front_anchor_offset_mm": 0.0,
  "rear_anchor_offset_mm": 0.0,
  "left_anchor_offset_mm": 0.0,
  "right_anchor_offset_mm": 0.0
}
```

What it is for:

- tuning overlap windows without editing kernel code
- shifting overlap anchors independently on each side of the vehicle

What errors it helps isolate:

- when coverage is valid but the transition region is centered in the wrong place
- when vehicle dimensions alone are not the right overlap anchor

Typical bad result:

- if role-color or coverage views look geometrically reasonable but the blend seam is clearly centered wrong, the blend profile is the next place to tune

## Current Gaps

Not implemented yet:

- world-space ground-grid reprojection
- rendered frustum visualization in the 3D scene
- per-camera overlap identity mask that shows exactly which cameras overlap
