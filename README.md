# cudaSV

`cudaSV` is a CUDA-first surround-view renderer.

The project uses CUDA in two places:

1. surround-view stitching from 4 camera inputs onto the ground-plane / underlay
2. GPU rasterization of the 3D scene, including the vehicle model and overlays

The current sample app composes a stitched surround view with a rendered 3D
vehicle and interactive viewpoint controls.

## What Is In The Repo

- `src/rf/`: CUDA rasterizer and rendering core
- `src/sv_engine/`: surround-view engine, scene/view orchestration, vehicle state
- `src/sv_app/`: sample desktop app
- `src/engine_interface/`: shared runtime types
- `assets/sample_pack_4cam/`: public sample assets and config
- `scripts/`: developer workflow helpers such as build environment setup
- `docs/schema/rig.schema.json`: canonical rig schema

## Camera Rig

The current public runtime path uses a 4-camera surround-view rig with explicit
`right`, `left`, `front`, and `rear` roles. The sample rig is stored as a
canonical JSON file and uses fisheye cameras with `fisheye_polynomial4`
distortion, matching the four-coefficient Kannala-Brandt fisheye model used by
the projection path.

Rig extrinsics are expressed in vehicle coordinates with
`pose_vehicle.rotation` as a 3x3 matrix and `pose_vehicle.translation` as a
3-vector. See `docs/schema/rig.schema.json` for the schema and
`docs/camera_rig_debug.md` for validation/debug views.

## Rasterization Notes

The rasterizer is heavily based on the ideas and code structure of
`cudaraster`, especially:

- bin tiling
- coarse tiling
- triangle setup / transform stages

References:

- "High-Performance Software Rasterization on GPUs", Samuli Laine and Tero Karras, HPG 2011
- cudaraster archive: https://code.google.com/archive/p/cudaraster/

This codebase does not treat that code as a frozen import. The coarse and bin
tilers were adapted for modern GPUs with independent thread scheduling, using
explicit warp-synchronization where the original code relied on older warp
execution behavior.

One notable divergence is the current `drawPacket` submission model. The old
`cudaraster` code did not use that concept. This renderer keeps uploaded mesh
geometry in draw packets and submits draw-packet id lists into the CUDA raster
pipe. Each raster submission still starts at the vertex, triangle setup, bin
tiler, coarse tiler, and fine raster stages, so per-submission setup and reset
costs still matter.

## Shading / PBR

The current main shading model is a glTF-style metallic-roughness PBR path:

- GGX specular BRDF
- Lambertian diffuse
- Schlick Fresnel
- image-based lighting
  - spherical-harmonics diffuse
  - prefiltered specular cubemap + BRDF LUT
- base-color, normal, emissive, and metallic-roughness texturing
- explicit mipmapped texture sampling with derivative-based LOD selection for
  the active material texture path
- CUDA-side mip generation for loaded material textures

The material path is intended to be practical glTF-style PBR rather than a
fully feature-complete glTF renderer. Some glTF inputs are deliberately not
implemented or are handled differently from a reference renderer, but the core
metallic-roughness texture path and mip-aware material sampling are active.

## Current Deliberate Limitations

Some parts of the raster path are intentionally simplified because the current
scene and camera setup are controlled:

- triangle setup does not implement general clipping yet
  - triangles fully outside the frustum are rejected
  - partially clipped triangles are not split against the frustum planes
- the coarse tiler has a simplified opaque path
  - opaque work is pushed directly into tile queues
  - blended work uses the heavier warp-mask emission path
- the current fine raster path is intentionally naive
  - each pixel scans the triangle list of its tile sequentially
  - opaque shading assumes the fragment shader does not modify depth
  - the opaque path shades only the top fragment after an early-Z style pass
  - translucent work shades per fragment in the same pass
- the PBR path is still incomplete
  - it is not a complete glTF material implementation
  - some material inputs exist in the data model before they are fully honored
    in shading
  - glTF occlusion texture sampling is not currently part of the material path;
    future ambient occlusion should be renderer-owned rather than only
    material-texture-driven
- several robustness/performance TODOs are still explicit in the raster code
  - guard-band / sample-edge cases
  - more complete clipping
  - some cleanup in triangle and tiler stages

These choices were useful while understanding and reshaping the renderer, but
they should evolve into a more general and more polished implementation.

## Why CUDA Here

The project is intentionally not "CUDA for one small kernel around a normal
renderer". CUDA is used as the main execution substrate for:

- camera projection and surround-view stitching
- tile/bin based scene rasterization
- post-process stages such as TAA/composition

That is the core identity of the project.

## System Requirements

The project is currently built and run as a Linux desktop CUDA/OpenGL ES
application.

Build tools:

- CMake 3.16 or newer
- Make, via the current `scripts/set_workspace.sh` workflow
- a C++20-capable host compiler compatible with the installed CUDA Toolkit
- CUDA Toolkit with `nvcc`
- an NVIDIA driver new enough for the installed CUDA Toolkit

GPU/runtime:

- NVIDIA GPU with CUDA support
- CUDA runtime and driver libraries
- the default CMake CUDA architecture is `89`; set
  `CMAKE_CUDA_ARCHITECTURES` explicitly if building for another GPU generation

System libraries discovered by CMake:

- EGL
- OpenGL ES 3
- GLFW 3
- GLM
- FFmpeg libraries:
  - `libavformat`
  - `libavcodec`
  - `libavutil`
  - `libswscale`

Vendored third-party code includes `spdlog`, `nlohmann/json`, and `tinygltf`.

## Bootstrap And Dependencies

Clone the repository with submodules enabled:

```bash
git clone --recurse-submodules <repo-url> cudaSV
cd cudaSV
```

If the repository was already cloned without submodules, initialize them before
building:

```bash
git submodule update --init --recursive
```

The current submodules are:

- `thirdparty/json`
- `thirdparty/spdlog`
- `thirdparty/tinygltf`

On a typical Ubuntu system, install the host build tools and system development
libraries with:

```bash
sudo apt update
sudo apt install \
  build-essential \
  cmake \
  git \
  make \
  pkg-config \
  ffmpeg \
  libavcodec-dev \
  libavformat-dev \
  libavutil-dev \
  libswscale-dev \
  libegl1-mesa-dev \
  libgles2-mesa-dev \
  libglfw3-dev \
  libglm-dev
```

Install the NVIDIA driver and CUDA Toolkit separately if they are not already
available on the machine. The build expects `nvcc` to be on `PATH` or available
under a standard CUDA installation path such as `/usr/local/cuda/bin`.

## Build

```bash
source scripts/set_workspace.sh
b
```

To switch build type in the current shell:

```bash
source scripts/set_workspace.sh
set_build_type Release
b
```

Supported values:

- `Debug`
- `Release`
- `RelWithDebInfo`
- `MinSizeRel`

This configures the project in `out/` and builds the binaries into:

- `out/usr/bin/`
- `out/usr/lib/`

To clean the build:

```bash
source scripts/set_workspace.sh
c
```

For compatibility, the repo root also provides a thin `set_workspace.sh` wrapper
that forwards to `scripts/set_workspace.sh`.

## Run

Run the sample app from the public sample pack directory:

```bash
cd assets/sample_pack_4cam
source ../../scripts/set_workspace.sh
sv_app --frames right.png left.png front.png rear.png \
       --rig canonical-rig.json \
       --width 1920 \
       --height 1080
```

To dump a specific rendered frame to PNG:

```bash
sv_app --frames right.png left.png front.png rear.png \
       --rig canonical-rig.json \
       --width 1920 \
       --height 1080 \
       --dump-frame /tmp/cudasv.png \
       --dump-frame-number 8
```

If `--dump-frame-number` is omitted, frame `0` is dumped.

There is also a sample launcher script in the sample pack:

```bash
cd assets/sample_pack_4cam
./run.sh
```

## Controls

- Hold the left mouse button and move the mouse to pan the active view.
- Right-click a viewpoint control in the 3D view to activate that viewpoint.
- Use the mouse wheel to zoom when the active viewpoint supports it.

## Future Directions

- improve raster robustness:
  - general triangle clipping
  - cleaner guard-band handling
  - continued warning cleanup and code simplification
- improve the current naive fine raster path:
  - reduce per-pixel sequential triangle scanning
  - move toward a more scalable tile-local fine raster stage
  - revisit translucent handling and depth/blend organization
- extend the PBR implementation:
  - continue tightening material-feature coverage beyond the active
    metallic-roughness, normal, emissive, and mip-aware texture paths
  - validate and benchmark remaining material variants before adding new shader
    specializations
  - tighter alignment between material data and active fragment evaluation
- replace remaining sample-asset compatibility shims with neutral project-owned assets
- extend the nuScenes source and inspector path toward renderer integration
- continue refining the CUDA raster path for modern GPUs and warp-level execution

## License

This repository is distributed under the GNU General Public License v3.0.
See `LICENSE`.

Some source files retain upstream third-party notices, especially in the CUDA
rasterization path derived from NVIDIA / `cudaraster` work. See
`THIRD_PARTY_NOTICES.md` and the
individual file headers for details.
