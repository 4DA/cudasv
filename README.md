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
`cudaraster` code did not use that concept, while this renderer currently
relaunches the raster pipeline from the top for each draw-packet submission.
That is one of the reasons per-submission setup cost matters in the current
implementation.

## Shading / PBR

The current main shading model is a glTF-style metallic-roughness PBR path:

- GGX specular BRDF
- Lambertian diffuse
- Schlick Fresnel
- image-based lighting
  - spherical-harmonics diffuse
  - prefiltered specular cubemap + BRDF LUT
- base color and emissive texturing
- metallic / roughness material inputs

Material support exists for more inputs than are fully exercised by the current
fragment path, but the renderer is not yet a fully feature-complete glTF PBR
implementation.

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
  - PBR world-position interpolation still uses linear barycentrics in the
    fragment setup instead of perspective-correct interpolation
- the PBR path is still incomplete
  - normal maps are not part of the active shading path yet
  - metallic-roughness texturing is not fully wired through the fragment logic
  - derivative-based shading support is not available
  - some material inputs exist in the data model before they are fully honored
    in shading
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

## Future Directions

- improve raster robustness:
  - general triangle clipping
  - cleaner guard-band handling
  - continued warning cleanup and code simplification
- improve the current naive fine raster path:
  - reduce per-pixel sequential triangle scanning
  - move toward a more scalable tile-local fine raster stage
  - improve interpolation correctness in the PBR path
  - revisit translucent handling and depth/blend organization
- extend the PBR implementation:
  - normal maps
  - full metallic-roughness texture support
  - derivative-aware shading features where needed
  - tighter alignment between material data and active fragment evaluation
- replace remaining sample-asset compatibility shims with neutral project-owned assets
- extend dataset support beyond the current sample flow
- continue refining the CUDA raster path for modern GPUs and warp-level execution

## License

This repository is distributed under the GNU General Public License v3.0.
See `LICENSE`.

Some source files retain upstream third-party notices, especially in the CUDA
rasterization path derived from NVIDIA / `cudaraster` work. See
`THIRD_PARTY_NOTICES.md` and the
individual file headers for details.
