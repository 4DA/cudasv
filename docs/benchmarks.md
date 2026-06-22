# Benchmarks

This document describes how the README benchmark table is produced. The goal is reproducibility and clear interpretation, not a broad performance claim.

# Current Benchmark

The public benchmark uses the sample pack at `assets/sample_pack_4cam` with the canonical four-camera rig, vehicle model, underlay, and viewpoint controls.

Measured system:

- GPU: NVIDIA GeForce RTX 4090 Laptop GPU.
- Driver: 575.57.08.
- CUDA toolkit/compiler recorded by CMake: 12.9.41.
- Resolution: 1920 x 1080.
- Build type: `Release`.
- CUDA profiling: enabled.
- TAA: disabled.

# Build

Build the benchmark configuration from the repository root:

```bash
source scripts/set_workspace.sh
set_build_type Release
set_cuda_profiling on
set_taa off
b
```

The profiling numbers are CUDA-event timings emitted by the built-in profiler. Profiling adds measurement overhead, so compare the two rows as benchmark-mode numbers rather than as maximum production throughput.

# Run

Run from the public sample pack directory:

```bash
cd assets/sample_pack_4cam
source ../../scripts/set_workspace.sh
sv_app --rig canonical-rig.json \
       --width 1920 \
       --height 1080 \
       --fps 1000 \
       --dump-frame /tmp/cudasv-bench.png \
       --dump-frame-number 40 \
       --frames right.png left.png front.png rear.png
```

To compare opaque paths, toggle `renderer_config.use_visibuf` in `config/overlays.json`:

- `1`: visibility-buffer opaque path.
- `0`: direct opaque raster path.

Restore the original config after each benchmark run if the change is only temporary.

# Reported Columns

The README table reports averages from the profiler output at frame 40, after the profiler history has warmed up. The relevant profiler event names are:

- `surround_view_projection`: CUDA projection of camera frames into the surround-view framebuffer.
- `scene_render`: opaque, translucent, and UI-overlay draw-list rendering.
- `compose`: final CUDA composition into the output buffer.
- `view_3d_total`: instrumented GPU time for the 3D view path.

`view_3d_total` excludes source image decode, GLFW presentation, CPU frame orchestration, and PNG dump overhead.

# Interpretation

The public sample shows the direct opaque and visibility-buffer opaque paths within a small margin of each other. On the measured laptop RTX 4090 configuration, the visibility-buffer row is slightly slower for the measured view.

This is expected for this stage of the renderer: the visibility-buffer path improves the architecture by separating visibility from material shading, but it still pays the same broad rasterization and scene submission costs. Future work such as material bucketing, queue/tile reset redesign, and higher-overdraw benchmark scenes would be needed before making stronger performance claims.

# Caveats

- The table is one GPU, one driver/toolkit stack, and one public sample scene.
- Results are CUDA-event timings, not end-to-end app frame times.
- Profiling builds include event instrumentation overhead.
- Different camera viewpoints, output resolutions, assets, or TAA settings may change the balance between paths.
