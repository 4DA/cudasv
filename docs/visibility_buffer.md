# Visibility Buffer Path

This renderer has two opaque scene paths: a direct opaque raster path and a visibility-buffer opaque path. They share the same scene, draw-list, material, and CUDA raster infrastructure, but split the work differently.

# Paths

The direct opaque path resolves visibility and shades fragments in the fine raster stage. It is the comparison path for visibility-buffer correctness and performance checks.

The visibility-buffer path separates visibility from material evaluation:

1. Rasterization finds the visible primitive/triangle winner for an output pixel.
2. A material pass shades the surviving visible samples using the stored winner data.

The path is selected at runtime with `renderer_config.use_visibuf` in `assets/sample_pack_4cam/config/overlays.json`.

# Why It Exists

The visibility-buffer path is intended to decouple visibility from shading. That gives the renderer a clearer boundary between raster work and material/PBR work, and it avoids running expensive material evaluation for fragments that lose visibility.

This is a correctness and architecture path first, not a broad performance claim. The benchmark shows the two paths close to parity on the public sample scene, with the visibility-buffer path slightly slower on the measured GPU/view.

# Stored Data

The visibility-buffer path stores enough information for a later material pass to reconstruct shading inputs for the visible winner. Conceptually, this includes the winning primitive/triangle identity, material identity, depth/visibility state, and interpolation data needed by the material pass.

The exact storage is implementation-specific and lives inside the CUDA raster path. The important contract is that the material pass shades only the selected visible winners.

# Limits

The visibility-buffer path uses the same binning, coarse tiling, triangle setup, and fine raster infrastructure as the direct opaque path. It does not remove all overdraw cost, and it does not replace clipping or broader scene submission work.

Limitations to keep in mind:

- Benchmarks are from one public sample scene and one GPU.
- The path is close to performance parity with direct opaque, not a demonstrated universal speedup.
- General triangle clipping limitations apply.
- The direct opaque path remains useful for comparison and regression checks.

# Code Map

Key files:

- `src/rf/src/renderer/cudarf/cudarf_pipe.cu`: raster pipe orchestration and profiling events.
- `src/rf/src/renderer/cudarf/visibuf.inl`: visibility-buffer material pass logic.
- `src/rf/src/renderer/cudarf/raster_naive.inl`: fine raster path.
- `src/rf/src/renderer/draw_list_renderer.cpp`: draw-list submission and `withOpaqueVisibuf` selection.
- `src/sv_engine/src/views/view_3d.cpp`: view-level stage timing and render flow.
- `assets/sample_pack_4cam/config/overlays.json`: `renderer_config.use_visibuf` runtime toggle.
