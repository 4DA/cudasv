# Architecture

This note describes the runtime architecture.

It focuses on:

1. the important files and ownership boundaries
2. the data flow from calibration and source frames into rendering
3. user-input flow
4. draw-packet creation and submission
5. when and how the surround-view stitching kernel runs

## Runtime Shape

- `src/sv_app`
  - app/bootstrap, command-line parsing, source selection, runtime loop, GLFW host
- `src/sv_engine`
  - engine lifecycle, world ownership, vehicle state, view orchestration, surround-view composition
- `src/rf`
  - scene graph, glTF loading, CUDA rasterizer, surround-view projector, materials, IBL

## Important Files

### App / Bootstrap

- `src/sv_app/src/main.cpp`
  - process entrypoint
  - parses CLI
  - creates the selected `FrameSource`
  - loads canonical rig into the fixed 4-camera runtime calibration bridge
  - initializes `Engine`
  - creates `DemoVehicleSignalProvider`
  - enters the main application loop

- `src/sv_app/src/app/app_runtime.cpp`
  - main frame loop
  - fetches vehicle signals
  - fetches source-side `FramePacket` from the active source
  - adapts it into `RuntimeFramePacket4Cam` through the fixed 4-camera render bridge
  - calls `engine->pre_process(...)`
  - calls `engine->process(...)`
  - swaps output windows and releases the source frame packet back to the source

- `src/sv_app/src/app/glfw_host.cpp`
  - GLFW window creation
  - input callback registration
  - converts mouse input into engine `InputEvent`s in normalized device coordinates

### Source / Ingestion

- `src/sv_engine/include/engine/frame_source.hpp`
  - `FrameSource` interface
  - `SourceInfo`
  - `SourceKind`
  - source capability contract for sample identity, synchronization, and timestamps

- `src/sv_engine/include/engine/frame_packet.hpp`
  - shared `FramePacketMetadata`
  - source-side `FramePacket`
  - renderer-side `RuntimeFramePacket4Cam`
  - the source packet carries source sample metadata and source-facing frame storage
  - the runtime packet carries fixed 4-camera render input

- `src/sv_app/src/sources/source_factory.cpp`
  - instantiates the active `FrameSource`

- `src/sv_app/src/sources/file_sequence_source.cpp`
  - public PNG-based source

- `src/sv_app/src/sources/render_bridge_4cam.cpp`
  - resolves the `right/left/front/rear` bridge contract
  - adapts source-side `FramePacket` into runtime `RuntimeFramePacket4Cam`
  - remaps source packet ordering into the renderer's expected slot order

- `src/sv_app/src/compat/render_bridge_4cam_contract.cpp`
  - shared role-to-slot contract for the fixed 4-camera bridge

- `src/sv_app/src/config/canonical_rig.cpp`
  - canonical rig JSON parsing

- `src/sv_app/src/config/runtime_calibration_bridge_4cam.cpp`
  - converts canonical `CameraRig` into the fixed-slot runtime calibration structure

### Engine / World / Views

- `src/sv_engine/include/engine/engine.hpp`
  - public engine API
  - frame-processing entrypoints consume `RuntimeFramePacket4Cam`

- `src/sv_engine/src/engine_init.cpp`
  - initializes CUDA streams and raster contexts
  - loads the vehicle GLTF
  - optionally loads a test-scenario GLTF under the vehicle root
  - initializes `World`
  - initializes `View3D`
  - creates the `DrawListRenderer`

- `src/sv_engine/src/engine_frame.cpp`
  - per-frame vehicle-state update
  - per-frame camera texture upload into the surround-view projector
  - per-frame `View3D` render dispatch

- `src/sv_engine/src/engine_view_control.cpp`
  - routes input events and viewpoint commands from engine API into `View3D`

- `src/sv_engine/src/world.cpp`
  - owns `rf::Scene`
  - owns the vehicle controller and underlay
  - owns surround-view projector runtime state and uploaded camera textures

- `src/sv_engine/src/views/view_3d.cpp`
  - top-level view composition for the main 3D surround-view output
  - updates virtual camera/navigation state
  - runs surround-view composition
  - builds scene-pass work
  - launches scene rendering
  - runs post-processing/output composition

- `src/sv_engine/src/views/scene_pass_builder.cpp`
  - builds opaque scene work for the vehicle and opaque control geometry
  - builds translucent scene work for vehicle translucent materials and underlay/shadow
  - builds separate UI-overlay work for translucent viewpoint/control geometry

- `src/sv_engine/src/views/surround_view_composer.cpp`
  - converts runtime calibration into projector parameters
  - maps the surround-view blend profile from config into the projector
  - maps the surround-view debug mode from config into the projector
  - calls the surround-view CUDA projector

- `src/sv_engine/src/views/view_post_process_pipeline.cpp`
  - handles TAA history setup for the scene framebuffer
  - final three-layer framebuffer composition into the output buffer

- `src/sv_engine/src/views/view_interaction_router.cpp`
  - maps engine `InputEvent`s into viewpoint selection, controller pan, scroll, and ray-pick behavior

### Rendering / Scene / CUDA

- `src/rf/include/rf/renderer/gltf_loader.hpp`
  - public GLTF loader API

- `src/rf/src/renderer/gltf_loader.cpp`
  - façade GLTF loader
  - also contains `add_naive_mesh(...)`

- `src/rf/src/renderer/gltf/scene_loader.cpp`
  - scene graph creation
  - primitive creation
  - draw-packet allocation and upload for GLTF meshes

- `src/rf/src/renderer/cudarf/cudarf.cpp`
  - CUDA raster context lifecycle
  - draw-packet allocation
  - draw-packet vertex/index upload
  - frame begin and raster pipeline entrypoints

- `src/rf/src/renderer/draw_list_renderer.cpp`
  - turns scene work into raster submissions
  - submits draw-packet ids, uniforms, and material ids to `cudarf::pipe::run_pipe(...)`

- `src/rf/src/renderer/surround_view/projector.cu`
  - surround-view projection and blending kernel path
  - uploads camera frames to CUDA textures
  - runs the surround-view stitching pass into the mesh framebuffer

## Data Flow

### 1. Calibration And Source Setup

1. `main.cpp` parses command-line options and loads app config.
2. `SourceFactory` creates the selected `FrameSource`.
3. `FrameSource::open()` loads the source-specific rig and prepares frame access.
4. `main.cpp` calls `load_camera_rig_into_runtime_calibration(...)`.
5. The canonical rig is parsed in `canonical_rig.cpp`.
6. The fixed-slot runtime calibration structure is filled in `runtime_calibration_bridge_4cam.cpp`.

Limitations:

- the runtime calibration bridge requires exactly 4 cameras
- the supported render roles are `right`, `left`, `front`, `rear`
- the shipped source path is the owned 4-camera PNG sample flow

### 2. Per-Frame Source Flow

1. `run_application_loop(...)` asks the active `VehicleSignalProvider` for the next signal sample.
2. It updates engine vehicle state with `engine->update_vehicle_state(...)`.
3. It asks the active `FrameSource` for the next source-side `FramePacket`.
4. It adapts that packet into the renderer runtime shape with:
   - `adapt_frame_packet_for_runtime_render_bridge_4cam(...)`
5. It calls `engine->pre_process(runtime_frame_packet)`.

Packet contract:

- `FramePacket` is the source-facing sample packet type.
- `RuntimeFramePacket4Cam` is the renderer-facing packet type.
- The engine consumes renderer packets rather than source packets directly.
- The 4-camera bridge adapts source packet shape to renderer packet shape.

### 3. Engine Pre-Process Flow

Inside `Engine::pre_process(...)`:

1. it validates that all render-bridge cameras are present
2. it advances vehicle state through `sv_vehicle_state_update(...)`
3. it updates the scene vehicle controller
4. if the frame sequence changed, it uploads the source RGB frames into surround-view CUDA textures with:
   - `world->frame_projector().load_rgb(...)`
5. it stores the latest frame sequence in the projector runtime

This is the point where source image buffers first become GPU camera textures for the surround-view path.

### 4. Engine Process Flow

Inside `Engine::process(...)`:

1. it selects the per-output raster context and output surface
2. it asks `View3D` to update the active virtual camera
3. it builds frame-wide raster state from the updated camera, scene lights,
   IBL resources, exposure, and TAA history
4. it calls `cudarf::pipe::begin_frame(...)`
5. it dispatches active views for the output
6. for `SV_VIEW_3D`, it calls `View3D::compose(...)`
7. after view rendering, it synchronizes the CUDA stream and presents the output

`begin_frame(...)` is the boundary between engine frame setup and draw
submission. It uploads the frame context used by every raster submission in
that output frame, including:

- the clean or TAA-jittered common camera uniforms
- previous-frame common uniforms for TAA
- velocity/TAA state
- camera position, exposure, lights, spherical harmonics, BRDF LUT, and
  specular IBL

### 5. View3D Flow

Inside `View3D::compose(...)`:

1. call `SurroundViewComposer::compose(...)`
2. build scene work with `ScenePassBuilder::build(...)`
3. prepare post-process state with `ViewPostProcessPipeline::begin_frame(...)`
4. get the internal scene raster framebuffer from `cudarf::pipe::get_internal_fb(...)`
5. get the transparent UI overlay framebuffer from `cudarf::pipe::get_ui_fb(...)`
6. render scene and UI passes with `ScenePassBuilder::render(...)`
7. combine the surround-view output, rasterized scene, and UI overlay with `ViewPostProcessPipeline::run(...)`

Camera controller and viewpoint animation updates happen earlier through
`View3D::update_camera(...)`, before the raster frame context is uploaded.
That keeps the camera seen by surround-view composition, scene pass building,
TAA jitter, and PBR shading consistent for the whole frame.

## How Frames Reach The Stitching Kernel

The surround-view kernel is not called directly from `sv_app`. The path is:

1. source produces `FramePacket`
2. `Engine::pre_process(...)` uploads the 4 source camera images into projector-owned CUDA textures
3. `Engine::process(...)` calls `View3D::compose(...)`
4. `View3D::compose(...)` calls `SurroundViewComposer::compose(...)`
5. `SurroundViewComposer::compose(...)` builds:
   - `ViewParams` from the virtual camera
   - `RigParams` from runtime calibration
6. `SurroundViewComposer::compose(...)` calls:
   - `rf::surround_view::project_async(...)`
7. `project_async(...)` in `projector.cu` launches the surround-view projection/stitching CUDA kernel path into `meshGPUOutput`

So:

- frame upload happens during `pre_process`
- stitching happens during `View3D::compose`
- the stitched output lands in the mesh framebuffer before final composition

## Draw Packets

### Who Builds Them

Draw packets are created when geometry is loaded into the renderer.

Main paths:

- GLTF path:
  - `renderer/gltf/scene_loader.cpp`
- naive mesh path:
  - `renderer/gltf_loader.cpp` via `add_naive_mesh(...)`

The loader flow is:

1. allocate a draw packet with `cudarf::pipe::alloc_draw_packet(...)`
2. upload vertex/index buffers with `cudarf::pipe::set_draw_packet_buffers(...)`
3. store the resulting draw-packet id on the `PrimitiveComponent`

This is done during scene/model loading, not every frame.

### How They Are Submitted

At render time:

1. `ScenePassBuilder::build(...)` traverses the scene through `DrawListRenderer::add_work(...)`
2. it builds opaque, translucent, and UI-overlay work lists
3. `ScenePassBuilder::render(...)` calls `DrawListRenderer::render(...)` for scene and UI targets
4. `DrawListRenderer::render(...)` submits:
   - draw-packet ids
   - uniforms
   - material ids
   - pass flags such as blending and face culling
5. submission happens through `cudarf::pipe::run_pipe(...)`

That is the point where prebuilt geometry draw packets are actually consumed by the CUDA raster pipeline.

Common frame state is built once per output frame by `cudarf::pipe::begin_frame(...)`, then referenced by
the draw submissions through the persistent pipe frame context.

## CUDA Raster Pipe Contexts

The CUDA raster pipe splits its GPU-side state by lifetime:

- `PipeStaticContext`
  - persistent render-target and tiling constants
  - window size, padded rasterizer size, clock rate
  - uploaded when the rasterizer context is initialized or recreated
- `PipeFrameContext`
  - frame-wide camera, TAA, lighting, and IBL state
  - uploaded once by `cudarf::pipe::begin_frame(...)`
- `PipeSubmissionContext`
  - per-draw-batch state
  - draw packets, material table, draw order, offsets, and pass flags
  - uploaded by `cudarf::pipe::run_pipe(...)`
- `PipeScratchContext`
  - persistent scratch allocations with submission-local contents
  - vertex output, triangles, bin state, tile queues, and debug buffers

`PipeParams` is a small device parameter block that points at the static,
frame, and submission contexts and carries the scratch context. This preserves
the kernel entrypoint shape while making state lifetime explicit.

## Scene And View Composition

`View3D` output is a combination of three render products:

1. surround-view projection result
   - produced first by `SurroundViewComposer`
   - written into `meshGPUOutput`
2. 3D scene raster result
   - produced by `DrawListRenderer` through `run_pipe(...)`
   - written into the internal raster framebuffer
   - optionally resolved through TAA into the scene output framebuffer
3. UI overlay raster result
   - produced by `DrawListRenderer` from `ScenePassBuilder::WorkSet::ui`
   - written into the transparent UI framebuffer
   - used for translucent viewpoint/control overlays so UI alpha is independent
     from scene coverage alpha

`ViewPostProcessPipeline::run(...)` then combines stitched lower, scene upper,
and UI overlay into the final output image. The scene alpha controls how much
the rasterized vehicle/shadow layer replaces the stitched image. The UI overlay
is composited as a separate premultiplied-alpha layer on top.

## User Input Flow

### GLFW To Engine

`GLFWHost` registers callbacks for:

- key input
- mouse move
- mouse button
- scroll

Important mappings:

- `ESC`
  - closes the application
- left mouse drag
  - converted into `PAN`
- right mouse click
  - converted into `TAP`
- scroll wheel
  - converted into `PINCH_ZOOM`

Mouse coordinates are converted from window space into normalized device coordinates in `glfw_host.cpp`.

### Engine To View

1. `GLFWHost` constructs `engine::InputEvent`
2. it calls `engine->input_event(...)`
3. `Engine::input_event(...)` forwards the event to the active view
4. `View3D::handle_event(...)` delegates to `ViewInteractionRouter`

### ViewInteractionRouter Behavior

- `PAN`
  - forwarded to the active camera controller
- `PINCH_ZOOM`
  - forwarded to the top-view controller or orbital controller depending on navigation mode
- `TAP`
  - converted into a world-space ray from the virtual camera
  - ray-intersects the scene
  - if a viewpoint-control primitive is hit, selects the corresponding viewpoint
- `DOUBLE_TAP`
  - re-applies the viewpoint selection

There is also extra app-side drag logic in `GLFWHost::cursor_position_callback(...)` for some viewpoint-specific behavior before the normal engine event path is used.

## Practical Summary

The runtime is structured as:

1. `sv_app` reads config, rig, and source frames
2. `sv_app` maps source camera ordering into the fixed 4-camera bridge
3. `sv_engine` uploads the source frames into projector textures
4. `sv_engine` drives `View3D`
5. `SurroundViewComposer` launches the surround-view CUDA projector
6. `DrawListRenderer` submits prebuilt draw packets into the CUDA raster pipe
7. `ScenePassBuilder` renders scene work into the scene framebuffer and UI work into the UI framebuffer
8. `ViewPostProcessPipeline` combines surround-view, scene, and UI overlay results into the final output

The renderer and runtime calibration path assume the fixed `right/left/front/rear` camera roles.

For NuScenes, the codebase provides a metadata/image source and a dedicated inspector path. The surround-view renderer uses the fixed 4-camera render bridge:

1. `NuScenesSource` can resolve dataset metadata, camera samples, and rig data.
2. `main.cpp` routes NuScenes sources to the NuScenes inspector loop instead of
   the surround-view renderer.
3. `prepare_source_for_runtime_bridge_4cam(...)` does not route NuScenes samples into the fixed 4-camera renderer path.

The renderer-facing path uses the canonical-rig plus fixed `right/left/front/rear` bridge.

Surround-view blending windows flow through config into `SurroundViewComposer`; the renderer uses the fixed 4-camera composition model.

The blend profile also carries per-side anchor offsets, so the overlap centers can move independently from the default `vehicle_length / 2` and `vehicle_width / 2` assumptions.

Surround-view visual debug modes include:

- normal output
- camera-role coloring
- coverage mask
- per-camera reprojection grid

These modes are configured through overlays config and applied inside the surround-view projector kernel.
