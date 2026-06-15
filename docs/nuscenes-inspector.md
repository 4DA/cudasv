# nuScenes Inspector

`sv_app` includes a source-side nuScenes inspector that bypasses the
fixed 4-camera surround-view runtime bridge and displays one decoded
nuScenes sample directly.

## Purpose

- Validate that `NuScenesSource` opens and decodes real 6-camera samples.
- Inspect raw camera images independently from the surround-view renderer.
- Check camera ordering, image orientation, timestamps, and sample stepping.

## Display Modes

- `Mosaic`
  - shows the 6 cameras in a `3x2` layout
  - top row: `front_left`, `front`, `front_right`
  - bottom row: `rear_left`, `rear`, `rear_right`
- `Focus`
  - shows one selected camera fullscreen
  - preserves image aspect ratio

Both mosaic and focus modes preserve source image aspect ratio with
letterboxing instead of stretching.

## Window Title

The inspector window title shows:

- sample index and total sample count
- sample token
- playback state: `play` or `pause`
- display mode: `mosaic` or `focus:<role>`
- in focus mode:
  - selected camera image size
  - selected camera timestamp

## Controls

- `Left` / `Right`
  - step by 1 sample
- `Shift+Left` / `Shift+Right`
  - step by 5 samples
- `Home`
  - jump to first sample in the resolved scene
- `End`
  - jump to last sample in the resolved scene
- `P`
  - toggle autoplay
- `1` .. `6`
  - select a camera directly
- `[` / `]`
  - cycle the selected camera
- `Space`
  - toggle between mosaic and focus view
- `Esc`
  - close the inspector

## Notes

- The inspector uses nuScenes key-frame camera images only.
- It is a source/debugging tool, not a stitched surround-view path.
- Sample changes are logged to the terminal, but not every frame.
