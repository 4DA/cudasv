# Third-Party Notices

This repository is distributed under the GNU General Public License v3.0 as
described in `LICENSE`, except where third-party
code retains its own upstream copyright and license notices.

## NVIDIA / cudaraster-Derived Rasterization Code

Parts of the CUDA rasterization path are derived from or closely based on the
`cudaraster` work described in:

- Samuli Laine and Tero Karras, "High-Performance Software Rasterization on GPUs", HPG 2011
- cudaraster archive: <https://code.google.com/archive/p/cudaraster/>

Files in this repository that retain or correspond to that upstream lineage
include:

- `src/rf/src/renderer/cudarf/triangle.inl`
- `src/rf/src/renderer/cudarf/tiler_bin.inl`
- `src/rf/src/renderer/cudarf/tiler_coarse.inl`

The files that retain upstream NVIDIA / cudaraster notices should keep those
notices intact. Their upstream notices describe redistribution under a BSD-style
license.

## NVIDIA Helper Math Code

This repository also includes CUDA helper/vector utility code with an upstream
NVIDIA notice in:

- `src/rf/src/renderer/cudarf/helpers_cudavec.inl`

That file retains its upstream NVIDIA notice and BSD-style redistribution
terms in the file header.

## Third-Party Assets

### Citrus Orchard Road (HDRI)

- Source: Poly Haven
- Source URL: <https://polyhaven.com/a/citrus_orchard_road>
- Authors: Dimitrios Savva (photography), Jarod Guest (processing)
- License: CC0 1.0
- Local file: `assets/citrus_orchard_ibl`
- Modifications: none

### Vehicle Model GLBs

- Source: "[FREE] Polestar 1"
- Source URL: <https://skfb.ly/oGUBP>
- Author: Martin Trafas
- License: Creative Commons Attribution 4.0 International
  (<http://creativecommons.org/licenses/by/4.0/>)
- Local files: vehicle model GLBs under `assets/glb/`
- Modifications: the local vehicle GLBs are derivative works maintained for this
  project and may include geometry, material, texture, scale, orientation, or
  packaging changes from the original source asset.

## Project And Sample Assets

The following assets are included with the repository as project-owned or
maintainer-authorized sample assets, unless another notice in this file states
otherwise.

### Viewpoint Controls GLBs

- Local files: controls GLBs under `assets/glb/`
- Provenance: project-owned assets created for cudaSV.
- License: distributed with the repository under the top-level GPLv3 license.

### Sample Camera Images

- Local files:
  - `assets/sample_pack_4cam/right.png`
  - `assets/sample_pack_4cam/left.png`
  - `assets/sample_pack_4cam/front.png`
  - `assets/sample_pack_4cam/rear.png`
- Provenance: The sample images are included with permission for demonstration
  and testing use in this repository. They contain ordinary road/environment
  content and no vendor-specific, customer-identifying, or confidential data.
- License: Redistributed with permission for demonstration and testing use in
  this repository.

### Vehicle Shadow / Underlay

- Local file: `assets/sample_pack_4cam/vehicle_shadow.png`
- Provenance: project-owned sample asset created for cudaSV.
- License: distributed with the repository under the top-level GPLv3 license.

### BRDF LUTs

- Local files:
  - `assets/citrus_orchard_ibl/brdfLUT.png`
  - `assets/citrus_orchard_ibl/brdfLUT_16.png`
- Provenance: project-owned generated assets.
- Generator: <https://github.com/4DA/brdfgen>
- License: distributed with the repository under the top-level GPLv3 license.

### Test GLBs

- Local files: test GLBs under `assets/cudarf_test/`
- Provenance: project-owned test assets created for cudaSV.
- License: distributed with the repository under the top-level GPLv3 license.

## Practical Interpretation

For repository users:

- treat the repository as GPLv3-licensed at the top level
- preserve any third-party notices that remain in individual source files
- preserve this notice file when redistributing the repository or substantial
  portions of it
