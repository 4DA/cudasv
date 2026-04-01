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

## Practical Interpretation

For repository users:

- treat the repository as GPLv3-licensed at the top level
- preserve any third-party notices that remain in individual source files
- preserve this notice file when redistributing the repository or substantial
  portions of it
