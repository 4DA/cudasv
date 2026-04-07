#!/usr/bin/env bash
set -euo pipefail

script_path="${BASH_SOURCE[0]}"
while [ -L "$script_path" ]; do
  script_dir="$(cd "$(dirname "$script_path")" && pwd)"
  script_path="$(readlink "$script_path")"
  [[ "$script_path" != /* ]] && script_path="$script_dir/$script_path"
done

asset_dir="$(cd "$(dirname "$script_path")" && pwd)"
repo_root="$(cd "$asset_dir/../.." && pwd)"

width="${1:-1936}"
height="${2:-1220}"

cd "$asset_dir"

source "$repo_root/scripts/set_workspace.sh"

sv_app \
  --rig "$asset_dir/canonical-rig.json" \
  --width "$width" \
  --height "$height" \
  --frames \
    "$asset_dir/right.png" \
    "$asset_dir/left.png" \
    "$asset_dir/front.png" \
    "$asset_dir/rear.png"
