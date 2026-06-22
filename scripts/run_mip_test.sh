#!/usr/bin/env bash
set -euo pipefail

script_path="${BASH_SOURCE[0]}"
while [ -L "$script_path" ]; do
  script_dir="$(cd "$(dirname "$script_path")" && pwd)"
  script_path="$(readlink "$script_path")"
  [[ "$script_path" != /* ]] && script_path="$script_dir/$script_path"
done

script_dir="$(cd "$(dirname "$script_path")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
asset_dir="$repo_root/assets/sample_pack_4cam"
test_dir="$repo_root/assets/cudarf_test"

width="${1:-1920}"
height="${2:-1080}"

cd "$asset_dir"

source "$repo_root/scripts/set_workspace.sh"

sv_app \
  --rig "$asset_dir/canonical-rig.json" \
  --width "$width" \
  --height "$height" \
  --test-scenario "$test_dir/mip-checker.json" \
  --frames \
    "$asset_dir/right.png" \
    "$asset_dir/left.png" \
    "$asset_dir/front.png" \
    "$asset_dir/rear.png"
