#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mapfile -d '' files < <(find . -type f -name compile_commands.json ! -path './compile_commands.json' -print0)

if [ "${#files[@]}" -eq 0 ]; then
    exit 0
fi

tmpfile=$(mktemp)
jq -s 'add' "${files[@]}" > "${tmpfile}"
mv "${tmpfile}" compile_commands.json
