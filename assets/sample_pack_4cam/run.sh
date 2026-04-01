#!/bin/bash

source "$(dirname "$0")/../../scripts/set_workspace.sh" >/dev/null
cd "$(dirname "$0")"

sv_app --frames frames/right.png frames/left.png frames/front.png frames/rear.png \
        --rig canonical-rig.json \
	--width 1920 \
	--height 1080
