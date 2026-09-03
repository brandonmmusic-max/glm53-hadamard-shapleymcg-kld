#!/usr/bin/env bash
set -euo pipefail

ROTATION=${1:?usage: $0 ROTATION [ROTATION_FILE] [START_LAYER]}
ROTATION_FILE=${2:-}
START_LAYER=${3:-3}
REPO=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53
CAMPAIGN=/media/brandonmusic/klcstore/bmxfp4-glm53
LOG=$CAMPAIGN/logs-v3/mxfp6-v10-had16-all-campaign.log
mkdir -p "$(dirname "$LOG")"
for LAYER in $(seq "$START_LAYER" 44); do
  "$REPO/scripts/run_mxfp6_layer.sh" "$LAYER" "$ROTATION" "$ROTATION_FILE" >>"$LOG" 2>&1
done
