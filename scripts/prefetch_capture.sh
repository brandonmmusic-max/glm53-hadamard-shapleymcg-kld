#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 LAYER" >&2
  exit 2
fi
LAYER=$1
[[ "$LAYER" =~ ^[0-9]+$ ]] && [ "$LAYER" -ge 3 ] && [ "$LAYER" -le 44 ] || { echo "layer must be 3..44" >&2; exit 2; }

REPO=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53
CAMPAIGN=/media/brandonmusic/klcstore/bmxfp4-glm53
CAPTURE=$CAMPAIGN/teacher/calibration/main-ep4-full
LOCKS=$CAMPAIGN/locks
printf -v L3 '%03d' "$LAYER"
LOGS=$CAMPAIGN/logs/layer-$L3
mkdir -p "$LOCKS" "$LOGS"

CAPTURE_FILES=(
  "calibration/main-ep4-full/layers/layer-$L3/hidden.bf16.bin"
  "calibration/main-ep4-full/layers/layer-$L3/topk_ids.u16le.bin"
  "calibration/main-ep4-full/layers/layer-$L3/topk_weights.f32le.bin"
)
exec 8>"$LOCKS/capture-layer-$L3.lock"
flock -w 7200 8
HF_XET_HIGH_PERFORMANCE=1 /home/brandonmusic/.local/bin/hf download brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits "${CAPTURE_FILES[@]}" \
  --type dataset --revision 95f4fdd94bf29989db2e0d1054e4931f55edb6aa --local-dir "$CAMPAIGN/teacher" --max-workers 3 --format agent \
  >"$LOGS/capture-prefetch.log" 2>&1
cd "$REPO"
/usr/bin/python3 -m glm53_nvfp4.verify_capture --capture-root "$CAPTURE" --layer "$LAYER" \
  --output "$CAMPAIGN/evidence/capture-layer-$L3-prefetch.json" | tee "$LOGS/capture-prefetch-integrity.log"
