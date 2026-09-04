#!/usr/bin/env bash
set -euo pipefail

REPO=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53
CAMPAIGN=/media/brandonmusic/klcstore/bmxfp4-glm53
ROOT=$CAMPAIGN/codec-v2/output-aware/layer3-full
SOURCE=$CAMPAIGN/downloads/GLM-5.3-Flash-BF16
INDEX=$SOURCE/model.safetensors.index.json
CAPTURE=$CAMPAIGN/teacher/calibration/main-ep4-full
ROLES=$CAMPAIGN/roles/roles-v3.json
mkdir -p "$ROOT/dense" "$ROOT/codec" "$ROOT/receipts" "$ROOT/logs"

pids=()
starts=(0 72 144 216)
ends=(72 144 216 288)
for gpu in 0 1 2 3; do
  start=${starts[$gpu]}
  end=${ends[$gpu]}
  CUDA_VISIBLE_DEVICES=$gpu python3 -m glm53_nvfp4.quantize_hessian_trellis_layer \
    --source "$SOURCE" --source-index "$INDEX" \
    --capture-root "$CAPTURE" --roles "$ROLES" --layer 3 \
    --expert-start "$start" --expert-end "$end" --samples 256 --device cuda:0 \
    --dense-output "$ROOT/dense/hessian-trellis-layer-003-experts-$(printf '%03d' "$start")-$(printf '%03d' "$end").safetensors" \
    --codec-output "$ROOT/codec/hessian-trellis-layer-003-experts-$(printf '%03d' "$start")-$(printf '%03d' "$end").safetensors" \
    --receipt "$ROOT/receipts/hessian-trellis-layer-003-experts-$(printf '%03d' "$start")-$(printf '%03d' "$end").json" \
    >"$ROOT/logs/chunk-$(printf '%03d' "$start")-$(printf '%03d' "$end").log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
