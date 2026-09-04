#!/usr/bin/env bash
set -euo pipefail

[ "$#" -eq 1 ] || { echo "usage: $0 LAYER" >&2; exit 2; }
LAYER=$1
[[ "$LAYER" =~ ^[0-9]+$ ]] && [ "$LAYER" -ge 3 ] && [ "$LAYER" -le 44 ] || { echo "layer must be 3..44" >&2; exit 2; }

REPO=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53
CAMPAIGN=/media/brandonmusic/klcstore/bmxfp4-glm53
printf -v L3 '%03d' "$LAYER"
ROOT="$CAMPAIGN/codec-v2/output-aware/layer${LAYER}-full"
SOURCE="$CAMPAIGN/downloads/GLM-5.3-Flash-BF16"
INDEX="$SOURCE/model.safetensors.index.json"
CAPTURE="$CAMPAIGN/teacher/calibration/main-ep4-full"
ROLES="$CAMPAIGN/roles/roles-v3.json"
LOCK=/run/lock/klc/model-stack.lock
mkdir -p "$ROOT/dense" "$ROOT/codec" "$ROOT/receipts" "$ROOT/logs"
cd "$REPO"

exec 9>"$LOCK"
flock -w 900 9
backend_was_active=false
systemctl --user is-active --quiet klc-backend.service && backend_was_active=true
restore() {
  set +e
  [ "$backend_was_active" = true ] && systemctl --user start klc-backend.service
}
trap restore EXIT
[ "$backend_was_active" = true ] && systemctl --user stop klc-backend.service

pids=()
gpus=(1 2 3)
starts=(0 96 192)
ends=(96 192 288)
for slot in 0 1 2; do
  gpu=${gpus[$slot]}
  start=${starts[$slot]}
  end=${ends[$slot]}
  printf -v S3 '%03d' "$start"
  printf -v E3 '%03d' "$end"
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. python3 -m glm53_nvfp4.quantize_hessian_trellis_layer \
    --source "$SOURCE" --source-index "$INDEX" \
    --capture-root "$CAPTURE" --roles "$ROLES" --layer "$LAYER" \
    --expert-start "$start" --expert-end "$end" --samples 256 --device cuda:0 \
    --dense-output "$ROOT/dense/hessian-trellis-layer-$L3-experts-$S3-$E3.safetensors" \
    --codec-output "$ROOT/codec/hessian-trellis-layer-$L3-experts-$S3-$E3.safetensors" \
    --receipt "$ROOT/receipts/hessian-trellis-layer-$L3-experts-$S3-$E3.json" \
    >"$ROOT/logs/chunk-$S3-$E3.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
