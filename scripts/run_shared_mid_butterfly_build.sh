#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 GRID_ARM" >&2
  exit 2
fi
ARM=$1
case "$ARM" in
  m0250|m0125|m00625|m003125|zero|p003125|p00625|p0125|p0250) ;;
  *) echo "unknown sealed V8 grid arm: $ARM" >&2; exit 2 ;;
esac

REPO=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-rotation-v6-blocklocal-kld
PLAN=${GLM53_V8_PLAN:?set GLM53_V8_PLAN to the immutable V8 plan path}
ROOT=${GLM53_V8_ROOT:-/media/brandonmusic/nvme1n1p3/glm53-rotation-v8/shared-mid-butterfly-layer3}
SOURCE=/media/brandonmusic/klcstore/bmxfp4-glm53/downloads/GLM-5.3-Flash-BF16
SOURCE_INDEX=$SOURCE/model.safetensors.index.json
CAPTURE=/media/brandonmusic/klcstore/bmxfp4-glm53/teacher/calibration/main-ep4-full
ROLES=/media/brandonmusic/klcstore/bmxfp4-glm53/roles/roles-v3.json
ROTATION=$ROOT/rotations/$ARM.safetensors
ARM_ROOT=$ROOT/arms/$ARM
CHUNKS=$ARM_ROOT/chunks
RECEIPTS=$ARM_ROOT/receipts
LOGS=$ARM_ROOT/logs
CANDIDATE=$ARM_ROOT/candidate
VALIDATION=$ARM_ROOT/layer-validation.json
BUILD=$ARM_ROOT/FULL_BUILD.json
for path in "$CANDIDATE" "$VALIDATION" "$BUILD"; do
  [ ! -e "$path" ] || { echo "refusing to reuse V8 output: $path" >&2; exit 1; }
done
mkdir -p "$CHUNKS" "$RECEIPTS" "$LOGS"

exec 9>/run/lock/klc/model-stack.lock
flock -w 900 9
cd "$REPO"
pids=()
for gpu in 0 1 2 3; do
  start=$((gpu * 72))
  end=$((start + 72))
  printf -v range '%03d-%03d' "$start" "$end"
  CUDA_VISIBLE_DEVICES=$gpu /home/brandonmusic/klc-env/bin/python \
    -m glm53_nvfp4.quantize_shared_mid_butterfly_layer \
    --plan "$PLAN" --source "$SOURCE" --source-index "$SOURCE_INDEX" \
    --capture-root "$CAPTURE" --roles "$ROLES" --rotation-file "$ROTATION" \
    --expert-start "$start" --expert-end "$end" --device cuda:0 \
    --output "$CHUNKS/$ARM-$range.safetensors" \
    --receipt "$RECEIPTS/$ARM-$range.json" >"$LOGS/gpu-$gpu.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
[ "$failed" -eq 0 ] || { echo "one or more V8 quantizers failed" >&2; exit 1; }

receipt_args=()
chunk_args=()
for range in 000-072 072-144 144-216 216-288; do
  receipt_args+=(--receipt "$RECEIPTS/$ARM-$range.json")
  chunk_args+=(--chunk "$CHUNKS/$ARM-$range.safetensors")
done
python3 -m glm53_nvfp4.validate_layer --layer 3 --projections down-only \
  "${receipt_args[@]}" --output "$VALIDATION" | tee "$LOGS/validation.log"
python3 -m glm53_nvfp4.candidate \
  --carrier /home/brandonmusic/models/GLM-5.3-Flash-NVFP4 \
  --output "$CANDIDATE" "${chunk_args[@]}" | tee "$LOGS/candidate.log"
python3 -m glm53_nvfp4.finalize_shared_mid_butterfly_build \
  --plan "$PLAN" --rotation "$ROTATION" "${receipt_args[@]}" \
  --validation "$VALIDATION" --candidate "$CANDIDATE" --output "$BUILD" \
  | tee "$LOGS/finalize.log"
