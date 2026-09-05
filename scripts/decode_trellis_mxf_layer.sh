#!/usr/bin/env bash
set -euo pipefail

[ "$#" -eq 1 ] || { echo "usage: $0 LAYER" >&2; exit 2; }
LAYER=$1
[[ "$LAYER" =~ ^[0-9]+$ ]] && [ "$LAYER" -ge 3 ] && [ "$LAYER" -le 44 ] || {
  echo "layer must be 3..44" >&2
  exit 2
}

REPO=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53
CAMPAIGN=/media/brandonmusic/klcstore/bmxfp4-glm53
ROOT=$CAMPAIGN/codec-v2/output-aware/layer${LAYER}-full
LOCK=/run/lock/klc/model-stack.lock
printf -v L3 '%03d' "$LAYER"
mkdir -p "$ROOT/dense" "$ROOT/decode-receipts" "$ROOT/logs"
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
starts=(0 72 144 216)
ends=(72 144 216 288)
for gpu in 0 1 2 3; do
  start=${starts[$gpu]}
  end=${ends[$gpu]}
  printf -v S3 '%03d' "$start"
  printf -v E3 '%03d' "$end"
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. python3 -m glm53_nvfp4.decode_trellis_mxf_bf16 \
    --codec "$ROOT/codec/hessian-trellis-layer-$L3-experts-$S3-$E3.safetensors" \
    --output "$ROOT/dense/hessian-trellis-layer-$L3-experts-$S3-$E3.safetensors" \
    --receipt "$ROOT/decode-receipts/hessian-trellis-layer-$L3-experts-$S3-$E3.json" \
    --device cuda:0 >"$ROOT/logs/decode-$S3-$E3.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
