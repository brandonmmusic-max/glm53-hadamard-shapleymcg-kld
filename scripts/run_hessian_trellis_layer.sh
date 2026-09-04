#!/usr/bin/env bash
set -euo pipefail

[ "$#" -eq 1 ] || { echo "usage: $0 LAYER" >&2; exit 2; }
LAYER=$1
[[ "$LAYER" =~ ^[0-9]+$ ]] && [ "$LAYER" -ge 3 ] && [ "$LAYER" -le 44 ] || { echo "layer must be 3..44" >&2; exit 2; }

REPO=${GLM53_ENCODER_REPO:-/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53}
CAMPAIGN=${GLM53_CAMPAIGN_ROOT:-/media/brandonmusic/klcstore/bmxfp4-glm53}
printf -v L3 '%03d' "$LAYER"
ROOT=${GLM53_P8_OUTPUT_ROOT:-$CAMPAIGN/codec-v2/native6-v2/layer${LAYER}-full}
SOURCE=${GLM53_BF16_SOURCE:-$CAMPAIGN/downloads/GLM-5.3-Flash-BF16}
INDEX="$SOURCE/model.safetensors.index.json"
CAPTURE=${GLM53_CAPTURE_ROOT:-$CAMPAIGN/teacher/calibration/main-ep4-full}
ROLES=${GLM53_FIT_ROLES:-$CAMPAIGN/roles/roles-v3.json}
DESIGN=${GLM53_NATIVE6_DESIGN:-$REPO/experiments/p8-kld-shapley-native6-v2.json}
PYTHON_BIN=${GLM53_ENCODER_PYTHON:-/home/brandonmusic/klc-env/bin/python}
WRITE_DENSE=${GLM53_P8_WRITE_DENSE:-0}
LOCK=/run/lock/klc/model-stack.lock
[[ "$WRITE_DENSE" = 0 || "$WRITE_DENSE" = 1 ]] || { echo "GLM53_P8_WRITE_DENSE must be 0 or 1" >&2; exit 2; }
[ -x "$PYTHON_BIN" ] || { echo "missing encoder Python: $PYTHON_BIN" >&2; exit 2; }
[ -f "$DESIGN" ] || { echo "missing native6 v2 design: $DESIGN" >&2; exit 2; }
mkdir -p "$ROOT/codec" "$ROOT/receipts" "$ROOT/logs"
[ "$WRITE_DENSE" = 0 ] || mkdir -p "$ROOT/dense"
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
gpus=(0 1 2 3)
starts=(0 72 144 216)
ends=(72 144 216 288)
for slot in 0 1 2 3; do
  gpu=${gpus[$slot]}
  start=${starts[$slot]}
  end=${ends[$slot]}
  printf -v S3 '%03d' "$start"
  printf -v E3 '%03d' "$end"
  dense_args=()
  if [ "$WRITE_DENSE" = 1 ]; then
    dense_args=(--dense-output "$ROOT/dense/hessian-trellis-layer-$L3-experts-$S3-$E3.safetensors")
  fi
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. "$PYTHON_BIN" -m glm53_nvfp4.quantize_hessian_trellis_layer \
    --source "$SOURCE" --source-index "$INDEX" \
    --capture-root "$CAPTURE" --roles "$ROLES" --layer "$LAYER" \
    --design "$DESIGN" \
    --expert-start "$start" --expert-end "$end" --samples 256 --device cuda:0 \
    "${dense_args[@]}" \
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
