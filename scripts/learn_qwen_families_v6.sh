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
SOURCE=$CAMPAIGN/downloads/GLM-5.3-Flash-BF16
SOURCE_INDEX=$CAMPAIGN/downloads/source-metadata/model.safetensors.index.json
CAPTURE=$CAMPAIGN/teacher/calibration/main-ep4-full
ROLES=$CAMPAIGN/roles/roles-v3.json
printf -v L3 '%03d' "$LAYER"
ROOT=${GLM53_ROTATION_FIT_ROOT:-$CAMPAIGN/exact-v7/layer-$L3/rotation-fit-shared-route256}
SHARING=${GLM53_ROTATION_SHARING:-shared}
MAX_SAMPLES=${GLM53_ROTATION_MAX_SAMPLES:-256}
BATCH=${GLM53_ROTATION_BATCH:-16}
ACCUMULATION=${GLM53_ROTATION_ACCUMULATION:-1}
mkdir -p "$ROOT"

exec 9>/run/lock/klc/model-stack.lock
flock -w 900 9
timer_was_active=false
backend_was_active=false
systemctl is-active --quiet klc-model-stack.timer && timer_was_active=true
systemctl --user is-active --quiet klc-backend.service && backend_was_active=true
restore() {
  set +e
  [ "$backend_was_active" = false ] || systemctl --user start klc-backend.service
  [ "$timer_was_active" = false ] || sudo -n systemctl start klc-model-stack.timer
}
trap restore EXIT
[ "$timer_was_active" = false ] || sudo -n systemctl stop klc-model-stack.timer
[ "$backend_was_active" = false ] || systemctl --user stop klc-backend.service

families=(in in mid mid)
inits=(identity had16 identity had16)
pids=()
for gpu in 0 1 2 3; do
  family=${families[$gpu]}
  init=${inits[$gpu]}
  stem=$family-$init
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$REPO" /home/brandonmusic/klc-env/bin/python \
    -m glm53_nvfp4.learn_qwen_rotation_layer \
    --source "$SOURCE" --source-index "$SOURCE_INDEX" \
    --capture-root "$CAPTURE" --roles "$ROLES" --layer "$LAYER" \
    --objective per-projection --family "$family" --init "$init" --device cuda:0 \
    --sharing "$SHARING" --experts 288 --max-samples "$MAX_SAMPLES" \
    --steps 100 --batch "$BATCH" --accumulation "$ACCUMULATION" --lr 5e-3 --search-grid 12 \
    --output "$ROOT/$stem.safetensors" --receipt "$ROOT/$stem.json" \
    >"$ROOT/$stem.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[ "$failed" -eq 0 ] || { echo "one or more family fits failed; see $ROOT" >&2; exit 1; }

select_args=()
for family in in mid; do
  for init in identity had16; do
    select_args+=(--candidate "$ROOT/$family-$init.safetensors")
    select_args+=(--receipt "$ROOT/$family-$init.json")
  done
done
PYTHONPATH="$REPO" /usr/bin/python3 -m glm53_nvfp4.select_qwen_rotation_families \
  --layer "$LAYER" "${select_args[@]}" \
  --output "$ROOT/selected.safetensors" \
  --selection-receipt "$ROOT/selection.json"

echo "independent Qwen rotation-family fits for layer $LAYER complete"
