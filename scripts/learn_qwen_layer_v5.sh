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
ROOT=$CAMPAIGN/exact-v5/layer-$L3/rotation-fit
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

objectives=(per-projection per-projection full-expert full-expert)
inits=(identity had16 identity had16)
pids=()
for gpu in 0 1 2 3; do
  objective=${objectives[$gpu]}
  init=${inits[$gpu]}
  stem=$objective-$init
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$REPO" /home/brandonmusic/klc-env/bin/python \
    -m glm53_nvfp4.learn_qwen_rotation_layer \
    --source "$SOURCE" --source-index "$SOURCE_INDEX" \
    --capture-root "$CAPTURE" --roles "$ROLES" --layer "$LAYER" \
    --objective "$objective" --init "$init" --device cuda:0 \
    --experts 288 --max-samples 256 --steps 100 --batch 16 --lr 5e-3 --search-grid 12 \
    --output "$ROOT/$stem.safetensors" --receipt "$ROOT/$stem.json" \
    >"$ROOT/$stem.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[ "$failed" -eq 0 ] || { echo "one or more rotation fits failed; see $ROOT" >&2; exit 1; }

for objective in per-projection full-expert; do
  PYTHONPATH="$REPO" /usr/bin/python3 -m glm53_nvfp4.select_qwen_rotation_layer \
    --layer "$LAYER" --include-fixed \
    --candidate "$ROOT/$objective-identity.safetensors" \
    --receipt "$ROOT/$objective-identity.json" \
    --candidate "$ROOT/$objective-had16.safetensors" \
    --receipt "$ROOT/$objective-had16.json" \
    --output "$ROOT/$objective-selected.safetensors" \
    --selection-receipt "$ROOT/$objective-selection.json"
done

echo "Qwen rotation fits for layer $LAYER complete"
