#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "usage: $0 LAYER had16|learned [ROTATION_FILE]" >&2
  exit 2
fi
LAYER=$1
ROTATION=$2
ROTATION_FILE=${3:-}
[[ "$LAYER" =~ ^[0-9]+$ ]] && [ "$LAYER" -ge 3 ] && [ "$LAYER" -le 44 ] || { echo "layer must be 3..44" >&2; exit 2; }
[ "$ROTATION" = had16 ] || [ "$ROTATION" = learned ] || { echo "rotation must be had16 or learned" >&2; exit 2; }
if [ "$ROTATION" = learned ] && [ -z "$ROTATION_FILE" ]; then
  echo "learned requires ROTATION_FILE" >&2
  exit 2
fi

REPO=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53
CAMPAIGN=/media/brandonmusic/klcstore/bmxfp4-glm53
SOURCE=$CAMPAIGN/downloads/GLM-5.3-Flash-BF16
SOURCE_INDEX=$CAMPAIGN/downloads/source-metadata/model.safetensors.index.json
CAPTURE=$CAMPAIGN/teacher/calibration/main-ep4-full
ROLES=$CAMPAIGN/roles/roles-v3.json
CHUNKS=$CAMPAIGN/chunks-v3/$ROTATION
EVIDENCE=$CAMPAIGN/evidence-v3/$ROTATION
LOGS=$CAMPAIGN/logs-v3/$ROTATION/layer-$(printf '%03d' "$LAYER")
LOCK=/run/lock/klc/model-stack.lock
mkdir -p "$CHUNKS" "$EVIDENCE" "$LOGS"

exec 9>"$LOCK"
flock -w 900 9
timer_was_active=false
backend_was_active=false
systemctl is-active --quiet klc-model-stack.timer && timer_was_active=true
systemctl --user is-active --quiet klc-backend.service && backend_was_active=true
restore() {
  set +e
  [ "$backend_was_active" = true ] && systemctl --user start klc-backend.service
  [ "$timer_was_active" = true ] && sudo -n systemctl start klc-model-stack.timer
}
trap restore EXIT
[ "$timer_was_active" = true ] && sudo -n systemctl stop klc-model-stack.timer
[ "$backend_was_active" = true ] && systemctl --user stop klc-backend.service

printf -v L3 '%03d' "$LAYER"
pids=()
for gpu in 0 1 2 3; do
  start=$((gpu * 72)); end=$((start + 72))
  printf -v S3 '%03d' "$start"
  printf -v E3 '%03d' "$end"
  out="$CHUNKS/$ROTATION-layer-$L3-experts-$S3-$E3.safetensors"
  receipt="$EVIDENCE/$ROTATION-layer-$L3-experts-$S3-$E3.json"
  extra=()
  [ -n "$ROTATION_FILE" ] && extra+=(--rotation-file "$ROTATION_FILE")
  CUDA_VISIBLE_DEVICES=$gpu /home/brandonmusic/klc-env/bin/python -m glm53_nvfp4.quantize_layer \
    --source "$SOURCE" --source-index "$SOURCE_INDEX" --capture-root "$CAPTURE" --roles "$ROLES" \
    --output "$out" --receipt "$receipt" --layer "$LAYER" --expert-start "$start" --expert-end "$end" \
    --device cuda:0 --max-samples 256 --search-grid 8 --rotation "$ROTATION" --projections gate-up \
    "${extra[@]}" >"$LOGS/gpu-$gpu.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[ "$failed" -eq 0 ] || { echo "one or more layer chunks failed; see $LOGS" >&2; exit 1; }

receipts=()
for gpu in 0 1 2 3; do
  start=$((gpu * 72)); end=$((start + 72))
  printf -v S3 '%03d' "$start"
  printf -v E3 '%03d' "$end"
  receipts+=(--receipt "$EVIDENCE/$ROTATION-layer-$L3-experts-$S3-$E3.json")
done
/usr/bin/python3 -m glm53_nvfp4.validate_layer --layer "$LAYER" --projections gate-up "${receipts[@]}" \
  --output "$EVIDENCE/$ROTATION-layer-$L3-validation.json" | tee "$LOGS/layer-validation.log"
echo "$ROTATION layer $LAYER complete"
