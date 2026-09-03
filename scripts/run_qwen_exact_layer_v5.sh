#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "usage: $0 LAYER identity|had16|learned [ROTATION_FILE]" >&2
  exit 2
fi
LAYER=$1
ROTATION=$2
ROTATION_FILE=${3:-}
ROTATION_SCOPE=${GLM53_ROTATION_SCOPE:-all}
[[ "$LAYER" =~ ^[0-9]+$ ]] && [ "$LAYER" -ge 3 ] && [ "$LAYER" -le 44 ] || { echo "layer must be 3..44" >&2; exit 2; }
[ "$ROTATION" = identity ] || [ "$ROTATION" = had16 ] || [ "$ROTATION" = learned ] || { echo "invalid rotation" >&2; exit 2; }
[ "$ROTATION" != learned ] || [ -n "$ROTATION_FILE" ] || { echo "learned requires ROTATION_FILE" >&2; exit 2; }
[ "$ROTATION_SCOPE" = gate-up ] || [ "$ROTATION_SCOPE" = mid-only ] || [ "$ROTATION_SCOPE" = all ] || { echo "invalid GLM53_ROTATION_SCOPE" >&2; exit 2; }

REPO=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53
CAMPAIGN=/media/brandonmusic/klcstore/bmxfp4-glm53
SOURCE=$CAMPAIGN/downloads/GLM-5.3-Flash-BF16
SOURCE_INDEX=$CAMPAIGN/downloads/source-metadata/model.safetensors.index.json
CAPTURE=$CAMPAIGN/teacher/calibration/main-ep4-full
ROLES=$CAMPAIGN/roles/roles-v3.json
EXACT_VERSION=${GLM53_EXACT_VERSION:-v6}
ARM=$ROTATION
[ "$ROTATION_SCOPE" = all ] || ARM=$ROTATION-$ROTATION_SCOPE
ROOT=$CAMPAIGN/exact-$EXACT_VERSION/layer-$(printf '%03d' "$LAYER")/$ARM
CHUNKS=$ROOT/chunks
EVIDENCE=$ROOT/evidence
LOGS=$ROOT/logs
CANDIDATE=$ROOT/candidate
LOCK=/run/lock/klc/model-stack.lock
PRODUCTION=glm53-flash-exl3-k4-tp4-vision-mtp3
mkdir -p "$CHUNKS" "$EVIDENCE" "$LOGS"

exec 9>"$LOCK"
flock -w 900 9
timer_was_active=false
backend_was_active=false
production_was_running=false
systemctl is-active --quiet klc-model-stack.timer && timer_was_active=true
systemctl --user is-active --quiet klc-backend.service && backend_was_active=true
[ "$(docker inspect -f '{{.State.Running}}' "$PRODUCTION" 2>/dev/null || true)" = true ] && production_was_running=true
restore() {
  set +e
  if [ "$production_was_running" = true ] && [ "$(docker inspect -f '{{.State.Running}}' "$PRODUCTION" 2>/dev/null || true)" != true ]; then
    docker start "$PRODUCTION" >/dev/null
  fi
  [ "$backend_was_active" = false ] || systemctl --user start klc-backend.service
  [ "$timer_was_active" = false ] || sudo -n systemctl start klc-model-stack.timer
}
trap restore EXIT
[ "$timer_was_active" = false ] || sudo -n systemctl stop klc-model-stack.timer
[ "$backend_was_active" = false ] || systemctl --user stop klc-backend.service
[ "$production_was_running" = false ] || docker stop --time 120 "$PRODUCTION" >/dev/null

printf -v L3 '%03d' "$LAYER"
pids=()
for gpu in 0 1 2 3; do
  start=$((gpu * 72)); end=$((start + 72))
  printf -v S3 '%03d' "$start"
  printf -v E3 '%03d' "$end"
  extra=()
  [ -z "$ROTATION_FILE" ] || extra+=(--rotation-file "$ROTATION_FILE")
  CUDA_VISIBLE_DEVICES=$gpu /home/brandonmusic/klc-env/bin/python -m glm53_nvfp4.quantize_layer \
    --source "$SOURCE" --source-index "$SOURCE_INDEX" --capture-root "$CAPTURE" --roles "$ROLES" \
    --output "$CHUNKS/$ROTATION-layer-$L3-experts-$S3-$E3.safetensors" \
    --receipt "$EVIDENCE/$ROTATION-layer-$L3-experts-$S3-$E3.json" \
    --layer "$LAYER" --expert-start "$start" --expert-end "$end" --device cuda:0 \
    --max-samples 10000 --search-grid 12 --rotation "$ROTATION" --rotation-scope "$ROTATION_SCOPE" \
    --gptq-geometry full --projections all "${extra[@]}" >"$LOGS/quant-gpu-$gpu.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[ "$failed" -eq 0 ] || { echo "one or more exact quantization chunks failed" >&2; exit 1; }

receipts=()
chunks=()
for gpu in 0 1 2 3; do
  start=$((gpu * 72)); end=$((start + 72))
  printf -v S3 '%03d' "$start"
  printf -v E3 '%03d' "$end"
  receipts+=(--receipt "$EVIDENCE/$ROTATION-layer-$L3-experts-$S3-$E3.json")
  chunks+=(--chunk "$CHUNKS/$ROTATION-layer-$L3-experts-$S3-$E3.safetensors")
done
/usr/bin/python3 -m glm53_nvfp4.validate_layer --layer "$LAYER" --projections all \
  "${receipts[@]}" --output "$EVIDENCE/layer-validation.json" | tee "$LOGS/layer-validation.log"

scale_extra=()
[ -z "$ROTATION_FILE" ] || scale_extra+=(--rotation-file "$ROTATION_FILE")
CUDA_VISIBLE_DEVICES=0 /home/brandonmusic/klc-env/bin/python -m glm53_nvfp4.calibrate_input_scale \
  --capture-root "$CAPTURE" --source "$SOURCE" --source-index "$SOURCE_INDEX" \
  --roles "$ROLES" --layer "$LAYER" --rotation "$ROTATION" --rotation-scope "$ROTATION_SCOPE" \
  --device cuda:0 --max-samples 10000 --output "$CHUNKS/$ROTATION-layer-$L3-input-scales.safetensors" \
  --receipt "$EVIDENCE/input-scales.json" "${scale_extra[@]}" >"$LOGS/input-scales.log" 2>&1
chunks+=(--chunk "$CHUNKS/$ROTATION-layer-$L3-input-scales.safetensors")

cd "$REPO"
/usr/bin/python3 -m glm53_nvfp4.candidate \
  --carrier /home/brandonmusic/models/GLM-5.3-Flash-NVFP4 \
  --output "$CANDIDATE" "${chunks[@]}" >"$LOGS/candidate-overlay.json"
sha256sum "$CANDIDATE/model.safetensors.index.json" >"$EVIDENCE/candidate-index.sha256"
echo "$ROTATION exact Qwen-style layer $LAYER candidate complete"
