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
ROLES=$CAMPAIGN/roles/roles.json
CHUNKS=$CAMPAIGN/chunks
CANDIDATE=$CAMPAIGN/candidates/uniform-gptq
LOGS=$CAMPAIGN/logs/layer-$(printf '%03d' "$LAYER")
LOCK=/run/lock/klc/model-stack.lock
CAPTURE_LOCKS=$CAMPAIGN/locks
PRODUCTION=glm53-flash-exl3-k4-tp4-vision-mtp3
mkdir -p "$CHUNKS" "$CANDIDATE" "$LOGS" "$CAPTURE_LOCKS"

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
  if [ "$production_was_running" = true ] && [ "$(docker inspect -f '{{.State.Running}}' "$PRODUCTION" 2>/dev/null || true)" != true ]; then docker start "$PRODUCTION" >/dev/null; fi
  [ "$backend_was_active" = true ] && systemctl --user start klc-backend.service
  [ "$timer_was_active" = true ] && sudo -n systemctl start klc-model-stack.timer
}
trap restore EXIT
[ "$timer_was_active" = true ] && sudo -n systemctl stop klc-model-stack.timer
[ "$backend_was_active" = true ] && systemctl --user stop klc-backend.service
[ "$production_was_running" = true ] && docker stop --time 120 "$PRODUCTION" >/dev/null

printf -v L3 '%03d' "$LAYER"
CAPTURE_FILES=(
  "calibration/main-ep4-full/layers/layer-$L3/hidden.bf16.bin"
  "calibration/main-ep4-full/layers/layer-$L3/topk_ids.u16le.bin"
  "calibration/main-ep4-full/layers/layer-$L3/topk_weights.f32le.bin"
)
exec 8>"$CAPTURE_LOCKS/capture-layer-$L3.lock"
flock -w 7200 8
HF_XET_HIGH_PERFORMANCE=1 /home/brandonmusic/.local/bin/hf download brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits "${CAPTURE_FILES[@]}" \
  --type dataset --revision 95f4fdd94bf29989db2e0d1054e4931f55edb6aa --local-dir "$CAMPAIGN/teacher" --max-workers 3 --format agent \
  >"$LOGS/capture-download.log" 2>&1
/usr/bin/python3 -m glm53_nvfp4.verify_capture --capture-root "$CAPTURE" --layer "$LAYER" \
  --output "$CAMPAIGN/evidence/capture-layer-$L3.json" | tee "$LOGS/capture-integrity.log"
flock -u 8

pids=()
for gpu in 0 1 2 3; do
  start=$((gpu * 72)); end=$((start + 72))
  out="$CHUNKS/layer-$L3-experts-$(printf '%03d' "$start")-$(printf '%03d' "$end").safetensors"
  receipt="$CAMPAIGN/evidence/layer-$L3-experts-$(printf '%03d' "$start")-$(printf '%03d' "$end").json"
  CUDA_VISIBLE_DEVICES=$gpu /home/brandonmusic/klc-env/bin/python -m glm53_nvfp4.quantize_layer \
    --source "$SOURCE" --source-index "$SOURCE_INDEX" --capture-root "$CAPTURE" --roles "$ROLES" \
    --output "$out" --receipt "$receipt" --layer "$LAYER" --expert-start "$start" --expert-end "$end" \
    --device cuda:0 --max-samples 256 --search-grid 8 >"$LOGS/gpu-$gpu.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[ "$failed" -eq 0 ] || { echo "one or more layer chunks failed; see $LOGS" >&2; exit 1; }

receipts=()
for gpu in 0 1 2 3; do
  start=$((gpu * 72)); end=$((start + 72))
  receipts+=(--receipt "$CAMPAIGN/evidence/layer-$L3-experts-$(printf '%03d' "$start")-$(printf '%03d' "$end").json")
done
/usr/bin/python3 -m glm53_nvfp4.validate_layer --layer "$LAYER" "${receipts[@]}" \
  --output "$CAMPAIGN/evidence/layer-$L3-validation.json" | tee "$LOGS/layer-validation.log"

mapfile -t all_chunks < <(find "$CHUNKS" -maxdepth 1 -type f -name 'layer-*.safetensors' | sort)
args=()
for chunk in "${all_chunks[@]}"; do args+=(--chunk "$chunk"); done
cd "$REPO"
/usr/bin/python3 -m glm53_nvfp4.candidate --carrier /home/brandonmusic/models/GLM-5.3-Flash-NVFP4 --output "$CANDIDATE" "${args[@]}" >"$LOGS/candidate-overlay.json"
sha256sum "$CANDIDATE/model.safetensors.index.json" >"$LOGS/candidate-index.sha256"
# Streamed capture files are recoverable from the pinned Hub revision. Keep their
# verified identities and all quantization evidence, but not 464 GB of local copies.
unlink -- "$CAPTURE/layers/layer-$L3/hidden.bf16.bin"
unlink -- "$CAPTURE/layers/layer-$L3/topk_ids.u16le.bin"
unlink -- "$CAPTURE/layers/layer-$L3/topk_weights.f32le.bin"
echo "layer $LAYER complete"
