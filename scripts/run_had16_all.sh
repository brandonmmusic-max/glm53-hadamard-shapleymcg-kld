#!/usr/bin/env bash
set -euo pipefail

REPO=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53
CAMPAIGN=/media/brandonmusic/klcstore/bmxfp4-glm53
CAPTURE=$CAMPAIGN/teacher/calibration/main-ep4-full
LOG=$CAMPAIGN/logs-v3/had16/campaign.log
mkdir -p "$(dirname "$LOG")"

for first in 4 12 20 28 36 44; do
  last=$((first + 7)); [ "$last" -le 44 ] || last=44
  layers=()
  for ((layer=first; layer<=last; layer++)); do layers+=("$layer"); done
  "$REPO/scripts/prefetch_capture_batch.sh" "${layers[@]}" >>"$LOG" 2>&1
  for layer in "${layers[@]}"; do
    "$REPO/scripts/run_rotation_layer.sh" "$layer" had16 >>"$LOG" 2>&1
    printf -v l3 '%03d' "$layer"
    unlink -- "$CAPTURE/layers/layer-$l3/hidden.bf16.bin"
    unlink -- "$CAPTURE/layers/layer-$l3/topk_ids.u16le.bin"
    unlink -- "$CAPTURE/layers/layer-$l3/topk_weights.f32le.bin"
  done
done

chunks=()
while IFS= read -r chunk; do chunks+=(--chunk "$chunk"); done < <(find "$CAMPAIGN/chunks-v3/had16" -maxdepth 1 -type f -name 'had16-layer-*.safetensors' | sort)
PYTHONPATH="$REPO" /usr/bin/python3 -m glm53_nvfp4.candidate \
  --carrier "$CAMPAIGN/candidates/uniform-gptq" \
  --output "$CAMPAIGN/candidates-v3/had16-full" "${chunks[@]}" \
  >"$CAMPAIGN/evidence-v3/had16/full-overlay.json"
sha256sum "$CAMPAIGN/candidates-v3/had16-full/model.safetensors.index.json" \
  >"$CAMPAIGN/evidence-v3/had16/full-index.sha256"
