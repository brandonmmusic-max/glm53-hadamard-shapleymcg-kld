#!/usr/bin/env bash
set -euo pipefail

REPO=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53
CAMPAIGN=/media/brandonmusic/klcstore/bmxfp4-glm53
CAPTURE=$CAMPAIGN/teacher/calibration/main-ep4-full
SOURCE=$CAMPAIGN/downloads/GLM-5.3-Flash-BF16
SOURCE_INDEX=$CAMPAIGN/downloads/source-metadata/model.safetensors.index.json
ROLES=$CAMPAIGN/roles/roles-v3.json
LEARNED_ROOT=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-v3-large
ROTATIONS=$CAMPAIGN/rotations-v3
EVIDENCE=$CAMPAIGN/evidence-v3/learned
LOG=$CAMPAIGN/logs-v3/combined-campaign.log
mkdir -p "$LEARNED_ROOT/chunks" "$ROTATIONS/identity" "$ROTATIONS/had16" "$ROTATIONS/selected" "$EVIDENCE" "$(dirname "$LOG")"

learn_layer() {
  local layer=$1 l3 identity_rotation had_rotation identity_receipt had_receipt selected p0 p1
  printf -v l3 '%03d' "$layer"
  identity_rotation="$ROTATIONS/identity/layer-$l3.safetensors"
  had_rotation="$ROTATIONS/had16/layer-$l3.safetensors"
  identity_receipt="$EVIDENCE/fit-layer-$l3-identity.json"
  had_receipt="$EVIDENCE/fit-layer-$l3-had16.json"
  selected="$ROTATIONS/selected/layer-$l3.safetensors"
  common=(--source "$SOURCE" --source-index "$SOURCE_INDEX" --capture-root "$CAPTURE" --roles "$ROLES" --layer "$layer" --experts 16 --steps 100 --batch 16 --lr 0.001 --search-grid 8)
  CUDA_VISIBLE_DEVICES=0 /home/brandonmusic/klc-env/bin/python -m glm53_nvfp4.learn_rotation_layer "${common[@]}" --init identity --device cuda:0 --output "$identity_rotation" --receipt "$identity_receipt" >"$EVIDENCE/fit-layer-$l3-identity.log" 2>&1 &
  p0=$!
  CUDA_VISIBLE_DEVICES=1 /home/brandonmusic/klc-env/bin/python -m glm53_nvfp4.learn_rotation_layer "${common[@]}" --init had16 --device cuda:0 --output "$had_rotation" --receipt "$had_receipt" >"$EVIDENCE/fit-layer-$l3-had16.log" 2>&1 &
  p1=$!
  wait "$p0"
  wait "$p1"
  /usr/bin/python3 -m glm53_nvfp4.select_layer_rotation --layer "$layer" \
    --candidate "$identity_rotation" --receipt "$identity_receipt" \
    --candidate "$had_rotation" --receipt "$had_receipt" \
    --output "$selected" --selection-receipt "$EVIDENCE/fit-layer-$l3-selection.json" >>"$LOG"
  GLM53_V3_CHUNK_ROOT="$LEARNED_ROOT/chunks" "$REPO/scripts/run_rotation_layer.sh" "$layer" learned "$selected" >>"$LOG" 2>&1
}

for first in 3 9 15 21 27 33 39; do
  last=$((first + 5)); [ "$last" -le 44 ] || last=44
  layers=()
  for ((layer=first; layer<=last; layer++)); do layers+=("$layer"); done
  "$REPO/scripts/prefetch_capture_batch.sh" "${layers[@]}" >>"$LOG" 2>&1
  for layer in "${layers[@]}"; do
    printf -v l3 '%03d' "$layer"
    if [ ! -s "$CAMPAIGN/evidence-v3/had16/had16-layer-$l3-validation.json" ]; then
      "$REPO/scripts/run_rotation_layer.sh" "$layer" had16 >>"$LOG" 2>&1
    fi
    learn_layer "$layer"
    unlink -- "$CAPTURE/layers/layer-$l3/hidden.bf16.bin"
    unlink -- "$CAPTURE/layers/layer-$l3/topk_ids.u16le.bin"
    unlink -- "$CAPTURE/layers/layer-$l3/topk_weights.f32le.bin"
  done
done

had_chunks=()
while IFS= read -r chunk; do had_chunks+=(--chunk "$chunk"); done < <(find "$CAMPAIGN/chunks-v3/had16" -maxdepth 1 -type f -name 'had16-layer-*.safetensors' | sort)
PYTHONPATH="$REPO" /usr/bin/python3 -m glm53_nvfp4.candidate --carrier "$CAMPAIGN/candidates/uniform-gptq" \
  --output "$CAMPAIGN/candidates-v3/had16-full" "${had_chunks[@]}" >"$CAMPAIGN/evidence-v3/had16/full-overlay.json"

learned_chunks=()
while IFS= read -r chunk; do learned_chunks+=(--chunk "$chunk"); done < <(find "$LEARNED_ROOT/chunks/learned" -maxdepth 1 -type f -name 'learned-layer-*.safetensors' | sort)
PYTHONPATH="$REPO" /usr/bin/python3 -m glm53_nvfp4.candidate --carrier "$CAMPAIGN/candidates/uniform-gptq" \
  --output "$CAMPAIGN/candidates-v3/learned-full" "${learned_chunks[@]}" >"$EVIDENCE/full-overlay.json"

bundle_args=()
while IFS= read -r rotation; do bundle_args+=(--input "$rotation"); done < <(find "$ROTATIONS/selected" -maxdepth 1 -type f -name 'layer-*.safetensors' | sort)
PYTHONPATH="$REPO" /usr/bin/python3 -m glm53_nvfp4.bundle_rotations "${bundle_args[@]}" \
  --output "$ROTATIONS/learned-full.safetensors" --receipt "$EVIDENCE/full-bundle.json"
sha256sum "$CAMPAIGN/candidates-v3/had16-full/model.safetensors.index.json" "$CAMPAIGN/candidates-v3/learned-full/model.safetensors.index.json" \
  "$ROTATIONS/learned-full.safetensors" >"$CAMPAIGN/evidence-v3/full-candidate-indexes.sha256"
