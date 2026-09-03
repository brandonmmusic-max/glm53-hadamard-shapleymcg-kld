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
START_LAYER=${1:-3}
[[ "$START_LAYER" =~ ^[0-9]+$ ]] && [ "$START_LAYER" -ge 3 ] && [ "$START_LAYER" -le 44 ] || { echo "start layer must be 3..44" >&2; exit 2; }

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

starts=()
for first in 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39 41 43; do
  [ "$first" -ge "$START_LAYER" ] && starts+=("$first")
done
batch_layers() {
  local first=$1 last layer
  last=$((first + 1)); [ "$last" -le 44 ] || last=44
  for ((layer=first; layer<=last; layer++)); do echo "$layer"; done
}
mapfile -t first_layers < <(batch_layers "${starts[0]}")
"$REPO/scripts/prefetch_capture_batch.sh" "${first_layers[@]}" >>"$LOG" 2>&1
for ((batch=0; batch<${#starts[@]}; batch++)); do
  first=${starts[$batch]}
  mapfile -t layers < <(batch_layers "$first")
  next_pid=
  if [ $((batch + 1)) -lt "${#starts[@]}" ]; then
    mapfile -t next_layers < <(batch_layers "${starts[$((batch + 1))]}")
    "$REPO/scripts/prefetch_capture_batch.sh" "${next_layers[@]}" >>"$LOG" 2>&1 &
    next_pid=$!
  fi
  for layer in "${layers[@]}"; do
    printf -v l3 '%03d' "$layer"
    if ! grep -q '"status": "pass"' "$CAMPAIGN/evidence-v3/had16/had16-layer-$l3-validation.json" 2>/dev/null; then
      "$REPO/scripts/run_rotation_layer.sh" "$layer" had16 >>"$LOG" 2>&1
    fi
    if ! grep -q '"status": "pass"' "$EVIDENCE/learned-layer-$l3-validation.json" 2>/dev/null; then
      learn_layer "$layer"
    fi
    [ ! -f "$CAPTURE/layers/layer-$l3/hidden.bf16.bin" ] || unlink -- "$CAPTURE/layers/layer-$l3/hidden.bf16.bin"
    [ ! -f "$CAPTURE/layers/layer-$l3/topk_ids.u16le.bin" ] || unlink -- "$CAPTURE/layers/layer-$l3/topk_ids.u16le.bin"
    [ ! -f "$CAPTURE/layers/layer-$l3/topk_weights.f32le.bin" ] || unlink -- "$CAPTURE/layers/layer-$l3/topk_weights.f32le.bin"
  done
  [ -z "$next_pid" ] || wait "$next_pid"
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
