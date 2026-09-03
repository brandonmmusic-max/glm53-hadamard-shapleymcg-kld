#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "usage: $0 LAYER ROTATION [ROTATION_FILE]" >&2
  exit 2
fi
LAYER=$1
ROTATION=$2
ROTATION_FILE=${3:-}
[[ "$LAYER" =~ ^[0-9]+$ ]] && [ "$LAYER" -ge 3 ] && [ "$LAYER" -le 44 ] || { echo "layer must be 3..44" >&2; exit 2; }
[ "$ROTATION" = identity ] || [ "$ROTATION" = had16 ] || [ "$ROTATION" = learned ] || { echo "invalid rotation" >&2; exit 2; }
[ "$ROTATION" != learned ] || [ -n "$ROTATION_FILE" ] || { echo "learned requires ROTATION_FILE" >&2; exit 2; }

REPO=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53
CAMPAIGN=/media/brandonmusic/klcstore/bmxfp4-glm53
SOURCE=$CAMPAIGN/downloads/GLM-5.3-Flash-BF16
SOURCE_INDEX=$CAMPAIGN/downloads/source-metadata/model.safetensors.index.json
IMAGE=klc/glm53-flash-nvfp4:r19-sm120-tp4-ep4-dcp4-v79-dflash2-packed-aux-candidate
if [ $((LAYER % 2)) -eq 0 ]; then
  ROOT=$CAMPAIGN/mxfp6-v3
else
  ROOT=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-v3-large/mxfp6
fi
mkdir -p "$ROOT/chunks" "$ROOT/evidence"
printf -v L3 '%03d' "$LAYER"

pids=()
for GPU in 0 1 2 3; do
  FIRST=$((GPU * 72))
  LAST=$((FIRST + 72))
  CHUNK="$ROOT/chunks/mxfp6-layer-$L3-experts-$(printf '%03d' "$FIRST")-$(printf '%03d' "$LAST").safetensors"
  RECEIPT="$ROOT/evidence/mxfp6-layer-$L3-experts-$(printf '%03d' "$FIRST")-$(printf '%03d' "$LAST").json"
  if [ -f "$RECEIPT" ] && grep -q '"payload_bpw"' "$RECEIPT"; then
    continue
  fi
  extra=()
  mounts=()
  if [ "$ROTATION" = learned ]; then
    ROTATION_FILE=$(readlink -f "$ROTATION_FILE")
    extra+=(--rotation-file /rotations/rotations.safetensors)
    mounts+=(-v "$ROTATION_FILE:/rotations/rotations.safetensors:ro")
  fi
  docker run --rm --gpus "device=$GPU" --entrypoint /bin/bash \
    -e CUDA_VISIBLE_DEVICES=0 \
    -v "$REPO:/work:ro" -v "$SOURCE:/source:ro" -v "$SOURCE_INDEX:/source-index.json:ro" \
    -v "$ROOT:/output:rw" "${mounts[@]}" "$IMAGE" -lc \
    "PYTHONPATH=/work:/opt/infernal-invocation/b12x:/opt/infernal-invocation/vllm exec python -m glm53_nvfp4.mxfp6_layer \
      --source /source --source-index /source-index.json --output /output/chunks/$(basename "$CHUNK") \
      --receipt /output/evidence/$(basename "$RECEIPT") --layer $LAYER --expert-start $FIRST --expert-end $LAST \
      --device cuda:0 --rotation $ROTATION --block-scale-rule mse ${extra[*]}" \
    >"$ROOT/evidence/$(basename "${RECEIPT%.json}.log")" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

PYTHONPATH="$REPO" /usr/bin/python3 - <<PY
import json
from pathlib import Path
layer=$LAYER
roots=[Path('$CAMPAIGN/mxfp6-v3/evidence'),Path('/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-v3-large/mxfp6/evidence')]
rows=[]
for root in roots:
    rows += [json.loads(p.read_text()) for p in root.glob(f'mxfp6-layer-{layer:03d}-experts-*.json')]
if len(rows) != 4 or sorted((r['expert_start'],r['expert_end']) for r in rows) != [(0,72),(72,144),(144,216),(216,288)]:
    raise SystemExit(f'layer {layer}: incomplete receipt set')
if any(abs(r['payload_bpw'] - 6.250007629394531) > 1e-9 for r in rows):
    raise SystemExit(f'layer {layer}: unexpected exact payload bpw')
print(json.dumps({'layer':layer,'status':'pass','payload_bytes':sum(r['payload_bytes'] for r in rows),'payload_bpw':rows[0]['payload_bpw']}))
PY
