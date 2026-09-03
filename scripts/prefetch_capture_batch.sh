#!/usr/bin/env bash
set -euo pipefail

[ "$#" -ge 1 ] || { echo "usage: $0 LAYER..." >&2; exit 2; }
CAMPAIGN=/media/brandonmusic/klcstore/bmxfp4-glm53
CAPTURE=$CAMPAIGN/teacher/calibration/main-ep4-full
files=()
for layer in "$@"; do
  [[ "$layer" =~ ^[0-9]+$ ]] && [ "$layer" -ge 3 ] && [ "$layer" -le 44 ] || { echo "invalid layer $layer" >&2; exit 2; }
  printf -v l3 '%03d' "$layer"
  files+=(
    "calibration/main-ep4-full/layers/layer-$l3/hidden.bf16.bin"
    "calibration/main-ep4-full/layers/layer-$l3/topk_ids.u16le.bin"
    "calibration/main-ep4-full/layers/layer-$l3/topk_weights.f32le.bin"
  )
done
HF_XET_HIGH_PERFORMANCE=1 /home/brandonmusic/.local/bin/hf download \
  brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits "${files[@]}" \
  --type dataset --revision 95f4fdd94bf29989db2e0d1054e4931f55edb6aa \
  --local-dir "$CAMPAIGN/teacher" --max-workers 12 --format agent
for layer in "$@"; do
  PYTHONPATH=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53 /usr/bin/python3 -m glm53_nvfp4.verify_capture \
    --capture-root "$CAPTURE" --layer "$layer" \
    --output "$CAMPAIGN/evidence-v3/capture-layer-$(printf '%03d' "$layer").json"
done
