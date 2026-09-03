#!/usr/bin/env bash
set -euo pipefail

REPO=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53
CAMPAIGN=/media/brandonmusic/klcstore/bmxfp4-glm53
SOURCE=$CAMPAIGN/downloads/GLM-5.3-Flash-BF16
INDEX=$CAMPAIGN/downloads/source-metadata/model.safetensors.index.json
EVIDENCE=$CAMPAIGN/evidence
CAPTURE=$CAMPAIGN/teacher/calibration/main-ep4-full/layers
LOG=$CAMPAIGN/logs/capture-prefetch-queue.log
mkdir -p "$CAPTURE"

log() { echo "[$(date --iso-8601=seconds)] $*" | tee -a "$LOG"; }

while systemctl --user is-active --quiet glm53-nvfp4-v2-campaign.service; do
  current=$(pgrep -af 'run_quant_layer.sh [0-9]+' | sed -n 's/.*run_quant_layer.sh \([0-9][0-9]*\).*/\1/p' | head -1 || true)
  state=$(/usr/bin/python3 - "$INDEX" "$SOURCE" "$EVIDENCE" "$CAPTURE" "${current:-none}" <<'PY'
import json
import pathlib
import sys

index = json.load(open(sys.argv[1]))["weight_map"]
source = pathlib.Path(sys.argv[2])
evidence = pathlib.Path(sys.argv[3])
capture = pathlib.Path(sys.argv[4])
current = None if sys.argv[5] == "none" else int(sys.argv[5])

def captured(layer: int) -> bool:
    root = capture / f"layer-{layer:03d}"
    return any((root / name).is_file() for name in (
        "hidden.bf16.bin", "topk_ids.u16le.bin", "topk_weights.f32le.bin"
    ))

ahead = [layer for layer in range(3, 45)
         if layer != current
         and not (evidence / f"layer-{layer:03d}-validation.json").is_file()
         and captured(layer)]
candidate = None
if not ahead:
    for layer in range(3, 45):
        if layer == current or (evidence / f"layer-{layer:03d}-validation.json").is_file() or captured(layer):
            continue
        files = {name for tensor, name in index.items()
                 if f".layers.{layer}." in tensor and ".mlp.experts." in tensor}
        if files and all((source / name).is_file() for name in files):
            candidate = layer
            break
print(json.dumps({"current": current, "ahead": ahead, "candidate": candidate}))
PY
)
  candidate=$(python3 -c 'import json,sys; x=json.loads(sys.argv[1]); print("" if x["candidate"] is None else x["candidate"])' "$state")
  if [ -n "$candidate" ]; then
    log "prefetching source-ready layer $candidate state=$state"
    "$REPO/scripts/prefetch_capture.sh" "$candidate"
    log "verified prefetched layer $candidate"
  else
    sleep 20
  fi
done
log "campaign runner inactive; prefetch queue exiting"
