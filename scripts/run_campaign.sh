#!/usr/bin/env bash
set -euo pipefail

REPO=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53
CAMPAIGN=/media/brandonmusic/klcstore/bmxfp4-glm53
SOURCE=$CAMPAIGN/downloads/GLM-5.3-Flash-BF16
INDEX=$CAMPAIGN/downloads/source-metadata/model.safetensors.index.json
PROGRESS=$CAMPAIGN/logs/campaign-progress.log

cd "$REPO"
while true; do
  layer=$(/usr/bin/python3 - "$INDEX" "$SOURCE" "$CAMPAIGN/evidence" <<'PY'
import json, pathlib, sys
index=json.load(open(sys.argv[1]))['weight_map']
root=pathlib.Path(sys.argv[2]); evidence=pathlib.Path(sys.argv[3])
for layer in range(3,45):
    if (evidence/f'layer-{layer:03d}-validation.json').is_file():
        continue
    files={name for tensor,name in index.items() if f'.layers.{layer}.' in tensor and '.mlp.experts.' in tensor}
    if files and all((root/name).is_file() for name in files):
        print(layer)
        break
PY
)
  if [ -n "$layer" ]; then
    echo "[$(date --iso-8601=seconds)] starting layer $layer" | tee -a "$PROGRESS"
    "$REPO/scripts/run_quant_layer.sh" "$layer"
    echo "[$(date --iso-8601=seconds)] completed layer $layer" | tee -a "$PROGRESS"
    continue
  fi
  remaining=$(/usr/bin/python3 - "$CAMPAIGN/evidence" <<'PY'
import pathlib,sys
p=pathlib.Path(sys.argv[1])
print(sum(not (p/f'layer-{x:03d}-validation.json').is_file() for x in range(3,45)))
PY
)
  [ "$remaining" -gt 0 ] || break
  if ! systemctl --user is-active --quiet hf-download-glm53-bf16-v2.service; then
    echo "source download stopped with $remaining layers unavailable" >&2
    exit 1
  fi
  sleep 20
done
echo "[$(date --iso-8601=seconds)] all routed-expert layers complete" | tee -a "$PROGRESS"
