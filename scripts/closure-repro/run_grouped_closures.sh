#!/usr/bin/env bash
# Grouped M64/N128 device closures at K5 and K4 on the rebuilt image, before any speed run.
set -euo pipefail
REPO=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53/.claude/worktrees/fable-p8-full-coupled-v1
CAMP=/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6
OUT=$CAMP/phase1-campaign-v1
IMAGE=sha256:9846694e7dd3546f6ef3d259ecd2c129a5d54147662a76d32c5a3d15673f5e25
MANIFEST_SHA=acd55382242b3d79d6bdae242ff5268fb298f41c5f88aace121d4d9531afa89a
TRANSFORM=$REPO/experiments/p8-coupled-transform-draw0-silu10-v1.json
cd "$REPO"
export PYTHONPATH="$REPO"
sha() { sha256sum "$1" | cut -d' ' -f1; }

wait_cool() {
  for _ in $(seq 1 120); do
    t=$(nvidia-smi -i 0 --query-gpu=temperature.gpu --format=csv,noheader,nounits | tr -d ' ')
    [ "$t" -gt 75 ] || return 0
    sleep 5
  done
  return 1
}

run_case() {
  local bits=$1 layer=$2 design=$3 sidecar=$4
  local name="grouped-k$bits"
  echo "[$(date --iso-8601=seconds)] === grouped M64 closure K$bits on layer $layer ==="
  [ -f "$sidecar" ] || { echo "missing sidecar $sidecar"; return 1; }
  rm -rf "$OUT/$name"
  wait_cool || { echo "GPU 0 did not cool below 75 C"; return 1; }
  /usr/bin/python3 scripts/run_p8_mixed_rate_prefill_device_closure.py --execute \
    --image "$IMAGE" --gpu-device 0 \
    --sidecar "$sidecar" --sidecar-sha256 "$(sha "$sidecar")" \
    --design "$design" --design-sha256 "$(sha "$design")" \
    --transform "$TRANSFORM" --transform-sha256 "$(sha "$TRANSFORM")" \
    --runtime-manifest-sha256 "$MANIFEST_SHA" \
    --bits "$bits" --layer "$layer" \
    --output "$OUT/$name" >>"$OUT/$name.log" 2>&1 || {
      echo "[$(date --iso-8601=seconds)] K$bits grouped closure FAILED; see $OUT/$name.log"; return 1; }
  echo "[$(date --iso-8601=seconds)] K$bits: $(grep -o '\"decision\": *\"[a-z]*\"' "$OUT/$name/result.json" | head -1)"
}

# K5 is the new path; K4 re-run proves the rate parameterization did not disturb it.
run_case 5 3 "$REPO/results/P8_COUPLED_RATE_DESIGN_K5_V1.json" \
  "$CAMP/rate-candidates-v1/candidate-sidecars/layer-003/k5/p8-layer-003-tp4-rank-0.safetensors"
run_case 4 3 "$REPO/results/P8_COUPLED_SCALE_FLASH_EVIDENCE_PREPARATION_V3.json" \
  "$CAMP/full-coupled-p8-all42-v1/sidecars/p8-layer-003-tp4-rank-0.safetensors"
echo "[$(date --iso-8601=seconds)] grouped closures complete"
