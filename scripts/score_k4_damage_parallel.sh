#!/usr/bin/env bash
# K4-only Shapley damage for every routed layer, one worker per GPU, no encodes.
# Holds the model-stack lock for the whole pass and runs score_layer_rates_all42.sh
# workers with GLM53_RATE_CANDIDATES="" so each layer's receipt lands under
# $RATE_ROOT/receipts/k4-only/damage-layer-LLL.json.
set -euo pipefail

REPO=${GLM53_COUPLED_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CAMPAIGN=${GLM53_CAMPAIGN_ROOT:-/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6}
ROOT=${GLM53_RATE_SCORE_ROOT:-$CAMPAIGN/rate-candidates-v1}
read -r -a GPUS <<<"${GLM53_ENCODER_GPUS:-0 1 2 3}"
LAYERS=${GLM53_RATE_LAYERS:-$(seq 3 44 | tr '\n' ' ')}
SUBDIR=${GLM53_RATE_RECEIPT_SUBDIR:-k4-only}
LOCK=/run/lock/klc/model-stack.lock
: "${GLM53_RATE_DESIGN_K3:?set GLM53_RATE_DESIGN_K3}" "${GLM53_RATE_DESIGN_K5:?set GLM53_RATE_DESIGN_K5}" "${GLM53_MAX_NEW_BYTES:?set GLM53_MAX_NEW_BYTES}"

mkdir -p "$ROOT/logs"
LOG=$ROOT/score.log
log() { echo "[$(date --iso-8601=seconds)] $*" | tee -a "$LOG"; }

exec 9>"$LOCK"
flock -w 900 9
log "K4-only damage pass started gpus=${GPUS[*]} layers='$LAYERS' receipts=$ROOT/receipts/$SUBDIR"

declare -a worker_layers pids
for i in "${!GPUS[@]}"; do worker_layers[$i]=""; done
i=0
for layer in $LAYERS; do
  worker_layers[$i]="${worker_layers[$i]} $layer"
  i=$(( (i + 1) % ${#GPUS[@]} ))
done
for i in "${!GPUS[@]}"; do
  [ -n "${worker_layers[$i]// }" ] || continue
  GLM53_RATE_CANDIDATES="" GLM53_RATE_RECEIPT_SUBDIR=$SUBDIR GLM53_SKIP_LOCK=1 GLM53_SCORE_GPU=${GPUS[$i]} \
    GLM53_RATE_LAYERS="${worker_layers[$i]}" bash "$REPO/scripts/score_layer_rates_all42.sh" \
    >"$ROOT/logs/k4-only-worker-gpu${GPUS[$i]}.log" 2>&1 &
  pids[$i]=$!
  log "worker gpu=${GPUS[$i]} pid=${pids[$i]} layers='${worker_layers[$i]}'"
done
status=0
for i in "${!pids[@]}"; do wait "${pids[$i]}" || { log "worker gpu=${GPUS[$i]} failed (see logs/k4-only-worker-gpu${GPUS[$i]}.log)"; status=1; }; done
[ "$status" -eq 0 ] || exit 1
missing=0
for layer in $LAYERS; do
  [ -f "$ROOT/receipts/$SUBDIR/damage-layer-$(printf '%03d' "$layer").json" ] || { log "layer=$layer has no K4-only receipt"; missing=1; }
done
[ "$missing" -eq 0 ] || exit 1
log "K4-only damage pass complete receipts=$ROOT/receipts/$SUBDIR"
