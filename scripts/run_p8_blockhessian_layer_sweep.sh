#!/usr/bin/env bash
set -euo pipefail

REPO=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53
CAMPAIGN=/media/brandonmusic/klcstore/bmxfp4-glm53
ROOT="$CAMPAIGN/codec-v2/output-aware/blockhessian-layer-sweep-v1"
SOURCE="$CAMPAIGN/downloads/GLM-5.3-Flash-BF16"
SOURCE_INDEX="$SOURCE/model.safetensors.index.json"
HESSIAN_ROOT="$CAMPAIGN/hessians-v3"
ROLES="$CAMPAIGN/roles/roles-v3.json"
INVALIDATION="$CAMPAIGN/codec-v2/output-aware/causal-layer-sweep-v1/INVALIDATED.json"
LOCK=/run/lock/klc/model-stack.lock

mkdir -p "$ROOT/logs"
cd "$REPO"

exec 9>"$LOCK"
flock -w 900 9
timer_was_active=false
backend_was_active=false
systemctl is-active --quiet klc-model-stack.timer && timer_was_active=true
systemctl --user is-active --quiet klc-backend.service && backend_was_active=true
restore() {
  set +e
  [ "$backend_was_active" = true ] && systemctl --user start klc-backend.service
  [ "$timer_was_active" = true ] && sudo -n systemctl start klc-model-stack.timer
  echo "restored timer=$timer_was_active backend=$backend_was_active"
}
trap restore EXIT
[ "$timer_was_active" = true ] && sudo -n systemctl stop klc-model-stack.timer
[ "$backend_was_active" = true ] && systemctl --user stop klc-backend.service

if [ ! -f "$ROOT/hessian-manifest.json" ]; then
  PYTHONPATH=. python3 -m glm53_nvfp4.hash_tree \
    --root "$HESSIAN_ROOT" \
    --output "$ROOT/hessian-manifest.json"
fi
if [ ! -f "$ROOT/analysis-plan.json" ]; then
  PYTHONPATH=. python3 -m glm53_nvfp4.prepare_p8_blockhessian_layer_sweep \
    --source-index "$SOURCE_INDEX" \
    --hessian-manifest "$ROOT/hessian-manifest.json" \
    --roles "$ROLES" \
    --invalidated-causal-plan "$INVALIDATION" \
    --output "$ROOT/analysis-plan.json"
fi

worker() {
  local gpu=$1
  local slot=$2
  local workers=$3
  local layer
  for layer in $(seq "$((7 + slot))" "$workers" 44); do
    local layer_root="$ROOT/layer-$(printf '%03d' "$layer")"
    mkdir -p "$layer_root"
    if [ ! -f "$layer_root/raw.json" ]; then
      CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. python3 -m glm53_nvfp4.screen_p8_blockhessian_layer \
        --plan "$ROOT/analysis-plan.json" \
        --layer "$layer" \
        --source "$SOURCE" \
        --source-index "$SOURCE_INDEX" \
        --hessian-file "$HESSIAN_ROOT/layer-$(printf '%03d' "$layer")/experts-000-072.safetensors" \
        --output "$layer_root/raw.json" \
        --device cuda:0 >"$layer_root/screen.log" 2>&1
    fi
    echo "gpu=$gpu layer=$layer complete"
  done
}

pids=()
IFS=',' read -r -a gpu_list <<<"${GLM53_SWEEP_GPUS:-1,2,3}"
worker_count=${#gpu_list[@]}
for slot in "${!gpu_list[@]}"; do
  gpu=${gpu_list[$slot]}
  worker "$gpu" "$slot" "$worker_count" >"$ROOT/logs/gpu-$gpu.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [ "$status" -ne 0 ]; then
  echo "one or more block-Hessian workers failed" >&2
  exit 1
fi

PYTHONPATH=. python3 -m glm53_nvfp4.analyze_p8_blockhessian_layer_sweep \
  --plan "$ROOT/analysis-plan.json" \
  --root "$ROOT" \
  --output "$ROOT/analysis.json" | tee "$ROOT/logs/aggregate.log"
sha256sum "$ROOT/analysis.json"
