#!/usr/bin/env bash
set -euo pipefail

REPO=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53
CAMPAIGN=/media/brandonmusic/klcstore/bmxfp4-glm53
ROOT="$CAMPAIGN/codec-v2/output-aware/causal-layer-sweep-v1"
SOURCE="$CAMPAIGN/downloads/GLM-5.3-Flash-BF16"
SOURCE_INDEX="$SOURCE/model.safetensors.index.json"
CAPTURE_ROOT="$CAMPAIGN/teacher/calibration/main-ep4-full"
ROLES="$CAMPAIGN/roles/roles-v3.json"
ENCODER_ANALYSIS="$CAMPAIGN/codec-v2/output-aware/sealed/hessian-16expert-v1/analysis.json"
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

if [ ! -f "$ROOT/analysis-plan.json" ]; then
  PYTHONPATH=. python3 -m glm53_nvfp4.prepare_p8_causal_layer_sweep \
    --source-index "$SOURCE_INDEX" \
    --capture-manifest "$CAPTURE_ROOT/capture-manifest.json" \
    --roles "$ROLES" \
    --encoder-analysis "$ENCODER_ANALYSIS" \
    --output "$ROOT/analysis-plan.json"
fi

for layer in $(seq 3 44); do
  layer_root="$ROOT/layer-$(printf '%03d' "$layer")"
  mkdir -p "$layer_root"
  if [ ! -f "$layer_root/analysis-plan.json" ]; then
    PYTHONPATH=. python3 -m glm53_nvfp4.prepare_hessian_trellis_full_screen \
      --projection-analysis "$ENCODER_ANALYSIS" \
      --source-index "$SOURCE_INDEX" \
      --capture-manifest "$CAPTURE_ROOT/capture-manifest.json" \
      --roles "$ROLES" \
      --layer "$layer" \
      --output "$layer_root/analysis-plan.json"
  fi
done

worker() {
  local gpu=$1
  local layer
  for layer in $(seq "$((3 + gpu))" 4 44); do
    local layer_root="$ROOT/layer-$(printf '%03d' "$layer")"
    if [ ! -f "$layer_root/raw.json" ]; then
      CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. python3 -m glm53_nvfp4.screen_hessian_trellis_full_experts \
        --plan "$layer_root/analysis-plan.json" \
        --source "$SOURCE" \
        --source-index "$SOURCE_INDEX" \
        --capture-root "$CAPTURE_ROOT" \
        --roles "$ROLES" \
        --output "$layer_root/raw.json" \
        --device cuda:0 >"$layer_root/screen.log" 2>&1
    fi
    if [ ! -f "$layer_root/analysis.json" ]; then
      PYTHONPATH=. python3 -m glm53_nvfp4.analyze_hessian_trellis_full_experts \
        --plan "$layer_root/analysis-plan.json" \
        --input "$layer_root/raw.json" \
        --output "$layer_root/analysis.json" >"$layer_root/analysis.log" 2>&1
    fi
    echo "gpu=$gpu layer=$layer complete"
  done
}

pids=()
for gpu in 0 1 2 3; do
  worker "$gpu" >"$ROOT/logs/gpu-$gpu.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [ "$status" -ne 0 ]; then
  echo "one or more layer-sweep workers failed" >&2
  exit 1
fi

PYTHONPATH=. python3 -m glm53_nvfp4.analyze_p8_causal_layer_sweep \
  --plan "$ROOT/analysis-plan.json" \
  --root "$ROOT" \
  --output "$ROOT/analysis.json" | tee "$ROOT/logs/aggregate.log"
sha256sum "$ROOT/analysis.json"
