#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 OUTPUT [canary_mxfp6_reap arguments]" >&2
  exit 2
fi

OUTPUT=$1
shift
REPO=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53
CAMPAIGN=/media/brandonmusic/klcstore/bmxfp4-glm53
IMAGE=klc/glm53-flash-nvfp4:r19-sm120-tp4-ep4-dcp4-v79-dflash2-packed-aux-candidate
MODEL=$CAMPAIGN/codec-v2/alphabet/layer3-scalar-mxfp6/candidate
CACHE=/home/brandonmusic/KLC_SANDBOXES/glm53-exl3-k4-sm120/cache-dflash2-nvfp4-v77
CONTAINER=glm53-w6a8-reap-scale
LOCK=/run/lock/klc/model-stack.lock
DYNAMIC_DOWN_SCALE=${GLM53_REAP_DYNAMIC_DOWN_SCALE:-1}

exec 9>"$LOCK"
flock -w 900 9
backend_was_active=false
timer_was_active=false
systemctl --user is-active --quiet klc-backend.service && backend_was_active=true
systemctl is-active --quiet klc-model-stack.timer && timer_was_active=true
restore() {
  set +e
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  [ "$backend_was_active" = true ] && systemctl --user start klc-backend.service
  [ "$timer_was_active" = true ] && sudo -n systemctl start klc-model-stack.timer
}
trap restore EXIT
[ "$backend_was_active" = true ] && systemctl --user stop klc-backend.service
[ "$timer_was_active" = true ] && sudo -n systemctl stop klc-model-stack.timer
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

docker run --rm --name "$CONTAINER" --gpus device=0 --shm-size 16g \
  --entrypoint /opt/venv/bin/python \
  -e CUDA_VISIBLE_DEVICES=0 -e B12X_ENABLE_FP6=1 -e GLM53_MIXED_MXFP6=1 \
  -e B12X_FP6_MODEL_DIR=/model -e GLM53_W6A8_SWIGLU_LIMIT=10 \
  -e B12X_FAST_MATH=0 -e B12X_ENABLE_DYNAMIC_DOWN_SCALE="$DYNAMIC_DOWN_SCALE" \
  -e B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1 \
  -e PYTHONPATH=/runtime-patch/b12x_h16:/runtime-patch:/workspace:/opt/exllamav3:/opt/infernal-invocation/vllm:/opt/infernal-invocation/b12x \
  -v "$REPO:/workspace:ro" -v "$REPO/runtime_patch:/runtime-patch:ro" \
  -v "$CAMPAIGN:/campaign:rw" -v "$MODEL:/model:ro" -v "$CACHE:/cache:rw" \
  "$IMAGE" -m glm53_nvfp4.canary_mxfp6_reap \
  --native /campaign/codec-v2/alphabet/layer3-scalar-mxfp6/chunks/mxfp6-layer-003-experts-000-072.safetensors \
  --dense /campaign/codec-v2/w6a8-diagnosis/decoded-bf16/layer-003-experts-000-072.safetensors \
  --capture-root /campaign/teacher/calibration/main-ep4-full \
  --roles /campaign/roles/roles-v3.json --layer 3 --swiglu-limit 10 \
  --output "$OUTPUT" "$@"
