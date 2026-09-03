#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 5 ]; then
  echo "usage: $0 MODEL_DIR SESSION_DIR [identity|had16|learned] [LAYERS] [ROTATION_FILE]" >&2
  exit 2
fi

MODEL_DIR=$(readlink -f "$1")
SESSION=$(readlink -m "$2")
REPO=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53
ROTATION=${3:-identity}
ROTATED_LAYERS=${4:-3-44}
ROTATION_FILE=${5:-}
[ "$ROTATION" = identity ] || [ "$ROTATION" = had16 ] || [ "$ROTATION" = learned ] || { echo "invalid rotation" >&2; exit 2; }
[ "$ROTATION" != learned ] || [ -n "$ROTATION_FILE" ] || { echo "learned rotation requires a file" >&2; exit 2; }
IMAGE=klc/glm53-flash-nvfp4:r19-sm120-tp4-ep4-dcp4-v79-dflash2-packed-aux-candidate
MODEL_NAME=GLM-5.3-Flash-NVFP4-V2-PILOT
TEST=glm53-nvfp4-v2-canary
PRODUCTION=glm53-flash-exl3-k4-tp4-vision-mtp3
PORT=8016
LOCK=/run/lock/klc/model-stack.lock
CACHE_DIR=/home/brandonmusic/KLC_SANDBOXES/glm53-exl3-k4-sm120/cache-dflash2-nvfp4-v77
LEARNED_CHUNK_ROOT=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-v3-large
CAMPAIGN=/media/brandonmusic/klcstore/bmxfp4-glm53
mkdir -p "$SESSION"

rotation_env=()
rotation_mount=()
if grep -q '"quant_algo": "MXFP6"' "$MODEL_DIR/config.json"; then
  rotation_env+=( -e PYTHONPATH=/runtime-patch:/opt/exllamav3:/opt/infernal-invocation/vllm:/opt/infernal-invocation/b12x -e GLM53_MIXED_MXFP6=1 -e B12X_ENABLE_FP6=1 -e B12X_FP6_MODEL_DIR=/model )
  rotation_mount+=( -v "$REPO/runtime_patch:/runtime-patch:ro" )
fi
if [ "$ROTATION" != identity ]; then
  rotation_env+=( -e PYTHONPATH=/runtime-patch:/opt/exllamav3:/opt/infernal-invocation/vllm:/opt/infernal-invocation/b12x -e GLM53_ROUTED_ROTATION="$ROTATION" -e GLM53_ROTATED_LAYERS="$ROTATED_LAYERS" )
  if [ ${#rotation_mount[@]} -eq 0 ]; then
    rotation_mount+=( -v "$REPO/runtime_patch:/runtime-patch:ro" )
  fi
fi
if [ "$ROTATION" = learned ]; then
  ROTATION_FILE=$(readlink -f "$ROTATION_FILE")
  rotation_env+=( -e GLM53_ROTATION_FILE=/rotation/rotations.safetensors )
  rotation_mount+=( -v "$ROTATION_FILE:/rotation/rotations.safetensors:ro" )
fi

log() { echo "[$(date --iso-8601=seconds)] $*" | tee -a "$SESSION/session.log"; }

exec 9>"$LOCK"
flock -w 900 9
timer_was_active=false
backend_was_active=false
production_was_running=false
systemctl is-active --quiet klc-model-stack.timer && timer_was_active=true
systemctl --user is-active --quiet klc-backend.service && backend_was_active=true
[ "$(docker inspect -f '{{.State.Running}}' "$PRODUCTION" 2>/dev/null || true)" = true ] && production_was_running=true
log "state timer=$timer_was_active backend=$backend_was_active production=$production_was_running"

restore() {
  set +e
  docker logs "$TEST" >"$SESSION/server-final.log" 2>&1 || true
  docker rm -f "$TEST" >/dev/null 2>&1 || true
  if [ "$production_was_running" = true ] && [ "$(docker inspect -f '{{.State.Running}}' "$PRODUCTION" 2>/dev/null || true)" != true ]; then
    docker start "$PRODUCTION" >/dev/null
  fi
  [ "$backend_was_active" = true ] && systemctl --user start klc-backend.service
  [ "$timer_was_active" = true ] && sudo -n systemctl start klc-model-stack.timer
  log "restored timer=$timer_was_active backend=$backend_was_active production=$production_was_running"
}
trap restore EXIT

[ "$timer_was_active" = true ] && sudo -n systemctl stop klc-model-stack.timer
[ "$backend_was_active" = true ] && systemctl --user stop klc-backend.service
[ "$production_was_running" = true ] && docker stop --time 120 "$PRODUCTION" >/dev/null
docker rm -f "$TEST" >/dev/null 2>&1 || true

docker run -d --name "$TEST" --gpus all --network host --shm-size 32g --restart no --entrypoint /bin/bash \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e OMP_NUM_THREADS=2 \
  -e NCCL_IB_DISABLE=1 -e NCCL_P2P_DISABLE=0 -e NCCL_P2P_LEVEL=4 -e NCCL_PROTO=LL,LL128,Simple \
  -e NCCL_CUMEM_ENABLE=0 -e VLLM_USE_B12X_DCP_A2A=1 -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  -e VLLM_PCIE_ALLREDUCE_BACKEND=cpp -e VLLM_CPP_AR_1STAGE_NCCL_CUTOFF=56KB \
  -e VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS=0 -e KV_FP8_ROPE=0 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e NVIDIA_TF32_OVERRIDE=0 \
  "${rotation_env[@]}" \
  -v "$MODEL_DIR:/model:ro" \
  -v /home/brandonmusic/models/GLM-5.3-Flash-NVFP4:/home/brandonmusic/models/GLM-5.3-Flash-NVFP4:ro \
  -v "$LEARNED_CHUNK_ROOT:$LEARNED_CHUNK_ROOT:ro" \
  -v "$CAMPAIGN:$CAMPAIGN:ro" -v "$CACHE_DIR:/cache:rw" "${rotation_mount[@]}" \
  "$IMAGE" -lc "exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve /model \
    --served-model-name $MODEL_NAME --host 0.0.0.0 --port $PORT --language-model-only \
    --tensor-parallel-size 4 --enable-expert-parallel --decode-context-parallel-size 4 \
    --dcp-comm-backend a2a --dtype bfloat16 --quantization modelopt --load-format safetensors \
    --attention-backend FLASHINFER_MLA_SPARSE_SM120 --kv-cache-dtype fp8_ds_mla \
    --max-model-len 32768 --max-num-batched-tokens 2048 --max-num-seqs 1 --gpu-memory-utilization 0.94 \
    --enable-chunked-prefill --no-enable-prefix-caching --generation-config /model \
    --reasoning-parser glm45 --disable-custom-all-reduce --enforce-eager" >/dev/null

deadline=$((SECONDS + 2400))
until curl -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models" >"$SESSION/models.json" 2>/dev/null; do
  if [ "$(docker inspect -f '{{.State.Running}}' "$TEST" 2>/dev/null || true)" != true ]; then
    log "server exited before readiness"
    exit 1
  fi
  [ "$SECONDS" -ge "$deadline" ] && { log "server readiness timeout"; exit 1; }
  sleep 5
done
docker inspect "$TEST" >"$SESSION/container.json"
docker logs "$TEST" >"$SESSION/server-ready.log" 2>&1 || true
curl -fsS --max-time 300 "http://127.0.0.1:$PORT/v1/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_NAME\",\"prompt\":\"Give three concise reasons evidence matters in engineering.\",\"max_tokens\":96,\"temperature\":0,\"seed\":7}" \
  >"$SESSION/canary.json"
python3 - "$SESSION/canary.json" <<'PY'
import json, sys
x=json.load(open(sys.argv[1]))
text=x['choices'][0]['text'].strip()
if len(text) < 20 or len(set(text.split())) < 5:
    raise SystemExit(f'degenerate canary: {text!r}')
print(text)
PY
grep -Ei "Using .*NvFp4|NvFp4.*backend|FLASHINFER_MLA_SPARSE_SM120|modelopt|GLM53_BLOCK_ROTATION_PATCH_ACTIVE|GLM53_MIXED_MXFP6_PATCH_ACTIVE" "$SESSION/server-ready.log" >"$SESSION/backend-proof.log" || true
[ "$ROTATION" = identity ] || grep -q "GLM53_BLOCK_ROTATION_PATCH_ACTIVE mode=$ROTATION layers=$ROTATED_LAYERS" "$SESSION/server-ready.log"
if grep -q '"quant_algo": "MXFP6"' "$MODEL_DIR/config.json"; then
  grep -q 'GLM53_MIXED_MXFP6_PATCH_ACTIVE' "$SESSION/server-ready.log"
fi
log "candidate canary passed"
