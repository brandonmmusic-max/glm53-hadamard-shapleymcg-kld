#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
  echo "usage: $0 ROLE RUN_ID MODEL_DIR MODEL_NAME [FREEZE_RECEIPT]" >&2
  exit 2
fi
ROLE=$1
RUN_ID=$2
MODEL_DIR=$(readlink -f "$3")
MODEL_NAME=$4
FREEZE=${5:-}
[ ! -f "$MODEL_DIR/OVERLAY.json" ] || {
  echo "sparse overlays are forbidden in the legacy safetensors KLD launcher; use run_kld_v3.sh with InstantTensor" >&2
  exit 2
}
[ "$ROLE" = selection ] || [ "$ROLE" = confirmation ] || { echo "role must be selection or confirmation" >&2; exit 2; }
[ "$ROLE" != confirmation ] || [ -n "$FREEZE" ] || { echo "confirmation requires freeze receipt" >&2; exit 2; }

REPO=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53
CAMPAIGN=/media/brandonmusic/klcstore/bmxfp4-glm53
IMAGE=klc/glm53-flash-nvfp4:r19-sm120-tp4-ep4-dcp4-v79-dflash2-packed-aux-candidate
IMAGE_DIGEST=sha256:ed027a3a2ff93b9cf60c95f7adfaf676cabc8e040a28cffa7486a262c82fdfbe
TEST=glm53-nvfp4-v2-kld
PRODUCTION=glm53-flash-exl3-k4-tp4-vision-mtp3
PORT=8016
LOCK=/run/lock/klc/model-stack.lock
CACHE_DIR=/home/brandonmusic/KLC_SANDBOXES/glm53-exl3-k4-sm120/cache-dflash2-nvfp4-v77
CAPTURES=$CAMPAIGN/kld/captures/$RUN_ID
SESSION=$CAMPAIGN/kld/sessions/$RUN_ID
mkdir -p "$CAPTURES" "$SESSION" "$CAMPAIGN/kld/records"
cd "$REPO"

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
  if [ "$production_was_running" = true ] && [ "$(docker inspect -f '{{.State.Running}}' "$PRODUCTION" 2>/dev/null || true)" != true ]; then docker start "$PRODUCTION" >/dev/null; fi
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
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e NVIDIA_TF32_OVERRIDE=0 -e VLLM_KLD_CAPTURE_DIR="$CAPTURES" \
  -v "$MODEL_DIR:/model:ro" \
  -v /home/brandonmusic/models/GLM-5.3-Flash-NVFP4:/home/brandonmusic/models/GLM-5.3-Flash-NVFP4:ro \
  -v "$CAMPAIGN:$CAMPAIGN:rw" -v "$CACHE_DIR:/cache:rw" -v "$CAPTURES:$CAPTURES:rw" \
  "$IMAGE" -lc "exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve /model \
    --served-model-name $MODEL_NAME --host 0.0.0.0 --port $PORT --language-model-only \
    --tensor-parallel-size 4 --enable-expert-parallel --decode-context-parallel-size 4 \
    --dcp-comm-backend a2a --dtype bfloat16 --quantization modelopt --load-format safetensors \
    --attention-backend FLASHINFER_MLA_SPARSE_SM120 --kv-cache-dtype fp8_ds_mla \
    --max-model-len 32768 --max-num-batched-tokens 2048 --max-num-seqs 1 --gpu-memory-utilization 0.94 \
    --enable-chunked-prefill --no-enable-prefix-caching --generation-config /model \
    --reasoning-parser glm45 --disable-custom-all-reduce --enforce-eager" >/dev/null

deadline=$((SECONDS + 2400))
until curl -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models" >"$SESSION/models.json" 2>/dev/null && grep -q "$MODEL_NAME" "$SESSION/models.json"; do
  [ "$(docker inspect -f '{{.State.Running}}' "$TEST" 2>/dev/null || true)" = true ] || { log "server exited"; exit 1; }
  [ "$SECONDS" -lt "$deadline" ] || { log "readiness timeout"; exit 1; }
  sleep 5
done
docker inspect "$TEST" >"$SESSION/container.json"
docker image inspect "$IMAGE" --format '{{json .RepoDigests}}' >"$SESSION/image-digests.json"
grep -q "$IMAGE_DIGEST" "$SESSION/image-digests.json"
docker logs "$TEST" >"$SESSION/server-ready.log" 2>&1 || true
grep -Ei "Using .*NvFp4|NvFp4.*backend|FLASHINFER_MLA_SPARSE_SM120|modelopt" "$SESSION/server-ready.log" >"$SESSION/backend-proof.log" || true

args=()
[ -z "$FREEZE" ] || args+=(--freeze-receipt "$FREEZE")
python3 -m glm53_nvfp4.role_eval --role "$ROLE" --roles "$CAMPAIGN/roles/roles.json" \
  --teacher-root "$CAMPAIGN/teacher" --run-root "$CAMPAIGN/kld" --run-id "$RUN_ID" \
  --config-id nvfp4-tp4-ep4-dcp4-eager-nomtp-kvfp8-flashinfer-chunk2048 \
  --url "http://127.0.0.1:$PORT/v1/completions" --model-name "$MODEL_NAME" \
  --capture-root "$CAPTURES" --container "$TEST" --resume "${args[@]}" 2>&1 | tee "$SESSION/role-eval.log"
log "$ROLE KLD run $RUN_ID complete"
