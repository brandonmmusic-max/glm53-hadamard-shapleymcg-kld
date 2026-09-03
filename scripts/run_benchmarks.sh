#!/usr/bin/env bash
set -euo pipefail

QUALIFICATION=${1:-/media/brandonmusic/klcstore/bmxfp4-glm53/evidence/confirmation-analysis.json}
python3 - "$QUALIFICATION" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
if x.get('decision') != 'pass': raise SystemExit('KLD qualification did not pass; performance suite blocked')
PY

CAMPAIGN=/media/brandonmusic/klcstore/bmxfp4-glm53
MODEL_DIR=$CAMPAIGN/candidates/uniform-gptq
MODEL_NAME=GLM-5.3-Flash-NVFP4-V2
BENCH=$CAMPAIGN/tooling/llm-inference-bench
BENCH_COMMIT=d115feee75095081bda2520aa046986a8885f449
IMAGE=klc/glm53-flash-nvfp4:r19-sm120-tp4-ep4-dcp4-v79-dflash2-packed-aux-candidate
TEST=glm53-nvfp4-v2-bench
PRODUCTION=glm53-flash-exl3-k4-tp4-vision-mtp3
PORT=8016
LOCK=/run/lock/klc/model-stack.lock
CACHE_DIR=/home/brandonmusic/KLC_SANDBOXES/glm53-exl3-k4-sm120/cache-dflash2-nvfp4-v77
SESSION=$CAMPAIGN/benchmarks/$(date -u +%Y%m%dT%H%M%SZ)-candidate-target-only
mkdir -p "$SESSION"

[ "$(git -C "$BENCH" rev-parse HEAD)" = "$BENCH_COMMIT" ]
[ -z "$(git -C "$BENCH" status --porcelain)" ]
git -C "$BENCH" show --no-patch --format=fuller HEAD >"$SESSION/benchmark-commit.txt"
sha256sum "$BENCH/llm_decode_bench.py" >"$SESSION/benchmark-code.sha256"
sha256sum "$QUALIFICATION" >"$SESSION/qualification.sha256"

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

monitor_pid=
restore() {
  set +e
  [ -z "$monitor_pid" ] || kill "$monitor_pid" 2>/dev/null || true
  [ -z "$monitor_pid" ] || wait "$monitor_pid" 2>/dev/null || true
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
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e NVIDIA_TF32_OVERRIDE=0 \
  -v "$MODEL_DIR:/model:ro" \
  -v /home/brandonmusic/models/GLM-5.3-Flash-NVFP4:/home/brandonmusic/models/GLM-5.3-Flash-NVFP4:ro \
  -v "$CAMPAIGN:$CAMPAIGN:ro" -v "$CACHE_DIR:/cache:rw" \
  "$IMAGE" -lc "exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve /model \
    --served-model-name $MODEL_NAME --host 0.0.0.0 --port $PORT --language-model-only \
    --tensor-parallel-size 4 --enable-expert-parallel --decode-context-parallel-size 4 \
    --dcp-comm-backend a2a --dtype bfloat16 --quantization modelopt --load-format safetensors \
    --attention-backend FLASHINFER_MLA_SPARSE_SM120 --kv-cache-dtype fp8_ds_mla \
    --max-model-len 65536 --max-num-batched-tokens 2048 --max-num-seqs 16 --gpu-memory-utilization 0.94 \
    --enable-chunked-prefill --no-enable-prefix-caching --generation-config /model \
    --reasoning-parser glm45 --disable-custom-all-reduce --enforce-eager" >/dev/null

deadline=$((SECONDS + 2400))
until curl -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models" >"$SESSION/models.json" 2>/dev/null && grep -q "$MODEL_NAME" "$SESSION/models.json"; do
  [ "$(docker inspect -f '{{.State.Running}}' "$TEST" 2>/dev/null || true)" = true ] || exit 1
  [ "$SECONDS" -lt "$deadline" ] || exit 1
  sleep 5
done
docker inspect "$TEST" >"$SESSION/container.json"
docker logs "$TEST" >"$SESSION/server-ready.log" 2>&1 || true
grep -Ei "Using .*NvFp4|NvFp4.*backend|FLASHINFER_MLA_SPARSE_SM120|modelopt" "$SESSION/server-ready.log" >"$SESSION/backend-proof.log" || true

(
  echo 'timestamp,index,uuid,temperature_c,utilization_pct,memory_used_mib,power_w,graphics_clock_mhz,memory_clock_mhz'
  while true; do
    stamp=$(date --iso-8601=seconds)
    nvidia-smi --query-gpu=index,uuid,temperature.gpu,utilization.gpu,memory.used,power.draw,clocks.current.graphics,clocks.current.memory --format=csv,noheader,nounits | sed "s/^/$stamp,/"
    sleep 2
  done
) >"$SESSION/gpu-telemetry.csv" &
monitor_pid=$!

python3 "$BENCH/llm_decode_bench.py" --port "$PORT" --model "$MODEL_NAME" \
  --skip-prefill --contexts 0 --concurrency 1,4,8 --duration 30 --max-tokens 2048 \
  --decode-warmup-seconds 3 --display-mode plain --output "$SESSION/target-only.json" --no-resume \
  2>&1 | tee "$SESSION/target-only.log"
python3 "$BENCH/llm_decode_bench.py" --port "$PORT" --model "$MODEL_NAME" \
  --test-profile estonia --completion-stats-concurrency 1 --completion-stats-runs 5 --max-tokens 20000 \
  --display-mode plain --no-hw-monitor --output "$SESSION/estonia.json" --no-resume \
  2>&1 | tee "$SESSION/estonia.log"
python3 "$BENCH/llm_decode_bench.py" --port "$PORT" --model "$MODEL_NAME" \
  --test-profile lavd --completion-stats-concurrency 1 --completion-stats-runs 5 --max-tokens 20000 \
  --display-mode plain --no-hw-monitor --output "$SESSION/lavd.json" --no-resume \
  2>&1 | tee "$SESSION/lavd.log"

python3 - "$SESSION" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1])
out={'schema':'glm53-nvfp4-v2.benchmark-summary.v1','regimes':{}}
for name in ('target-only','estonia','lavd'):
    path=root/f'{name}.json'; data=json.load(open(path))
    out['regimes'][name]={'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    if name != 'target-only': out['regimes'][name]['selected_summary']=data.get('selected_summary')
(root/'summary.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
PY
log "target-only, Estonia, and LAVD regimes complete"
