#!/usr/bin/env bash
set -euo pipefail

REPO=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-smallm-v1
CAMPAIGN=/media/brandonmusic/klcstore/bmxfp4-glm53
ROOT=/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/uniform-p8-all42-v1
OUT=/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-smallm-scheduler-v1/integrated-v1
MODEL=$CAMPAIGN/codec-v2/p8-h128/identity-mcg-layer3-v1/candidate-bf16-pseudoquant
SIDECARS=$ROOT/sidecars
DESIGN=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-all42-runtime/experiments/p8-kld-shapley-native6-v2.json
PATCH=$REPO/runtime_patch
CACHE=/home/brandonmusic/KLC_SANDBOXES/glm53-exl3-k4-sm120/cache-dflash2-nvfp4-v77
BENCH=/home/brandonmusic/KLC_SANDBOXES/glm53-exl3-k4-r10-rebase/tooling/llm-inference-bench/llm_decode_bench.py
PLAN=$REPO/experiments/p8-smallm-integrated-v1.json
IMAGE=sha256:c9eddeca0d9dedf210eaaff918f1bf44d5338202902143f7d646d2325ad475ed
COMMIT=02418d30e245bfae6586cb34b46bafd7c9e6a98f
CONTAINER=glm53-p8-smallm-integrated-v1
MODEL_NAME=glm53-p8-smallm-integrated-v1
PORT=8019
LAYERS=$(seq -s, 3 44)

[ ! -e "$OUT" ] || { echo "refusing to overwrite $OUT" >&2; exit 2; }
git -C "$REPO" merge-base --is-ancestor "$COMMIT" HEAD
git -C "$REPO" diff --quiet "$COMMIT" -- runtime_patch
[ -z "$(git -C "$REPO" status --porcelain)" ]
[ "$(docker image inspect "$IMAGE" --format '{{.Id}}')" = "$IMAGE" ]
[ -d "$MODEL" ] && [ -d "$SIDECARS" ] && [ -f "$DESIGN" ] && [ -f "$BENCH" ]
! docker inspect "$CONTAINER" >/dev/null 2>&1
[ -z "$(ss -H -ltn "( sport = :$PORT )")" ]
mkdir -p "$OUT" "$CACHE"

exec 9>/run/lock/klc/model-stack.lock
flock -w 900 9
timer_was_active=false
backend_was_active=false
systemctl is-active --quiet klc-model-stack.timer && timer_was_active=true
systemctl --user is-active --quiet klc-backend.service && backend_was_active=true
cid=
thermal_pid=
cleaned=false

cleanup_owned() {
  [ "$cleaned" = true ] && return 0
  if [ -n "$cid" ]; then
    docker stop --time 120 "$cid" >/dev/null 2>&1 || true
    docker logs "$cid" >"$OUT/server-final.log" 2>&1 || true
    docker inspect "$cid" >"$OUT/container-final.private.json" 2>/dev/null || true
    [ "$(docker inspect -f '{{.State.Running}}' "$cid" 2>/dev/null || true)" != true ]
    docker rm "$cid" >/dev/null 2>&1 || true
  fi
  cleaned=true
}

restore() {
  rc=$?
  set +e
  [ -z "$thermal_pid" ] || { kill "$thermal_pid" 2>/dev/null; wait "$thermal_pid" 2>/dev/null; }
  cleanup_owned || rc=1
  [ "$backend_was_active" = false ] || systemctl --user start klc-backend.service || rc=1
  [ "$timer_was_active" = false ] || sudo -n systemctl start klc-model-stack.timer || rc=1
  restored_backend=false
  restored_timer=false
  systemctl --user is-active --quiet klc-backend.service && restored_backend=true
  systemctl is-active --quiet klc-model-stack.timer && restored_timer=true
  [ "$restored_backend" = "$backend_was_active" ] || rc=1
  [ "$restored_timer" = "$timer_was_active" ] || rc=1
  python3 - "$OUT" "$PLAN" "$REPO" "$IMAGE" "$COMMIT" "$rc" "$backend_was_active" "$restored_backend" "$timer_was_active" "$restored_timer" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
out, plan, repo = map(Path, sys.argv[1:4])
def sha(path):
    with path.open('rb') as handle: return hashlib.file_digest(handle, 'sha256').hexdigest()
files = {}
for name in ('result.json', 'server-ready.log', 'container.json', 'nvidia-before.xml', 'nvidia-after.xml'):
    path = out / name
    if path.is_file(): files[name] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
payload = {'schema':'glm53-p8-smallm-integrated-execution.v1',
 'finished_at':datetime.now(timezone.utc).isoformat(), 'exit_code':int(sys.argv[6]),
 'plan_sha256':sha(plan), 'runtime_commit':sys.argv[5],
 'image_sha256':sys.argv[4], 'prior_backend_active':sys.argv[7]=='true',
 'restored_backend_active':sys.argv[8]=='true', 'prior_timer_active':sys.argv[9]=='true',
 'restored_timer_active':sys.argv[10]=='true', 'files':files,
 'protected_roles_opened':[], 'ldlq':False}
(out/'execution.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY
  exit "$rc"
}
trap restore EXIT

[ "$timer_was_active" = false ] || sudo -n systemctl stop klc-model-stack.timer
[ "$backend_was_active" = false ] || systemctl --user stop klc-backend.service
deadline=$((SECONDS + 1800))
while true; do
  temps=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
  max_temp=$(printf '%s\n' "$temps" | sort -nr | head -1)
  [ "$max_temp" -le 75 ] && break
  [ "$SECONDS" -lt "$deadline" ] || { echo "thermal start timeout" >&2; exit 1; }
  sleep 15
done
nvidia-smi -q -x >"$OUT/nvidia-before.xml"

docker run -d --name "$CONTAINER" --gpus all --network host --ipc host --shm-size 64g \
  --restart no --entrypoint /bin/bash \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e OMP_NUM_THREADS=2 \
  -e NCCL_IB_DISABLE=1 -e NCCL_P2P_DISABLE=0 -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_PROTO=LL,LL128,Simple -e NCCL_CUMEM_ENABLE=0 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 -e VLLM_PCIE_ALLREDUCE_BACKEND=cpp \
  -e VLLM_CPP_AR_1STAGE_NCCL_CUTOFF=56KB -e VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS=0 \
  -e VLLM_B12X_GLM_NOPE_NVFP4=1 -e VLLM_NVFP4_MLA_DYNAMIC_SCALE=0 \
  -e VLLM_NVFP4_MLA_SCALES_FILE=/opt/glm53/calibration/glm53_nvfp4_mla_outer_scales_mtp_power2_v2.json \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e KV_FP8_ROPE=0 \
  -e PYTHONPATH=/runtime-patch:/opt/exllamav3:/opt/infernal-invocation/vllm:/opt/infernal-invocation/b12x \
  -e GLM53_P8_NATIVE=1 -e GLM53_P8_SMALL_M=1 \
  -e GLM53_P8_NATIVE_SIDECAR_DIR=/p8-sidecars -e GLM53_P8_NATIVE_LAYERS="$LAYERS" \
  -e GLM53_P8_NATIVE_DESIGN=/p8-design/native6-v2.json \
  -v "$MODEL:/model:ro" -v "$CACHE:/cache:rw" -v "$PATCH:/runtime-patch:ro" \
  -v "$SIDECARS:/p8-sidecars:ro" -v "$DESIGN:/p8-design/native6-v2.json:ro" \
  -v /home/brandonmusic/models/GLM-5.3-Flash-NVFP4:/home/brandonmusic/models/GLM-5.3-Flash-NVFP4:ro \
  -v "$CAMPAIGN:$CAMPAIGN:ro" "$IMAGE" -lc \
  "exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve /model --served-model-name $MODEL_NAME --host 0.0.0.0 --port $PORT --language-model-only --tensor-parallel-size 4 --decode-context-parallel-size 1 --dcp-comm-backend a2a --dtype bfloat16 --attention-backend B12X_MLA_SPARSE --kv-cache-dtype nvfp4_ds_mla --max-model-len 131072 --max-num-batched-tokens 2048 --max-num-seqs 1 --gpu-memory-utilization 0.94 --enable-chunked-prefill --no-enable-prefix-caching --generation-config /model --reasoning-parser glm45 --disable-custom-all-reduce --quantization modelopt --load-format instanttensor" >"$OUT/container.cid"
cid=$(cat "$OUT/container.cid")
[[ "$cid" =~ ^[a-f0-9]{64}$ ]]

(
  while docker inspect -f '{{.State.Running}}' "$cid" 2>/dev/null | grep -qx true; do
    temps=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits) || exit 1
    max_temp=$(printf '%s\n' "$temps" | sort -nr | head -1)
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$max_temp" >>"$OUT/thermal-max.log"
    if [ "$max_temp" -ge 90 ]; then
      echo "thermal abort at $max_temp C" >"$OUT/thermal-abort.txt"
      docker stop --time 5 "$cid" >/dev/null 2>&1 || true
      exit 1
    fi
    sleep 2
  done
) &
thermal_pid=$!

deadline=$((SECONDS + 2400))
until curl -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models" >"$OUT/models.json" 2>/dev/null && grep -q "$MODEL_NAME" "$OUT/models.json"; do
  [ "$(docker inspect -f '{{.State.Running}}' "$cid" 2>/dev/null || true)" = true ]
  kill -0 "$thermal_pid"
  [ "$SECONDS" -lt "$deadline" ] || { echo "server readiness timeout" >&2; exit 1; }
  sleep 5
done
docker inspect "$cid" >"$OUT/container.json"
docker logs "$cid" >"$OUT/server-ready.log" 2>&1
grep -q 'Breakable CUDA graph enabled' "$OUT/server-ready.log"
grep -q 'GLM53_P8_NATIVE_PATCH_ACTIVE.*small_m_scheduler=true' "$OUT/server-ready.log"
python3 - "$OUT/server-ready.log" <<'PY'
import re, sys
text=open(sys.argv[1], errors='replace').read()
pairs={(int(a),int(b)) for a,b in re.findall(r'GLM53_P8_NATIVE_FORWARD layer=(\d+) rank=(\d+).*small_m_scheduler=true', text)}
expected={(layer,rank) for layer in range(3,45) for rank in range(4)}
if pairs != expected: raise SystemExit(f'forward inventory mismatch missing={sorted(expected-pairs)[:8]} extra={sorted(pairs-expected)[:8]}')
PY
PYTHONPATH="$REPO" python3 -m glm53_nvfp4.audit_speed_graphs --log "$OUT/server-ready.log" >"$OUT/graph-audit.json"

python3 "$BENCH" --host 127.0.0.1 --port "$PORT" --model "$MODEL_NAME" \
  --concurrency 1 --contexts 32k --duration 20 --max-tokens 4096 \
  --decode-warmup-seconds 3 --standalone-prefill --prefill-contexts 32k \
  --prefill-duration 20 --prefill-metric auto --token-targeting estimate \
  --display-mode plain --no-resume --output "$OUT/result.json" | tee "$OUT/benchmark.log"

[ ! -f "$OUT/thermal-abort.txt" ]
kill -0 "$thermal_pid"
nvidia-smi -q -x >"$OUT/nvidia-after.xml"
cleanup_owned
