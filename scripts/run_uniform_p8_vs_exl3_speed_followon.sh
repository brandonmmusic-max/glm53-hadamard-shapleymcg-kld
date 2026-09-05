#!/usr/bin/env bash
set -euo pipefail

REPO=${GLM53_REPO:-/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53}
RUNTIME_REPO=${GLM53_P8_RUNTIME_REPO:-/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-all42-runtime}
CAMPAIGN=/media/brandonmusic/klcstore/bmxfp4-glm53
ROOT=/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/uniform-p8-all42-v1
KLD_SERVICE=${GLM53_P8_KLD_SERVICE:-glm53-uniform-p8-all42-kld-v1a2.service}
KLD_ANALYSIS=$ROOT/fullmodel-kld-vs-decoded-gptq-context.json
PLAN=$REPO/experiments/p8-uniform-all42-tp4-vs-exl3-speed-v1.json
PLAN_SEAL=$REPO/experiments/p8-uniform-all42-tp4-vs-exl3-speed-v1.sha256
IMAGE_AMENDMENT=$REPO/experiments/p8-all42-runtime-image-amendment-2.json
ANALYZER=$REPO/glm53_nvfp4/analyze_p8_vs_exl3_speed.py
RUNTIME_PATCH=$RUNTIME_REPO/runtime_patch
RUNTIME_MANIFEST=$ROOT/runtime-patch-manifest-efd250b.json
DESIGN=$RUNTIME_REPO/experiments/p8-kld-shapley-native6-v2.json
P8_MODEL=$CAMPAIGN/codec-v2/p8-h128/identity-mcg-layer3-v1/candidate-bf16-pseudoquant
P8_SIDECARS=$ROOT/sidecars
EXL3_MODEL=/home/brandonmusic/models/GLM-5.3-Flash-EXL3-4bpw
EXL3_RECEIPT=/home/brandonmusic/KLC_SANDBOXES/glm-5.3-flash-exl3-4bpw-release/results/materialization-receipt.json
EXL3_COMPOSE=/home/brandonmusic/KLC_SANDBOXES/glm-5.3-flash-exl3-4bpw-release/runtime/compose.sm120-tp4-vision-mtp5.yaml
BENCH_REPO=/home/brandonmusic/KLC_SANDBOXES/glm53-exl3-k4-r10-rebase/tooling/llm-inference-bench
BENCH=$BENCH_REPO/llm_decode_bench.py
OUT=$ROOT/speed-v1
ANALYSIS=$OUT/analysis.json
EXECUTION=$OUT/execution.json
LOG=$OUT/followon.log
CONTAINER=glm53-p8-exl3-speed-v1
PORT=8017
LOCK=/run/lock/klc/model-stack.lock
P8_IMAGE=sha256:5da4ef3e814a71c6bcc47a7eb409a02fe4e3d5d867261f0b8e2e9b2d6ebb8ef8
EXL3_IMAGE=verdictai/glm53-flash-exl3-k4@sha256:d1b6c021df11056cebde469cadc05f55cb21ec4ffc8b54ae6f08161df0c493bf
P8_NAME=glm53-p8-uniform-all42-native
EXL3_NAME=GLM-5.3-Flash-EXL3-4bpw
P8_CACHE=/home/brandonmusic/KLC_SANDBOXES/glm53-exl3-k4-sm120/cache-dflash2-nvfp4-v77
EXL3_CACHE=/home/brandonmusic/KLC_SANDBOXES/glm53-exl3-k4-sm120/cache-mtp5-tp4-v84
LAYERS=$(seq -s, 3 44)

mkdir -p "$OUT"
log() { echo "[$(date --iso-8601=seconds)] $*" | tee -a "$LOG"; }
sha() { sha256sum "$1" | cut -d' ' -f1; }

log "waiting for KLD service=$KLD_SERVICE"
while systemctl --user is-active --quiet "$KLD_SERVICE"; do
  sleep 30
done
[ "$(systemctl --user show "$KLD_SERVICE" -p Result --value)" = success ] || {
  log "KLD service did not finish successfully; speed gate will not start"
  exit 1
}
[ -f "$KLD_ANALYSIS" ] || { log "KLD analysis missing"; exit 1; }
python3 - "$KLD_ANALYSIS" <<'PY'
import json, math, sys
x=json.load(open(sys.argv[1]))
values=[]
def walk(v):
    if isinstance(v, dict):
        for z in v.values(): walk(z)
    elif isinstance(v, list):
        for z in v: walk(z)
    elif isinstance(v, float): values.append(v)
walk(x)
if not values or not all(math.isfinite(v) for v in values):
    raise SystemExit("KLD analysis is not complete and finite")
PY

(cd "$(dirname "$PLAN")" && sha256sum --check --strict "$(basename "$PLAN_SEAL")") | tee -a "$LOG"
(cd "$REPO/experiments" && sha256sum --check --strict p8-all42-runtime-image-amendment-2.sha256) | tee -a "$LOG"
PYTHONPATH="$REPO" python3 -m glm53_nvfp4.preflight_p8_runtime_image --amendment "$IMAGE_AMENDMENT" | tee -a "$LOG"
[ "$P8_IMAGE" = "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["image_id"])' "$IMAGE_AMENDMENT")" ]
PYTHONPATH="$REPO" python3 -m glm53_nvfp4.audit_p8_fullmodel_completion \
  --plan "$REPO/experiments/p8-uniform-all42-fullmodel-kld-v1.json" \
  --roles "$CAMPAIGN/roles/roles-codec-conditional-fit32-v1.json" \
  --manifest "$CAMPAIGN/kld-v3/run-p8-uniform-all42-native-cf32-v1.json" \
  --analysis "$KLD_ANALYSIS" \
  --log "$CAMPAIGN/kld-v3/sessions/p8-uniform-all42-native-cf32-v1/server-final.log" \
  --records "$CAMPAIGN/kld-v3/records/p8-uniform-all42-native-cf32-v1" \
  --output "$ROOT/fullmodel-completion-audit.json" | tee -a "$LOG"
[ "$(sha "$ANALYZER")" = dae589bbf84fcedf221c926957c7ae65684d29012ca436c997eb9478f4f1036f ]
[ "$(git -C "$RUNTIME_REPO" rev-parse HEAD)" = efd250b002e8da5a0e603253e0cd07ba6249c7b4 ]
[ "$(sha "$RUNTIME_MANIFEST")" = e1cf316e8e969625eac6749866a12c298b5ae7fef3dde6ebba37b53ed74dc873 ]
[ "$(git -C "$BENCH_REPO" rev-parse HEAD)" = 42c38fdd0476cf86940268f4e098581e5b61542e ]
[ "$(git -C "$BENCH_REPO" rev-parse HEAD^{tree})" = 86544a01e1e0faeb870cfff7ea751a000c9c6e98 ]
[ -z "$(git -C "$BENCH_REPO" status --porcelain=v1)" ]
[ "$(sha "$BENCH")" = c1d9b66dbf70df23545a71451b5bc2ecaaa746ed2441312e4fb77a8bf476504a ]
[ "$(sha "$EXL3_MODEL/model.safetensors.index.json")" = 2f64d21c67c90bbafeb36c4e9b2f06f54063ed439e9f7cf95962d425a1d8515d ]
[ "$(sha "$EXL3_RECEIPT")" = afe588284702c0676b7af5df48bc0e0568bb42b1821808ab71c6dfa1f0c48b61 ]
[ "$(sha "$EXL3_COMPOSE")" = 5558cb276b964e78571e71ffad82c9830c3a602b559c9472bbe92e1f050860d7 ]
[ "$(docker image inspect "$P8_IMAGE" --format '{{.Id}}')" = "$P8_IMAGE" ]
[ "$(docker image inspect "$EXL3_IMAGE" --format '{{.Id}}')" = sha256:d1b6c021df11056cebde469cadc05f55cb21ec4ffc8b54ae6f08161df0c493bf ]
PYTHONPATH="$REPO" python3 -m glm53_nvfp4.hash_tree --root "$RUNTIME_PATCH" --output "$RUNTIME_MANIFEST" --verify | tee -a "$LOG"

python3 - "$PLAN" "$KLD_ANALYSIS" "$RUNTIME_MANIFEST" "$EXL3_RECEIPT" "$BENCH" "$EXECUTION" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
plan, kld, runtime, exl3, bench, output=map(Path, sys.argv[1:])
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20), b''): h.update(chunk)
    return h.hexdigest()
static={
  'schema':'glm53-p8-uniform-all42-vs-exl3-speed-execution.v1',
  'plan_sha256':sha(plan), 'kld_analysis_sha256':sha(kld),
  'image_amendment_sha256':sha(plan.parent/'p8-all42-runtime-image-amendment-2.json'),
  'p8_image_id':json.loads((plan.parent/'p8-all42-runtime-image-amendment-2.json').read_text())['image_id'],
  'runtime_manifest_sha256':sha(runtime), 'exl3_receipt_sha256':sha(exl3),
  'benchmark_tool_sha256':sha(bench), 'started_at':datetime.now(timezone.utc).isoformat(),
  'protected_roles_opened':[],
}
if output.exists():
    old=json.loads(output.read_text())
    for key in static.keys()-{'started_at'}:
        if old.get(key)!=static[key]: raise SystemExit('execution identity changed')
else:
    output.write_text(json.dumps(static, indent=2, sort_keys=True)+'\n')
PY

exec 9>"$LOCK"
flock -w 900 9
timer_was_active=false
backend_was_active=false
systemctl is-active --quiet klc-model-stack.timer && timer_was_active=true
systemctl --user is-active --quiet klc-backend.service && backend_was_active=true
restore() {
  set +e
  docker logs "$CONTAINER" >"$OUT/server-final.log" 2>&1 || true
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  [ "$backend_was_active" = true ] && systemctl --user start klc-backend.service
  [ "$timer_was_active" = true ] && sudo -n systemctl start klc-model-stack.timer
  log "restored timer=$timer_was_active backend=$backend_was_active"
}
trap restore EXIT
[ "$timer_was_active" = true ] && sudo -n systemctl stop klc-model-stack.timer
[ "$backend_was_active" = true ] && systemctl --user stop klc-backend.service

wait_thermal() {
  local deadline=$((SECONDS + 1800))
  while true; do
    local max_temp
    max_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -nr | head -1)
    if [ "$max_temp" -le 75 ]; then return; fi
    [ "$SECONDS" -lt "$deadline" ] || { log "thermal start timeout max_temp=$max_temp"; return 1; }
    sleep 30
  done
}

run_arm() {
  local round=$1
  local arm=$2
  local slot=$OUT/round-$(printf '%02d' "$round")-$arm
  local result=$slot/benchmark.json
  local server_log=$slot/server.log
  if [ -f "$result" ] && [ -f "$slot/receipt.json" ]; then
    log "round=$round arm=$arm already complete"
    return
  fi
  [ ! -e "$slot" ] || { log "incomplete prior slot preserved at $slot"; return 1; }
  mkdir -p "$slot"
  wait_thermal
  nvidia-smi -q -x >"$slot/nvidia-before.xml"
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

  local image model name cache
  local -a envs mounts args
  envs=(
    -e CUDA_VISIBLE_DEVICES=0,1,2,3 -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e OMP_NUM_THREADS=2
    -e NCCL_IB_DISABLE=1 -e NCCL_P2P_DISABLE=0 -e NCCL_P2P_LEVEL=SYS
    -e NCCL_PROTO=LL,LL128,Simple -e NCCL_CUMEM_ENABLE=0
    -e VLLM_ENABLE_PCIE_ALLREDUCE=1 -e VLLM_PCIE_ALLREDUCE_BACKEND=cpp
    -e VLLM_CPP_AR_1STAGE_NCCL_CUTOFF=56KB -e VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS=0
    -e VLLM_B12X_GLM_NOPE_NVFP4=1 -e VLLM_NVFP4_MLA_DYNAMIC_SCALE=0
    -e VLLM_NVFP4_MLA_SCALES_FILE=/opt/glm53/calibration/glm53_nvfp4_mla_outer_scales_mtp_power2_v2.json
    -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e KV_FP8_ROPE=0
  )
  mounts=()
  args=(
    --host 0.0.0.0 --port "$PORT" --language-model-only --tensor-parallel-size 4
    --decode-context-parallel-size 1 --dcp-comm-backend a2a --dtype bfloat16
    --attention-backend B12X_MLA_SPARSE --kv-cache-dtype nvfp4_ds_mla
    --max-model-len 131072 --max-num-batched-tokens 2048 --max-num-seqs 1
    --gpu-memory-utilization 0.94 --enable-chunked-prefill --no-enable-prefix-caching
    --generation-config /model --reasoning-parser glm45 --disable-custom-all-reduce
  )
  if [ "$arm" = p8 ]; then
    image=$P8_IMAGE; model=$P8_MODEL; name=$P8_NAME; cache=$P8_CACHE
    envs+=(
      -e PYTHONPATH=/runtime-patch:/opt/exllamav3:/opt/infernal-invocation/vllm:/opt/infernal-invocation/b12x
      -e GLM53_P8_NATIVE=1 -e GLM53_P8_NATIVE_SIDECAR_DIR=/p8-sidecars
      -e GLM53_P8_NATIVE_LAYERS="$LAYERS" -e GLM53_P8_NATIVE_DESIGN=/p8-design/native6-v2.json
    )
    mounts+=(
      -v "$RUNTIME_PATCH:/runtime-patch:ro" -v "$P8_SIDECARS:/p8-sidecars:ro"
      -v "$DESIGN:/p8-design/native6-v2.json:ro"
      -v /home/brandonmusic/models/GLM-5.3-Flash-NVFP4:/home/brandonmusic/models/GLM-5.3-Flash-NVFP4:ro
      -v "$CAMPAIGN:$CAMPAIGN:ro"
    )
    args+=(--quantization modelopt --load-format instanttensor)
  else
    image=$EXL3_IMAGE; model=$EXL3_MODEL; name=$EXL3_NAME; cache=$EXL3_CACHE
    envs+=(
      -e VLLM_EXL3_PREFILL_BLOCK_M=128 -e VLLM_EXL3_PREFILL_TRELLIS=1
      -e B12X_GL53_ROUTE128_WIDE=1 -e B12X_GL53_ROUTE128_HYBRID_TAIL=1
    )
    args+=(--quantization exl3 --load-format safetensors --moe-backend b12x)
  fi
  mkdir -p "$cache"
  docker run -d --name "$CONTAINER" --gpus all --network host --ipc host --shm-size 64g \
    --restart no --entrypoint /bin/bash "${envs[@]}" \
    -v "$model:/model:ro" -v "$cache:/cache:rw" "${mounts[@]}" "$image" \
    -lc "exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve /model --served-model-name $name ${args[*]}" >/dev/null

  local deadline=$((SECONDS + 2400))
  until curl -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models" >"$slot/models.json" 2>/dev/null && grep -q "$name" "$slot/models.json"; do
    [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || true)" = true ] || {
      docker logs "$CONTAINER" >"$server_log" 2>&1 || true
      log "round=$round arm=$arm server exited"
      return 1
    }
    [ "$SECONDS" -lt "$deadline" ] || { log "round=$round arm=$arm readiness timeout"; return 1; }
    sleep 5
  done
  docker inspect "$CONTAINER" >"$slot/container.json"
  docker image inspect "$image" >"$slot/image.json"
  docker logs "$CONTAINER" >"$server_log" 2>&1
  grep -q 'tensor_parallel_size=4' "$server_log"
  grep -q 'decode_context_parallel_size=1' "$server_log"
  grep -q 'speculative_config=None' "$server_log"
  grep -q 'enforce_eager=False' "$server_log"
  grep -q 'kv_cache_dtype=nvfp4_ds_mla' "$server_log"
  grep -q 'Breakable CUDA graph enabled' "$server_log"
  if [ "$arm" = p8 ]; then
    grep -q "GLM53_P8_NATIVE_PATCH_ACTIVE layers=$LAYERS tp=4 .*physical_bpw=4.25 ldlq=false" "$server_log"
  else
    grep -q 'quantization=exl3' "$server_log"
    grep -q 'GLM-5.3 routed-only EXL3: streaming unsliced K4 experts' "$server_log"
  fi

  python3 "$BENCH" --host 127.0.0.1 --port "$PORT" --model "$name" \
    --concurrency 1 --contexts 32k,64k --duration 20 --max-tokens 4096 \
    --decode-warmup-seconds 3 --standalone-prefill --prefill-contexts 32k,64k \
    --prefill-duration 20 --prefill-metric auto --token-targeting estimate \
    --display-mode plain --no-resume --output "$result" 2>&1 | tee "$slot/benchmark.log"

  docker logs "$CONTAINER" >"$server_log" 2>&1
  grep -q 'Capturing decode CUDA graphs (FULL)' "$server_log"
  if [ "$arm" = p8 ]; then
    python3 - "$server_log" <<'PY'
import re, sys
text=open(sys.argv[1], errors='replace').read()
pairs={(int(a),int(b)) for a,b in re.findall(r'GLM53_P8_NATIVE_FORWARD layer=(\d+) rank=(\d+)', text)}
expected={(layer,rank) for layer in range(3,45) for rank in range(4)}
if pairs != expected:
    raise SystemExit(f'P8 forward inventory mismatch missing={sorted(expected-pairs)[:8]} extra={sorted(pairs-expected)[:8]}')
PY
  fi
  nvidia-smi -q -x >"$slot/nvidia-after.xml"
  python3 - "$PLAN" "$result" "$server_log" "$slot/container.json" "$slot/nvidia-before.xml" "$slot/nvidia-after.xml" "$slot/receipt.json" "$round" "$arm" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
plan, result, log, container, before, after, output=map(Path, sys.argv[1:8])
round_no=int(sys.argv[8]); arm=sys.argv[9]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20), b''): h.update(chunk)
    return h.hexdigest()
x=json.loads(result.read_text())
max_temp=max(float(v.get('hardware_summary',{}).get('temp_max_c',0)) for v in x.get('prefill',{}).values())
for row in x.get('results',[]):
    max_temp=max(max_temp,float(row.get('hardware_summary',{}).get('temp_max_c',0)))
if max_temp >= 94:
    raise SystemExit(f'thermal abort threshold reached: {max_temp}')
payload={'schema':'glm53-p8-vs-exl3-cold-run-receipt.v1','round':round_no,'arm':arm,
 'completed_at':datetime.now(timezone.utc).isoformat(),'max_measured_gpu_temp_c':max_temp,
 'files':{p.name:{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in (plan,result,log,container,before,after)},
 'protected_roles_opened':[]}
output.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY
  docker rm -f "$CONTAINER" >/dev/null
  log "round=$round arm=$arm complete"
}

orders=("exl3 p8" "p8 exl3" "exl3 p8" "p8 exl3" "exl3 p8")
for round in 1 2 3 4 5; do
  for arm in ${orders[$((round-1))]}; do
    run_arm "$round" "$arm"
  done
done

mapfile -t p8_runs < <(find "$OUT" -maxdepth 2 -path '*/round-*-p8/benchmark.json' | sort)
mapfile -t exl3_runs < <(find "$OUT" -maxdepth 2 -path '*/round-*-exl3/benchmark.json' | sort)
PYTHONPATH="$REPO" python3 -m glm53_nvfp4.analyze_p8_vs_exl3_speed \
  --plan "$PLAN" --p8 "${p8_runs[@]}" --exl3 "${exl3_runs[@]}" --output "$ANALYSIS" | tee -a "$LOG"
log "speed gate complete analysis=$ANALYSIS"
