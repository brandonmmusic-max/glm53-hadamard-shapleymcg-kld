#!/usr/bin/env bash
# Decode/prefill speed of the upgrades-only coupled P8 checkpoint (17 layers K5, 25 K4)
# on the NVFP4 DS-MLA cache and the B12X sparse MLA backend, TP4.
#
# Serving matches the quality arm exactly except for the capture environment: same image, same
# sidecars, same designs, same boundary, NVFP4 KV, TP4/DCP1, no EP, no MTP, CUDA graphs on.
# The decode-capture hooks are OFF here; a capture-enabled server is never speed-valid.
#
# Regimes: prefill sweep to 128k, then decode at C4 across the context ladder, then C8 and C16.
set -euo pipefail

REPO=${GLM53_COUPLED_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CAMPAIGN=${GLM53_CAMPAIGN_ROOT:-/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6}
CKPT=${GLM53_SPEED_CHECKPOINT:-$CAMPAIGN/group-probes-v1/k5-only}
BENCH=${GLM53_BENCH_REPO:-$CAMPAIGN/llm-inference-bench-latest}
OUT=${GLM53_SPEED_OUT:-$CAMPAIGN/speed-k5-only-nvfp4-tp4-v1}
MODEL=${GLM53_MODEL_ROOT:-/home/brandonmusic/models/GLM-5.3-Flash-NVFP4}
RECIPE=$CAMPAIGN/p8-smallm-scheduler-v1/fc1-integrated-v1/container-final.private.json
IMAGE=${GLM53_SPEED_IMAGE:-sha256:c121590d3371b406c1e456076b1fc9cf9b596aa425b4401c692d142cbc425cd4}
DESIGN_V4=$REPO/results/P8_COUPLED_SCALE_FLASH_EVIDENCE_PREPARATION_V4.json
DESIGN_V3=$REPO/results/P8_COUPLED_SCALE_FLASH_EVIDENCE_PREPARATION_V3.json
DESIGN_K5=$REPO/results/P8_COUPLED_RATE_DESIGN_K5_V1.json
TRANSFORM=$REPO/experiments/p8-coupled-transform-draw0-silu10-v1.json
NAME=glm53-p8-k5only-speed-v1
PORT=${GLM53_SPEED_PORT:-8033}
LOCK=/run/lock/klc/model-stack.lock
MAXLEN=${GLM53_SPEED_MAXLEN:-131072}
PREFILL_CONTEXTS=${GLM53_PREFILL_CONTEXTS:-8k,16k,32k,64k,128k}
DECODE_CONTEXTS=${GLM53_DECODE_CONTEXTS:-0,8k,32k,64k,128k}
DURATION=${GLM53_SPEED_DURATION:-30}
PY=/usr/bin/python3

mkdir -p "$OUT"
cd "$REPO"
export PYTHONPATH="$REPO"
LOG=$OUT/speed.log
log() { echo "[$(date --iso-8601=seconds)] $*" | tee -a "$LOG"; }
sha() { sha256sum "$1" | cut -d' ' -f1; }

command -v docker >/dev/null || { echo "docker missing" >&2; exit 2; }
[ -f "$CKPT/manifest.json" ] || { echo "no checkpoint manifest at $CKPT" >&2; exit 2; }
[ -f "$BENCH/llm_decode_bench.py" ] || { echo "no bench at $BENCH" >&2; exit 2; }

# Production must be off and the GPUs idle; the campaign lock serializes GPU work.
"$PY" -m glm53_nvfp4.full_coupled_build_support guard --root "$OUT" --campaign-path "$OUT" \
  --max-new-bytes "${GLM53_MAX_NEW_BYTES:-400000000000}" --need-bytes 1000000000 \
  --gpus 0,1,2,3 >>"$OUT/guard.jsonl"

exec 9>"$LOCK"
flock -w 7200 9

BENCH_SHA=$(sha "$BENCH/llm_decode_bench.py")
BENCH_COMMIT=$(git -C "$BENCH" rev-parse HEAD)
BENCH_VERSION=$(grep -m1 '^VERSION' "$BENCH/llm_decode_bench.py" | cut -d'"' -f2)
log "speed run: checkpoint=$CKPT image=$IMAGE bench=$BENCH_VERSION@${BENCH_COMMIT:0:12} sha=${BENCH_SHA:0:16}"
"$PY" - "$CKPT/manifest.json" <<'PY' | tee -a "$LOG"
import json, sys
m = json.load(open(sys.argv[1]))
p = m["payload"]
print(json.dumps({"counts": m["allocation"]["counts"], "file_bytes": p["file_bytes"],
                  "stored_bpw": p["stored_bpw_including_metadata"]}))
PY

cleanup() {
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  [ -n "${monitor_pid:-}" ] && kill "$monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT

layers=$(seq -s, 3 44)
serve="exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve /model \
--served-model-name $NAME --host 0.0.0.0 --port $PORT --language-model-only \
--tensor-parallel-size 4 --decode-context-parallel-size 1 --dcp-comm-backend a2a \
--dtype bfloat16 --attention-backend B12X_MLA_SPARSE --kv-cache-dtype nvfp4_ds_mla \
--max-model-len $MAXLEN --max-num-batched-tokens 8192 --max-num-seqs 32 \
--gpu-memory-utilization 0.94 --enable-chunked-prefill --no-enable-prefix-caching \
--generation-config /model --reasoning-parser glm45 --disable-custom-all-reduce \
--quantization modelopt --load-format instanttensor"

log "starting the server (capture hooks off; speed-valid)"
docker run -d --name "$NAME" --network host --ipc host --shm-size 1g --gpus all \
  --runtime nvidia --restart no --security-opt label=disable --workdir / \
  -e PYTHONPATH=/opt/p8-coupled-runtime -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e GLM53_P8_NATIVE=1 -e GLM53_P8_SMALL_M=1 -e GLM53_P8_FC1_TILE_N=128 \
  -e GLM53_P8_FUSED_SCRATCH= -e GLM53_P8_NATIVE_LAYERS="$layers" \
  -e GLM53_P8_NATIVE_SIDECAR_DIR=/p8-sidecars \
  -e GLM53_P8_NATIVE_DESIGN=/p8-design/design-0.json:/p8-design/design-1.json:/p8-design/design-2.json \
  -e GLM53_P8_NATIVE_TRANSFORM=/p8-design/transform.json \
  -e VLLM_EXL3_PREFILL_TRELLIS=1 -e VLLM_EXL3_EXT_PATH=/opt/exllamav3 \
  -e VLLM_EXL3_PREFILL_BLOCK_M=128 \
  -v "$MODEL:/model:ro" -v "$CKPT/sidecars:/p8-sidecars:ro" \
  -v "$DESIGN_V4:/p8-design/design-0.json:ro" -v "$DESIGN_V3:/p8-design/design-1.json:ro" \
  -v "$DESIGN_K5:/p8-design/design-2.json:ro" -v "$TRANSFORM:/p8-design/transform.json:ro" \
  --entrypoint /bin/bash "$IMAGE" -lc "$serve" >"$OUT/container.cid"
log "container $(cut -c1-12 "$OUT/container.cid")"

for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then ready=1; break; fi
  sleep 10
done
[ "${ready:-0}" = 1 ] || { docker logs "$NAME" >"$OUT/server-failed.log" 2>&1; log "server never became ready"; exit 1; }
docker logs "$NAME" >"$OUT/server-ready.log" 2>&1
grep -c "GLM53_P8_NATIVE_READY" "$OUT/server-ready.log" | xargs -I{} log "P8 ready lines: {}"
grep -o "stream=K[0-9]" "$OUT/server-ready.log" | sort | uniq -c | tee -a "$LOG"

(
  while true; do
    stamp=$(date --iso-8601=seconds)
    nvidia-smi --query-gpu=index,temperature.gpu,utilization.gpu,memory.used,power.draw,clocks.current.graphics \
      --format=csv,noheader,nounits | sed "s/^/$stamp,/"
    sleep 2
  done
) >"$OUT/gpu-telemetry.csv" &
monitor_pid=$!

log "regime 1: prefill sweep to 128k"
"$PY" "$BENCH/llm_decode_bench.py" --port "$PORT" --model "$NAME" --prefill-only \
  --prefill-contexts "$PREFILL_CONTEXTS" --display-mode plain --no-resume \
  --output "$OUT/prefill.json" 2>&1 | tee "$OUT/prefill.log"

log "regime 2: decode at C4 across the context ladder"
"$PY" "$BENCH/llm_decode_bench.py" --port "$PORT" --model "$NAME" --skip-prefill \
  --contexts "$DECODE_CONTEXTS" --concurrency 4 --duration "$DURATION" --max-tokens 2048 \
  --decode-warmup-seconds 5 --display-mode plain --no-resume \
  --output "$OUT/decode-c4.json" 2>&1 | tee "$OUT/decode-c4.log"

log "regime 3: decode at C8 and C16 across the context ladder"
"$PY" "$BENCH/llm_decode_bench.py" --port "$PORT" --model "$NAME" --skip-prefill \
  --contexts "$DECODE_CONTEXTS" --concurrency 8,16 --duration "$DURATION" --max-tokens 2048 \
  --decode-warmup-seconds 5 --display-mode plain --no-resume \
  --output "$OUT/decode-c8-c16.json" 2>&1 | tee "$OUT/decode-c8-c16.log"

docker logs "$NAME" >"$OUT/server-final.log" 2>&1
"$PY" - "$OUT" "$CKPT/manifest.json" "$IMAGE" "$BENCH_SHA" "$BENCH_COMMIT" "$BENCH_VERSION" <<'PY'
import hashlib, json, pathlib, sys
out = pathlib.Path(sys.argv[1])
manifest = json.load(open(sys.argv[2]))
record = {"schema": "glm53.p8-k5only-speed.v1", "image_id": sys.argv[3],
          "bench": {"sha256": sys.argv[4], "commit": sys.argv[5], "version": sys.argv[6]},
          "checkpoint": {"counts": manifest["allocation"]["counts"],
                         "file_bytes": manifest["payload"]["file_bytes"],
                         "stored_bpw": manifest["payload"]["stored_bpw_including_metadata"]},
          "serving": {"tp": 4, "dcp": 1, "ep": False, "mtp": False, "graphs": True,
                      "attention": "B12X_MLA_SPARSE", "kv_dtype": "nvfp4_ds_mla",
                      "max_model_len": 131072, "capture_hooks": False},
          "regimes": {}}
for name in ("prefill", "decode-c4", "decode-c8-c16"):
    path = out / f"{name}.json"
    if path.is_file():
        record["regimes"][name] = {"path": path.name, "bytes": path.stat().st_size,
                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
(out / "speed-receipt.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
print(json.dumps({"receipt": str(out / "speed-receipt.json"), "regimes": list(record["regimes"])}))
PY
log "speed run complete out=$OUT"
