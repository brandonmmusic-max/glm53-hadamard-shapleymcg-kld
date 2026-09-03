#!/usr/bin/env bash
set -euo pipefail

# Scale the passing Qwen-style layer-3 arm to every routed layer.  The BF16
# calibration capture is streamed one layer at a time from the pinned teacher
# dataset, quantized on four GPUs, verified, and removed after use.

REPO=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53
CAMPAIGN=/media/brandonmusic/klcstore/bmxfp4-glm53
SOURCE=$CAMPAIGN/downloads/GLM-5.3-Flash-BF16
SOURCE_INDEX=$CAMPAIGN/downloads/source-metadata/model.safetensors.index.json
CAPTURE=$CAMPAIGN/teacher/calibration/main-ep4-full
ROLES=$CAMPAIGN/roles/roles-v3.json
ROOT=${GLM53_EXACT_FULL_ROOT:-/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-v9-large/had16-all}
LAYERS_ROOT=$ROOT/layers
CANDIDATE=$ROOT/candidate
LOG=$ROOT/full-build.log
REUSE_LAYER3=$CAMPAIGN/exact-v10/layer-003/had16
LOCK=/run/lock/klc/model-stack.lock
PRODUCTION=glm53-flash-exl3-k4-tp4-vision-mtp3

mkdir -p "$LAYERS_ROOT" "$ROOT/evidence"
cd "$REPO"

available_root=$(df -B1 --output=avail / | tail -1 | tr -d ' ')
available_media=$(df -B1 --output=avail /media/brandonmusic/klcstore | tail -1 | tr -d ' ')
completed_receipts=$(find "$LAYERS_ROOT" -path '*/evidence/layer-validation.json' -type f 2>/dev/null | wc -l)
[ "$completed_receipts" -le 41 ] || {
  echo "unexpected number of completed layer receipts: $completed_receipts" >&2
  exit 1
}
remaining_layers=$((41 - completed_receipts))
# A resumed build must not demand the original whole-campaign free space after
# already-validated layer payloads have consumed it. Reserve 4.5 GB per
# unfinished routed layer plus 20 GB for streamed capture and filesystem headroom.
required_root=$((20000000000 + remaining_layers * 4500000000))
[ "$available_root" -ge "$required_root" ] || {
  echo "insufficient root space: need $required_root bytes for $remaining_layers remaining layers" >&2
  exit 1
}
[ "$available_media" -ge 15000000000 ] || {
  echo "less than 15 GB free for one streamed calibration layer" >&2
  exit 1
}
for required in "$SOURCE_INDEX" "$CAPTURE/capture-manifest.json" "$ROLES"; do
  [ -f "$required" ] || { echo "missing required input: $required" >&2; exit 1; }
done

layer_complete() {
  local layer=$1 l3 root validation
  printf -v l3 '%03d' "$layer"
  root=$LAYERS_ROOT/layer-$l3/had16
  validation=$root/evidence/layer-validation.json
  [ -f "$validation" ] || return 1
  python3 - "$validation" "$root" <<'PY'
import json, sys
from pathlib import Path
value=json.load(open(sys.argv[1])); root=Path(sys.argv[2]); layer=int(value["layer"])
if value.get("status") != "pass" or value.get("experts") != 288 or set(value.get("projections", {})) != {"down", "gate", "up"}:
    raise SystemExit(1)
chunks=list((root/"chunks").glob(f"had16-layer-{layer:03d}-experts-*.safetensors"))
if len(chunks) != 4 or not (root/"chunks"/f"had16-layer-{layer:03d}-input-scales.safetensors").is_file():
    raise SystemExit(1)
PY
}

quantize_layer() {
  local layer=$1 l3 root chunks evidence logs gpu start end s3 e3 pid failed
  printf -v l3 '%03d' "$layer"
  root=$LAYERS_ROOT/layer-$l3/had16
  chunks=$root/chunks
  evidence=$root/evidence
  logs=$root/logs
  mkdir -p "$chunks" "$evidence" "$logs"
  pids=()
  for gpu in 0 1 2 3; do
    start=$((gpu * 72)); end=$((start + 72))
    printf -v s3 '%03d' "$start"; printf -v e3 '%03d' "$end"
    CUDA_VISIBLE_DEVICES=$gpu /home/brandonmusic/klc-env/bin/python -m glm53_nvfp4.quantize_layer \
      --source "$SOURCE" --source-index "$SOURCE_INDEX" --capture-root "$CAPTURE" --roles "$ROLES" \
      --output "$chunks/had16-layer-$l3-experts-$s3-$e3.safetensors" \
      --receipt "$evidence/had16-layer-$l3-experts-$s3-$e3.json" \
      --layer "$layer" --expert-start "$start" --expert-end "$end" --device cuda:0 \
      --max-samples 256 --search-grid 12 --rotation had16 --rotation-scope all \
      --gptq-geometry full --projections all >"$logs/quant-gpu-$gpu.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  [ "$failed" -eq 0 ] || { echo "layer $layer quantization failed; see $logs" >&2; return 1; }

  receipts=()
  for gpu in 0 1 2 3; do
    start=$((gpu * 72)); end=$((start + 72))
    printf -v s3 '%03d' "$start"; printf -v e3 '%03d' "$end"
    receipts+=(--receipt "$evidence/had16-layer-$l3-experts-$s3-$e3.json")
  done
  /usr/bin/python3 -m glm53_nvfp4.validate_layer --layer "$layer" --projections all \
    "${receipts[@]}" --output "$evidence/layer-validation.json" >"$logs/layer-validation.log"

  CUDA_VISIBLE_DEVICES=0 /home/brandonmusic/klc-env/bin/python -m glm53_nvfp4.calibrate_input_scale \
    --capture-root "$CAPTURE" --source "$SOURCE" --source-index "$SOURCE_INDEX" \
    --roles "$ROLES" --layer "$layer" --rotation had16 --rotation-scope all \
    --device cuda:0 --max-samples 256 \
    --output "$chunks/had16-layer-$l3-input-scales.safetensors" \
    --receipt "$evidence/input-scales.json" >"$logs/input-scales.log" 2>&1
}

remove_capture() {
  local layer=$1 l3 path
  printf -v l3 '%03d' "$layer"
  for path in hidden.bf16.bin topk_ids.u16le.bin topk_weights.f32le.bin; do
    [ ! -f "$CAPTURE/layers/layer-$l3/$path" ] || unlink -- "$CAPTURE/layers/layer-$l3/$path"
  done
}

prefetch_layer() {
  local layer=$1 l3
  printf -v l3 '%03d' "$layer"
  /usr/bin/python3 -m glm53_nvfp4.prefetch_partial_capture \
    --capture-root "$CAPTURE" --roles "$ROLES" --layer "$layer" \
    --role fit --workers 8 \
    --receipt "$CAPTURE/layers/layer-$l3/partial-fit-capture.json"
}

# The winning layer-3 payload is immutable and already verified.  Reuse it
# rather than spending another four GB and another quantization pass.
for required in \
  "$REUSE_LAYER3/evidence/layer-validation.json" \
  "$REUSE_LAYER3/chunks/had16-layer-003-input-scales.safetensors"; do
  [ -f "$required" ] || { echo "missing passing layer-3 artifact: $required" >&2; exit 1; }
done
remove_capture 3

exec 9>"$LOCK"
flock -w 900 9
timer_was_active=false
backend_was_active=false
production_was_running=false
systemctl is-active --quiet klc-model-stack.timer && timer_was_active=true
systemctl --user is-active --quiet klc-backend.service && backend_was_active=true
[ "$(docker inspect -f '{{.State.Running}}' "$PRODUCTION" 2>/dev/null || true)" = true ] && production_was_running=true
restore() {
  set +e
  if [ "$production_was_running" = true ] && [ "$(docker inspect -f '{{.State.Running}}' "$PRODUCTION" 2>/dev/null || true)" != true ]; then
    docker start "$PRODUCTION" >/dev/null
  fi
  [ "$backend_was_active" = false ] || systemctl --user start klc-backend.service
  [ "$timer_was_active" = false ] || sudo -n systemctl start klc-model-stack.timer
}
trap restore EXIT
[ "$timer_was_active" = false ] || sudo -n systemctl stop klc-model-stack.timer
[ "$backend_was_active" = false ] || systemctl --user stop klc-backend.service
[ "$production_was_running" = false ] || docker stop --time 120 "$PRODUCTION" >/dev/null

prefetch_pid=
prefetch_layer=
if ! layer_complete 4; then
  prefetch_layer 4 >>"$LOG" 2>&1 &
  prefetch_pid=$!
  prefetch_layer=4
fi

for layer in $(seq 4 44); do
  if layer_complete "$layer"; then
    echo "layer $layer already complete" | tee -a "$LOG"
    remove_capture "$layer"
    continue
  fi
  if [ "$prefetch_layer" != "$layer" ]; then
    prefetch_layer "$layer" >>"$LOG" 2>&1 &
    prefetch_pid=$!
    prefetch_layer=$layer
  fi
  wait "$prefetch_pid"
  prefetch_pid=
  prefetch_layer=

  next=$((layer + 1))
  if [ "$next" -le 44 ] && ! layer_complete "$next"; then
    prefetch_layer "$next" >>"$LOG" 2>&1 &
    prefetch_pid=$!
    prefetch_layer=$next
  fi

  echo "quantizing layer $layer" | tee -a "$LOG"
  quantize_layer "$layer"
  remove_capture "$layer"
  echo "completed layer $layer" | tee -a "$LOG"
done
[ -z "$prefetch_pid" ] || wait "$prefetch_pid"

candidate_args=()
for layer in $(seq 3 44); do
  printf -v l3 '%03d' "$layer"
  if [ "$layer" -eq 3 ]; then base=$REUSE_LAYER3; else base=$LAYERS_ROOT/layer-$l3/had16; fi
  for first in 0 72 144 216; do
    last=$((first + 72)); printf -v s3 '%03d' "$first"; printf -v e3 '%03d' "$last"
    path=$base/chunks/had16-layer-$l3-experts-$s3-$e3.safetensors
    [ -f "$path" ] || { echo "missing full-candidate chunk: $path" >&2; exit 1; }
    candidate_args+=(--chunk "$path")
  done
  path=$base/chunks/had16-layer-$l3-input-scales.safetensors
  [ -f "$path" ] || { echo "missing full-candidate scale chunk: $path" >&2; exit 1; }
  candidate_args+=(--chunk "$path")
done

if [ ! -f "$CANDIDATE/OVERLAY.json" ]; then
  /usr/bin/python3 -m glm53_nvfp4.candidate \
    --carrier /home/brandonmusic/models/GLM-5.3-Flash-NVFP4 \
    --output "$CANDIDATE" "${candidate_args[@]}" >"$ROOT/evidence/full-overlay.json"
fi
python3 - "$ROOT" "$CANDIDATE" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, candidate = map(Path, sys.argv[1:])
overlay=json.loads((candidate/"OVERLAY.json").read_text())
if overlay.get("required_load_format") != "instanttensor":
    raise SystemExit("full candidate is not bound to instanttensor")
validations=[]
for layer in range(3,45):
    p=(Path("/media/brandonmusic/klcstore/bmxfp4-glm53/exact-v10/layer-003/had16/evidence/layer-validation.json")
       if layer == 3 else root/"layers"/f"layer-{layer:03d}"/"had16"/"evidence"/"layer-validation.json")
    v=json.loads(p.read_text())
    if v.get("status") != "pass": raise SystemExit(f"failed validation: {p}")
    validations.append({"layer":layer,"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
payload={
    "schema":"glm53-nvfp4-v9.qwen-exact-had16-full-build.v1",
    "status":"pass",
    "layers":list(range(3,45)),
    "rotation":"had16",
    "rotation_scope":"all",
    "gptq_geometry":"full",
    "projections":["gate_proj","up_proj","down_proj"],
    "runtime_load_format":"instanttensor",
    "calibration_max_samples_per_expert":256,
    "layer3_reused_from":"/media/brandonmusic/klcstore/bmxfp4-glm53/exact-v10/layer-003/had16",
    "candidate":str(candidate),
    "candidate_index_sha256":hashlib.sha256((candidate/"model.safetensors.index.json").read_bytes()).hexdigest(),
    "overlay_sha256":hashlib.sha256((candidate/"OVERLAY.json").read_bytes()).hexdigest(),
    "validations":validations,
}
(root/"evidence"/"full-build.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps({"status":"pass","candidate":str(candidate),"layers":42},sort_keys=True))
PY
