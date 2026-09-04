#!/usr/bin/env bash
set -euo pipefail

REPO=${GLM53_ENCODER_REPO:-/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53}
CAMPAIGN=${GLM53_CAMPAIGN_ROOT:-/media/brandonmusic/klcstore/bmxfp4-glm53}
ROOT=${GLM53_UNIFORM_P8_ROOT:-/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/uniform-p8-all42-v1}
SOURCE=${GLM53_BF16_SOURCE:-$CAMPAIGN/downloads/GLM-5.3-Flash-BF16}
INDEX=$SOURCE/model.safetensors.index.json
CAPTURE=${GLM53_CAPTURE_ROOT:-$CAMPAIGN/teacher/calibration/main-ep4-full}
ROLES=${GLM53_FIT_ROLES:-$CAMPAIGN/roles/roles-v3.json}
DESIGN=${GLM53_NATIVE6_DESIGN:-$REPO/experiments/p8-kld-shapley-native6-v2.json}
PYTHON_BIN=${GLM53_ENCODER_PYTHON:-/home/brandonmusic/klc-env/bin/python}
SIDECARS=$ROOT/sidecars
RECEIPTS=$ROOT/receipts
WORK=$ROOT/work
LOG=$ROOT/build.log
THERMAL=$ROOT/thermal-events.jsonl
LOCK=/run/lock/klc/model-stack.lock
REUSE20=${GLM53_P8_LAYER20_SIDECARS:-/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8/layer20-sidecars}

mkdir -p "$SIDECARS" "$RECEIPTS" "$WORK"
cd "$REPO"

log() { echo "[$(date --iso-8601=seconds)] $*" | tee -a "$LOG"; }

layer_complete() {
  local layer=$1 l3
  printf -v l3 '%03d' "$layer"
  PYTHONPATH="$REPO" "$PYTHON_BIN" - "$DESIGN" "$SIDECARS" "$RECEIPTS/layer-$l3.json" "$layer" <<'PY' >/dev/null
import hashlib, json, sys
from pathlib import Path
from safetensors import safe_open
design, sidecars, receipt, layer = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8<<20),b""): h.update(chunk)
    return h.hexdigest()
if not receipt.is_file(): raise SystemExit(1)
want = sha(design)
row = json.loads(receipt.read_text())
if row.get("layer") != layer or len(row.get("ranks", [])) != 4: raise SystemExit(1)
for rank in range(4):
    path = sidecars / f"p8-layer-{layer:03d}-tp4-rank-{rank}.safetensors"
    if not path.is_file(): raise SystemExit(1)
    with safe_open(path, framework="pt", device="cpu") as src:
        meta = src.metadata() or {}
        if meta.get("schema") != "glm53-p8-mcg-tp4-rank.v2" or meta.get("source_design_sha256") != want: raise SystemExit(1)
    rr = next((x for x in row["ranks"] if x["rank"] == rank), None)
    if rr is None or rr.get("payload_bpw") != 4.25: raise SystemExit(1)
    digest = sha(path)
    if rr.get("sha256") != digest: raise SystemExit(1)
PY
}

prefetch_layer() {
  local layer=$1 l3
  printf -v l3 '%03d' "$layer"
  PYTHONPATH="$REPO" /usr/bin/python3 -m glm53_nvfp4.prefetch_partial_capture \
    --capture-root "$CAPTURE" --roles "$ROLES" --layer "$layer" \
    --role fit --workers 8 \
    --receipt "$CAPTURE/layers/layer-$l3/partial-fit-capture.json"
}

remove_capture() {
  local layer=$1 l3 name path
  printf -v l3 '%03d' "$layer"
  for name in hidden.bf16.bin topk_ids.u16le.bin topk_weights.f32le.bin; do
    path=$CAPTURE/layers/layer-$l3/$name
    [ ! -f "$path" ] || unlink -- "$path"
  done
}

encode_layer() {
  local layer=$1 l3 layer_root slot gpu start end s3 e3 pid status temp stopped
  printf -v l3 '%03d' "$layer"
  layer_root=$WORK/layer-$l3
  mkdir -p "$layer_root/codec" "$layer_root/receipts" "$layer_root/logs"
  local -a pids=() gpus=() stopped_flags=(0 0 0 0)
  for slot in 0 1 2 3; do
    gpu=$slot; start=$((slot * 72)); end=$((start + 72))
    printf -v s3 '%03d' "$start"; printf -v e3 '%03d' "$end"
    if [ -f "$layer_root/codec/hessian-trellis-layer-$l3-experts-$s3-$e3.safetensors" ] && \
       [ -f "$layer_root/receipts/hessian-trellis-layer-$l3-experts-$s3-$e3.json" ]; then
      pids+=(0); gpus+=("$gpu"); continue
    fi
    if [ -e "$layer_root/codec/hessian-trellis-layer-$l3-experts-$s3-$e3.safetensors" ] || \
       [ -e "$layer_root/receipts/hessian-trellis-layer-$l3-experts-$s3-$e3.json" ]; then
      log "layer=$layer inconsistent partial chunk=$s3:$e3; refusing overwrite"
      return 1
    fi
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$REPO" "$PYTHON_BIN" -m glm53_nvfp4.quantize_hessian_trellis_layer \
      --source "$SOURCE" --source-index "$INDEX" --capture-root "$CAPTURE" \
      --roles "$ROLES" --design "$DESIGN" --layer "$layer" \
      --expert-start "$start" --expert-end "$end" --samples 256 --device cuda:0 \
      --codec-output "$layer_root/codec/hessian-trellis-layer-$l3-experts-$s3-$e3.safetensors" \
      --receipt "$layer_root/receipts/hessian-trellis-layer-$l3-experts-$s3-$e3.json" \
      >"$layer_root/logs/chunk-$s3-$e3.log" 2>&1 &
    pids+=("$!"); gpus+=("$gpu")
  done

  while :; do
    local alive=0
    for slot in 0 1 2 3; do
      pid=${pids[$slot]}
      [ "$pid" -ne 0 ] && kill -0 "$pid" 2>/dev/null || continue
      alive=1
      gpu=${gpus[$slot]}
      temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
      if [ "${stopped_flags[$slot]}" -eq 0 ] && [ "$temp" -ge 89 ]; then
        kill -STOP "$pid"
        stopped_flags[$slot]=1
        printf '{"at":"%s","event":"pause","layer":%d,"gpu":%d,"pid":%d,"temperature_c":%d}\n' \
          "$(date --iso-8601=seconds)" "$layer" "$gpu" "$pid" "$temp" >>"$THERMAL"
      elif [ "${stopped_flags[$slot]}" -eq 1 ] && [ "$temp" -le 78 ]; then
        kill -CONT "$pid"
        stopped_flags[$slot]=0
        printf '{"at":"%s","event":"resume","layer":%d,"gpu":%d,"pid":%d,"temperature_c":%d}\n' \
          "$(date --iso-8601=seconds)" "$layer" "$gpu" "$pid" "$temp" >>"$THERMAL"
      fi
    done
    [ "$alive" -eq 1 ] || break
    sleep 5
  done
  status=0
  for slot in 0 1 2 3; do
    pid=${pids[$slot]}
    [ "$pid" -eq 0 ] || wait "$pid" || status=1
  done
  [ "$status" -eq 0 ] || return 1

  local -a chunk_args=()
  for start in 0 72 144 216; do
    end=$((start + 72)); printf -v s3 '%03d' "$start"; printf -v e3 '%03d' "$end"
    chunk_args+=(--chunk "$layer_root/codec/hessian-trellis-layer-$l3-experts-$s3-$e3.safetensors")
  done
  PYTHONPATH="$REPO" "$PYTHON_BIN" -m glm53_nvfp4.build_p8_tp4_sidecars \
    "${chunk_args[@]}" --layer "$layer" --world-size 4 --output-dir "$SIDECARS" \
    --receipt "$RECEIPTS/layer-$l3.json" >"$layer_root/logs/sidecars.json"
  layer_complete "$layer"

  # Codec chunks are reproducible intermediates. Their content hashes remain
  # in the retained sidecar receipt; remove them after the serving checkpoint
  # has passed its exact-rate and hash checks.
  for path in "$layer_root"/codec/*.safetensors; do
    [ ! -f "$path" ] || unlink -- "$path"
  done
}

reuse_layer20() {
  local rank src dst receipt=$RECEIPTS/layer-020.json
  [ -f "$REUSE20/receipt.json" ] || return 1
  for rank in 0 1 2 3; do
    src=$REUSE20/p8-layer-020-tp4-rank-$rank.safetensors
    dst=$SIDECARS/p8-layer-020-tp4-rank-$rank.safetensors
    [ -f "$dst" ] || cp --reflink=auto "$src" "$dst"
  done
  [ -f "$receipt" ] || cp --reflink=auto "$REUSE20/receipt.json" "$receipt"
  layer_complete 20
}

write_manifest() {
  PYTHONPATH="$REPO" "$PYTHON_BIN" - "$DESIGN" "$SIDECARS" "$RECEIPTS" "$ROOT/manifest.json" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
from safetensors import safe_open
design, sidecars, receipts, output = map(Path, sys.argv[1:])
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8<<20),b""): h.update(chunk)
    return h.hexdigest()
design_sha = sha(design)
layers=[]; total_payload=0; total_logical=0
for layer in range(3,45):
    receipt_path=receipts/f"layer-{layer:03d}.json"
    receipt=json.loads(receipt_path.read_text())
    ranks=[]
    for rank in range(4):
        path=sidecars/f"p8-layer-{layer:03d}-tp4-rank-{rank}.safetensors"
        digest=sha(path)
        with safe_open(path,framework="pt",device="cpu") as src:
            meta=src.metadata() or {}
            if meta.get("source_design_sha256") != design_sha: raise SystemExit(f"design mismatch {path}")
        rr=next(x for x in receipt["ranks"] if x["rank"]==rank)
        if digest != rr["sha256"] or rr["payload_bpw"] != 4.25: raise SystemExit(f"receipt mismatch {path}")
        total_payload += rr["payload_bytes"]; total_logical += rr["logical_elements"]
        ranks.append({"rank":rank,"path":str(path.resolve()),"bytes":path.stat().st_size,"sha256":digest,"payload_bpw":rr["payload_bpw"]})
    layers.append({"layer":layer,"receipt":{"path":str(receipt_path.resolve()),"sha256":sha(receipt_path)},"ranks":ranks})
payload={
 "schema":"glm53-p8-uniform-all42-tp4-checkpoint.v1",
 "created_at":datetime.now(timezone.utc).isoformat(),
 "layers":layers,
 "layer_range":[3,44],
 "world_size":4,
 "design":{"path":str(design.resolve()),"sha256":design_sha},
 "codec":{"bits":4,"law":"procedural-mcg-alpha2","alphabet":"e4m3","scale":"ue8m0-k32","ldlq":False},
 "payload":{"bytes":total_payload,"logical_elements":total_logical,"bpw":total_payload*8/total_logical},
 "protected_boundary":"fit capture only; no selection, confirmation, final, or reserved confirmation logits opened",
}
if payload["payload"]["bpw"] != 4.25: raise SystemExit("aggregate rate mismatch")
if output.exists(): raise SystemExit(f"refusing overwrite {output}")
output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(sha(output))
PY
}

exec 9>"$LOCK"
flock -w 900 9
timer_was_active=false
backend_was_active=false
systemctl is-active --quiet klc-model-stack.timer && timer_was_active=true
systemctl --user is-active --quiet klc-backend.service && backend_was_active=true
restore() {
  set +e
  [ "$backend_was_active" = false ] || systemctl --user start klc-backend.service
  [ "$timer_was_active" = false ] || sudo -n systemctl start klc-model-stack.timer
  log "restored timer=$timer_was_active backend=$backend_was_active"
}
trap restore EXIT
[ "$timer_was_active" = false ] || sudo -n systemctl stop klc-model-stack.timer
[ "$backend_was_active" = false ] || systemctl --user stop klc-backend.service
log "uniform P8 build started root=$ROOT"

for layer in $(seq 3 44); do
  if layer_complete "$layer"; then
    log "layer=$layer already complete"
    remove_capture "$layer"
    continue
  fi
  if [ "$layer" -eq 20 ] && reuse_layer20; then
    log "layer=20 reused from design-bound device-closed sidecars"
    remove_capture 20
    continue
  fi
  log "layer=$layer prefetching fit-only capture ranges"
  prefetch_layer "$layer" >>"$LOG" 2>&1
  log "layer=$layer encoding"
  encode_layer "$layer"
  remove_capture "$layer"
  log "layer=$layer complete"
done

write_manifest | tee -a "$LOG"
log "uniform P8 build complete manifest=$ROOT/manifest.json"
