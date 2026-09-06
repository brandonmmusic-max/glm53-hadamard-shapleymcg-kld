#!/usr/bin/env bash
# Full coupled H512/H128 + suh/svh K4 P8 sidecar build for GLM-5.3-Flash routed
# layers 3..44.  Mirrors scripts/build_uniform_p8_all42.sh but drives the coupled
# encoder, packer and postwrite verifier, reuses the already encoded layers, and
# never restores production.
#
# Required environment:
#   GLM53_COUPLED_DESIGN   42-layer preparation design (prepare_p8_coupled_scale_v1)
#   GLM53_MAX_NEW_BYTES    owner-approved aggregate new-byte ceiling for this build
set -euo pipefail

REPO=${GLM53_COUPLED_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CAMPAIGN=${GLM53_CAMPAIGN_ROOT:-/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6}
ROOT=${GLM53_FULL_COUPLED_ROOT:-$CAMPAIGN/full-coupled-p8-all42-v1}
KLCSTORE=${GLM53_KLCSTORE:-/media/brandonmusic/klcstore/bmxfp4-glm53}
SOURCE=${GLM53_BF16_SOURCE:-$KLCSTORE/downloads/GLM-5.3-Flash-BF16}
INDEX=$SOURCE/model.safetensors.index.json
CAPTURE_SOURCE=${GLM53_CAPTURE_SOURCE:-$KLCSTORE/teacher/calibration/main-ep4-full}
CAPTURE=${GLM53_TRANSIENT_CAPTURE_ROOT:-$CAMPAIGN/fit-capture-all42-transient-v1}
ROLES=${GLM53_FIT_ROLES:-$KLCSTORE/roles/roles-v5.json}
DESIGN=${GLM53_COUPLED_DESIGN:?set GLM53_COUPLED_DESIGN to the 42-layer preparation design}
EXL3=${GLM53_EXL3_SCALES:-/home/brandonmusic/models/GLM-5.3-Flash-EXL3-4bpw}
TRANSFORM=$REPO/experiments/p8-coupled-transform-draw0-silu10-v1.json
IDENTITY_MANIFEST=${GLM53_IDENTITY_MANIFEST:-$CAMPAIGN/uniform-p8-all42-v1/manifest.json}
PRIOR=${GLM53_PRIOR_COUPLED_ROOT:-$CAMPAIGN/p8-coupled-three-layer-v1}
PRIOR_LAYERS=${GLM53_PRIOR_COUPLED_LAYERS:-"3 20 22"}
MAX_NEW_BYTES=${GLM53_MAX_NEW_BYTES:?set GLM53_MAX_NEW_BYTES to the owner-approved ceiling}
PYTHON=${GLM53_ENCODER_PYTHON:-/usr/bin/python3}
read -r -a GPUS <<<"${GLM53_ENCODER_GPUS:-0 1 2 3}"
THERMAL_PAUSE_C=${GLM53_P8_THERMAL_PAUSE_C:-90}
THERMAL_RESUME_C=${GLM53_P8_THERMAL_RESUME_C:-85}
SAMPLES=${GLM53_COUPLED_SAMPLES:-256}
LOCK=/run/lock/klc/model-stack.lock

SIDECARS=$ROOT/sidecars
RECEIPTS=$ROOT/receipts
WORK=$ROOT/work
LOGS=$ROOT/logs
LOG=$ROOT/build.log
LEDGER=$ROOT/storage-ledger.jsonl
THERMAL=$ROOT/thermal-events.jsonl

# Measured on the encoded layers: 4 x 963,497,456 sidecar bytes, the same for
# the four chunks, and 1,080,033,280 sparse fit-capture bytes per layer.
CHUNK_BYTES=3853989824
SIDECAR_BYTES=3853989824
CAPTURE_BYTES=1080033280
NEED_BYTES=$((CHUNK_BYTES + SIDECAR_BYTES + CAPTURE_BYTES))

[[ "$THERMAL_PAUSE_C" =~ ^[0-9]+$ && "$THERMAL_RESUME_C" =~ ^[0-9]+$ ]] || { echo "invalid thermal thresholds" >&2; exit 2; }
[ "$THERMAL_RESUME_C" -lt "$THERMAL_PAUSE_C" ] || { echo "thermal resume must be below pause" >&2; exit 2; }
[ "$THERMAL_PAUSE_C" -le 90 ] || { echo "thermal pause above the 90 C abort limit" >&2; exit 2; }
[[ "$MAX_NEW_BYTES" =~ ^[0-9]+$ ]] || { echo "invalid GLM53_MAX_NEW_BYTES" >&2; exit 2; }
[ "${#GPUS[@]}" -ge 1 ] || { echo "no encoder GPUs" >&2; exit 2; }

mkdir -p "$SIDECARS" "$RECEIPTS/chunks" "$WORK" "$LOGS" "$CAPTURE/layers"
cd "$REPO"
export PYTHONPATH="$REPO"

log() { echo "[$(date --iso-8601=seconds)] $*" | tee -a "$LOG"; }
support() { "$PYTHON" -m glm53_nvfp4.full_coupled_build_support "$@"; }
gpu_list() { local IFS=,; echo "${GPUS[*]}"; }

design_sha=$(sha256sum "$DESIGN" | cut -d' ' -f1)
prior_design_sha=$("$PYTHON" - "$PRIOR" <<'PY'
import json, sys
from pathlib import Path
prior = Path(sys.argv[1])
shas = set()
for receipt in sorted(prior.glob("receipts/layer-*-sidecars.json")):
    for rank in json.loads(receipt.read_text())["ranks"]:
        shas.add(rank["source_design_sha256"])
if len(shas) != 1:
    raise SystemExit(f"prior coupled sidecars carry {len(shas)} design hashes: {sorted(shas)}")
print(shas.pop())
PY
)

# The design must be the 42-layer preparation and must pin the same ceiling the
# owner approved for this build; the encoder re-validates every other input hash.
"$PYTHON" - "$DESIGN" "$MAX_NEW_BYTES" "$CAPTURE_SOURCE/capture-manifest.json" "$ROLES" <<'PY'
import hashlib, json, sys
from pathlib import Path
design = json.loads(Path(sys.argv[1]).read_text())
if design.get("layers") != list(range(3, 45)):
    raise SystemExit("design is not the 42-layer preparation")
if design.get("storage_budget", {}).get("max_new_bytes") != int(sys.argv[2]):
    raise SystemExit("design storage ceiling differs from GLM53_MAX_NEW_BYTES")
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()
if design["inputs"]["capture_manifest"]["sha256"] != sha(sys.argv[3]):
    raise SystemExit("capture manifest differs from the design")
if design["inputs"]["fit_roles"]["sha256"] != sha(sys.argv[4]):
    raise SystemExit("fit roles differ from the design")
PY

if [ ! -f "$CAPTURE/capture-manifest.json" ]; then
  cp "$CAPTURE_SOURCE/capture-manifest.json" "$CAPTURE/capture-manifest.json"
fi
cmp -s "$CAPTURE/capture-manifest.json" "$CAPTURE_SOURCE/capture-manifest.json" || { echo "transient capture manifest differs" >&2; exit 2; }

guard() {
  support guard --root "$ROOT" --campaign-path "$ROOT" --campaign-path "$CAPTURE" \
    --max-new-bytes "$MAX_NEW_BYTES" --need-bytes "$1" --gpus "$(gpu_list)" >>"$LOGS/guard.jsonl"
}

ledger() {
  support ledger --root "$ROOT" --campaign-path "$ROOT" --campaign-path "$CAPTURE" \
    --max-new-bytes "$MAX_NEW_BYTES" --output "$LEDGER" --note "$1" >/dev/null
}

layer_complete() {
  local layer=$1 l3
  printf -v l3 '%03d' "$layer"
  [ -f "$RECEIPTS/layer-$l3-sidecars.json" ] && [ -f "$RECEIPTS/layer-$l3-postwrite.json" ] || return 1
  support layer-complete --layer "$layer" --sidecars "$SIDECARS" \
    --packer-receipt "$RECEIPTS/layer-$l3-sidecars.json" \
    --postwrite-receipt "$RECEIPTS/layer-$l3-postwrite.json" \
    --allow-design-sha256 "$design_sha" --allow-design-sha256 "$prior_design_sha" \
    >"$LOGS/layer-$l3-complete.json"
}

reuse_layer() {
  local layer=$1 l3
  printf -v l3 '%03d' "$layer"
  support reuse-layer --layer "$layer" --source-dir "$PRIOR/sidecars/layer-$l3" \
    --source-packer-receipt "$PRIOR/receipts/layer-$l3-sidecars.json" \
    --source-postwrite-receipt "$PRIOR/receipts/layer-$l3-postwrite.json" \
    --sidecars "$SIDECARS" --receipts "$RECEIPTS" >"$LOGS/layer-$l3-reuse.json"
  layer_complete "$layer"
}

prefetch_layer() {
  local layer=$1 l3
  printf -v l3 '%03d' "$layer"
  "$PYTHON" -m glm53_nvfp4.prefetch_partial_capture \
    --capture-root "$CAPTURE" --roles "$ROLES" --layer "$layer" \
    --role fit --workers 8 \
    --receipt "$CAPTURE/layers/layer-$l3/partial-fit-capture.json" >>"$LOGS/layer-$l3-prefetch.log" 2>&1
}

remove_capture() {
  local layer=$1 l3 name path
  printf -v l3 '%03d' "$layer"
  for name in hidden.bf16.bin topk_ids.u16le.bin topk_weights.f32le.bin; do
    path=$CAPTURE/layers/layer-$l3/$name
    [ ! -f "$path" ] || unlink -- "$path"
  done
}

chunk_path() { printf '%s/layer-%03d/p8-coupled-layer-%03d-experts-%03d-%03d.safetensors' "$WORK" "$1" "$1" "$2" "$3"; }
chunk_receipt() { printf '%s/chunks/layer-%03d-experts-%03d-%03d.json' "$RECEIPTS" "$1" "$2" "$3"; }

encode_layer() {
  local layer=$1 l3 slot gpu start end codec receipt pid status temp
  printf -v l3 '%03d' "$layer"
  mkdir -p "$WORK/layer-$l3"
  local -a pids=(0 0 0 0) gpus=(0 0 0 0) stopped=(0 0 0 0)
  local wave=${#GPUS[@]}
  local first=0
  while [ "$first" -lt 4 ]; do
    local last=$((first + wave)); [ "$last" -le 4 ] || last=4
    for slot in $(seq "$first" $((last - 1))); do
      gpu=${GPUS[$((slot - first))]}; start=$((slot * 72)); end=$((start + 72))
      codec=$(chunk_path "$layer" "$start" "$end"); receipt=$(chunk_receipt "$layer" "$start" "$end")
      if [ -f "$codec" ] && [ -f "$receipt" ]; then
        pids[$slot]=0; gpus[$slot]=$gpu; continue
      fi
      if [ -e "$codec" ] || [ -e "$receipt" ]; then
        log "layer=$layer inconsistent partial chunk=$start:$end; refusing overwrite"
        return 1
      fi
      printf '{"at":"%s","layer":%d,"slot":%d,"gpu":%d,"experts":[%d,%d],"design_sha256":"%s"}\n' \
        "$(date --iso-8601=seconds)" "$layer" "$slot" "$gpu" "$start" "$end" "$design_sha" \
        >>"$LOGS/layer-$l3-chunk-launches.jsonl"
      CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 "$PYTHON" -u -m glm53_nvfp4.quantize_p8_coupled_scale_layer \
        --source "$SOURCE" --source-index "$INDEX" --capture-root "$CAPTURE" \
        --roles "$ROLES" --design "$DESIGN" --exl3-scales "$EXL3" \
        --codec-output "$codec" --receipt "$receipt" \
        --layer "$layer" --expert-start "$start" --expert-end "$end" \
        --samples "$SAMPLES" --device cuda:0 \
        >"$LOGS/layer-$l3-chunk-$start-$end.log" 2>&1 &
      pids[$slot]=$!; gpus[$slot]=$gpu
    done
    while :; do
      local alive=0
      for slot in $(seq "$first" $((last - 1))); do
        pid=${pids[$slot]}
        [ "$pid" -ne 0 ] && kill -0 "$pid" 2>/dev/null || continue
        alive=1
        gpu=${gpus[$slot]}
        temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
        if [ "${stopped[$slot]}" -eq 0 ] && [ "$temp" -ge "$THERMAL_PAUSE_C" ]; then
          kill -STOP "$pid"; stopped[$slot]=1
          printf '{"at":"%s","event":"pause","layer":%d,"gpu":%d,"pid":%d,"temperature_c":%d}\n' \
            "$(date --iso-8601=seconds)" "$layer" "$gpu" "$pid" "$temp" >>"$THERMAL"
        elif [ "${stopped[$slot]}" -eq 1 ] && [ "$temp" -le "$THERMAL_RESUME_C" ]; then
          kill -CONT "$pid"; stopped[$slot]=0
          printf '{"at":"%s","event":"resume","layer":%d,"gpu":%d,"pid":%d,"temperature_c":%d}\n' \
            "$(date --iso-8601=seconds)" "$layer" "$gpu" "$pid" "$temp" >>"$THERMAL"
        fi
      done
      [ "$alive" -eq 1 ] || break
      sleep 5
    done
    status=0
    for slot in $(seq "$first" $((last - 1))); do
      pid=${pids[$slot]}
      [ "$pid" -eq 0 ] || wait "$pid" || status=1
    done
    [ "$status" -eq 0 ] || { log "layer=$layer chunk encode failed; see $LOGS"; return 1; }
    first=$last
  done

  local -a chunk_args=() sidecar_args=()
  for start in 0 72 144 216; do
    chunk_args+=(--chunk "$(chunk_path "$layer" "$start" $((start + 72)))")
  done
  for rank in 0 1 2 3; do
    sidecar_args+=(--sidecar "$SIDECARS/p8-layer-$l3-tp4-rank-$rank.safetensors")
  done
  "$PYTHON" -u -m glm53_nvfp4.build_p8_coupled_scale_tp4_sidecars \
    "${chunk_args[@]}" --output-dir "$SIDECARS" --receipt "$RECEIPTS/layer-$l3-sidecars.json" \
    --layer "$layer" --world-size 4 >"$LOGS/layer-$l3-pack.log" 2>&1
  "$PYTHON" -u -m glm53_nvfp4.verify_p8_coupled_scale_tp4_sidecars \
    "${chunk_args[@]}" "${sidecar_args[@]}" --packer-receipt "$RECEIPTS/layer-$l3-sidecars.json" \
    --receipt "$RECEIPTS/layer-$l3-postwrite.json" --layer "$layer" >"$LOGS/layer-$l3-postwrite.log" 2>&1
  layer_complete "$layer"
  # Chunks are reproducible intermediates whose hashes live in the packer
  # receipt and the chunk receipts; the serving sidecars have passed their
  # exact-rate, hash and postwrite checks, so release the chunk bytes now.
  for start in 0 72 144 216; do
    codec=$(chunk_path "$layer" "$start" $((start + 72)))
    [ ! -f "$codec" ] || unlink -- "$codec"
  done
}

write_manifest() {
  support manifest --sidecars "$SIDECARS" --receipts "$RECEIPTS" --output "$ROOT/manifest.json" \
    --transform "$TRANSFORM" --identity-manifest "$IDENTITY_MANIFEST" \
    --allow-design-sha256 "$design_sha" --allow-design-sha256 "$prior_design_sha" | tee -a "$LOG"
}

exec 9>"$LOCK"
flock -w 900 9
log "full coupled P8 build started root=$ROOT design=$design_sha prior_design=$prior_design_sha ceiling=$MAX_NEW_BYTES gpus=$(gpu_list) thermal=$THERMAL_PAUSE_C/$THERMAL_RESUME_C"
guard "$NEED_BYTES"
ledger "start"

for layer in $(seq 3 44); do
  if layer_complete "$layer"; then
    log "layer=$layer already complete"
    remove_capture "$layer"
    continue
  fi
  if [[ " $PRIOR_LAYERS " == *" $layer "* ]]; then
    log "layer=$layer reusing prior coupled sidecars from $PRIOR"
    reuse_layer "$layer"
    ledger "reused layer $layer"
    continue
  fi
  guard "$NEED_BYTES"
  log "layer=$layer prefetching fit-only capture ranges"
  prefetch_layer "$layer"
  log "layer=$layer encoding"
  encode_layer "$layer"
  remove_capture "$layer"
  ledger "completed layer $layer"
  log "layer=$layer complete"
done

write_manifest
ledger "manifest written"
log "full coupled P8 build complete manifest=$ROOT/manifest.json"
