#!/usr/bin/env bash
# Phase 1 install pass: realize the same-size K3/K4/K5 allocation as one flat TP4
# sidecar checkpoint.  K4 layers are hard-linked from the full coupled K4 build;
# K3/K5 layers are encoded from BF16 with the rate encoder, packed, postwrite
# verified and hash-closed against the allocation.  Ends with the mixed-rate
# manifest (exact bytes, same-size gate).
#
# Required environment:
#   GLM53_RATE_ALLOCATION                        allocation JSON from p8_layer_rate_allocation
#   GLM53_RATE_DESIGN_K3 / GLM53_RATE_DESIGN_K5   rate preparation designs
#   GLM53_MAX_NEW_BYTES                          owner-approved aggregate new-byte ceiling
set -euo pipefail

REPO=${GLM53_COUPLED_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CAMPAIGN=${GLM53_CAMPAIGN_ROOT:-/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6}
K4_ROOT=${GLM53_FULL_COUPLED_ROOT:-$CAMPAIGN/full-coupled-p8-all42-v1}
ROOT=${GLM53_MIXED_RATE_ROOT:-$CAMPAIGN/mixed-rate-p8-all42-v1}
KLCSTORE=${GLM53_KLCSTORE:-/media/brandonmusic/klcstore/bmxfp4-glm53}
SOURCE=${GLM53_BF16_SOURCE:-$KLCSTORE/downloads/GLM-5.3-Flash-BF16}
INDEX=$SOURCE/model.safetensors.index.json
CAPTURE_SOURCE=${GLM53_CAPTURE_SOURCE:-$KLCSTORE/teacher/calibration/main-ep4-full}
CAPTURE=${GLM53_RATE_CAPTURE_ROOT:-$CAMPAIGN/fit-capture-scoring-v1}
ROLES=${GLM53_FIT_ROLES:-$KLCSTORE/roles/roles-v5.json}
ALLOCATION=${GLM53_RATE_ALLOCATION:?set GLM53_RATE_ALLOCATION}
DESIGN_K3=${GLM53_RATE_DESIGN_K3:?set GLM53_RATE_DESIGN_K3}
DESIGN_K5=${GLM53_RATE_DESIGN_K5:?set GLM53_RATE_DESIGN_K5}
EXL3=${GLM53_EXL3_SCALES:-/home/brandonmusic/models/GLM-5.3-Flash-EXL3-4bpw}
TRANSFORM=$REPO/experiments/p8-coupled-transform-draw0-silu10-v1.json
MAX_NEW_BYTES=${GLM53_MAX_NEW_BYTES:?set GLM53_MAX_NEW_BYTES}
PYTHON=${GLM53_ENCODER_PYTHON:-/usr/bin/python3}
read -r -a GPUS <<<"${GLM53_ENCODER_GPUS:-0 1 2 3}"
THERMAL_PAUSE_C=${GLM53_P8_THERMAL_PAUSE_C:-90}
THERMAL_RESUME_C=${GLM53_P8_THERMAL_RESUME_C:-85}
SAMPLES=${GLM53_COUPLED_SAMPLES:-256}
CANDIDATE_ROOT=${GLM53_RATE_CANDIDATE_SIDECARS:-$CAMPAIGN/rate-candidates-v1/candidate-sidecars}
# Diagnostic group probes install one arm of the allocation (K5 layers only), which is larger
# than the identity budget on purpose; the same-size gate is waived only for those runs.
ALLOW_LARGER=${GLM53_ALLOW_LARGER:-0}
LOCK=/run/lock/klc/model-stack.lock

SIDECARS=$ROOT/sidecars
RECEIPTS=$ROOT/receipts
WORK=$ROOT/work
LOGS=$ROOT/logs
LOG=$ROOT/build.log
LEDGER=$ROOT/storage-ledger.jsonl
THERMAL=$ROOT/thermal-events.jsonl
CAPTURE_BYTES=1080033280

[ "$THERMAL_RESUME_C" -lt "$THERMAL_PAUSE_C" ] && [ "$THERMAL_PAUSE_C" -le 90 ] || { echo "invalid thermal thresholds" >&2; exit 2; }
mkdir -p "$SIDECARS" "$RECEIPTS/chunks" "$WORK" "$LOGS" "$CAPTURE/layers"
cd "$REPO"
export PYTHONPATH="$REPO"

log() { echo "[$(date --iso-8601=seconds)] $*" | tee -a "$LOG"; }
support() { "$PYTHON" -m glm53_nvfp4.full_coupled_build_support "$@"; }
mixed() { "$PYTHON" -m glm53_nvfp4.mixed_rate_build_support "$@"; }
gpu_list() { local IFS=,; echo "${GPUS[*]}"; }

rate_of() { "$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))['assignment'][sys.argv[2]])" "$ALLOCATION" "$1"; }
design_sha() { sha256sum "$1" | cut -d' ' -f1; }
K4_DESIGN_SHAS=$("$PYTHON" - "$K4_ROOT/manifest.json" <<'PY'
import json, sys
print(" ".join(sorted(json.load(open(sys.argv[1]))["designs"])))
PY
)
ALLOW=()
for s in $K4_DESIGN_SHAS $(design_sha "$DESIGN_K3") $(design_sha "$DESIGN_K5"); do ALLOW+=(--allow-design-sha256 "$s"); done

if [ ! -f "$CAPTURE/capture-manifest.json" ]; then cp "$CAPTURE_SOURCE/capture-manifest.json" "$CAPTURE/capture-manifest.json"; fi
cmp -s "$CAPTURE/capture-manifest.json" "$CAPTURE_SOURCE/capture-manifest.json" || { echo "transient capture manifest differs" >&2; exit 2; }

guard() {
  support guard --root "$ROOT" --campaign-path "$K4_ROOT" --campaign-path "$ROOT" --campaign-path "$CAPTURE" \
    --max-new-bytes "$MAX_NEW_BYTES" --need-bytes "$1" --gpus "" >>"$LOGS/guard.jsonl"
}
ledger() {
  support ledger --root "$ROOT" --campaign-path "$K4_ROOT" --campaign-path "$ROOT" --campaign-path "$CAPTURE" \
    --max-new-bytes "$MAX_NEW_BYTES" --output "$LEDGER" --note "$1" >/dev/null
}
layer_complete() {
  local layer=$1 l3; printf -v l3 '%03d' "$layer"
  [ -f "$RECEIPTS/layer-$l3-sidecars.json" ] && [ -f "$RECEIPTS/layer-$l3-postwrite.json" ] || return 1
  mixed layer-complete --layer "$layer" --allocation "$ALLOCATION" --sidecars "$SIDECARS" \
    --packer-receipt "$RECEIPTS/layer-$l3-sidecars.json" --postwrite-receipt "$RECEIPTS/layer-$l3-postwrite.json" \
    "${ALLOW[@]}" >"$LOGS/layer-$l3-complete.json"
}
link_k4_layer() {
  local layer=$1 l3; printf -v l3 '%03d' "$layer"
  support reuse-layer --layer "$layer" --source-dir "$K4_ROOT/sidecars" \
    --source-packer-receipt "$K4_ROOT/receipts/layer-$l3-sidecars.json" \
    --source-postwrite-receipt "$K4_ROOT/receipts/layer-$l3-postwrite.json" \
    --sidecars "$SIDECARS" --receipts "$RECEIPTS" --reason "allocation keeps K4; identical to the uniform coupled build" \
    >"$LOGS/layer-$l3-reuse.json"
  layer_complete "$layer"
}
link_candidate_layer() {
  # Reuse packed K3/K5 candidate sidecars kept by the scoring pass, if present.
  local layer=$1 bits=$2 l3 src; printf -v l3 '%03d' "$layer"
  src=$CANDIDATE_ROOT/layer-$l3/k$bits
  [ -f "$src/receipt.json" ] && [ -f "$src/postwrite.json" ] || return 1
  mixed reuse-layer --layer "$layer" --allocation "$ALLOCATION" --source-dir "$src" \
    --source-packer-receipt "$src/receipt.json" --source-postwrite-receipt "$src/postwrite.json" \
    --sidecars "$SIDECARS" --receipts "$RECEIPTS" \
    --reason "allocation selects K$bits; candidate sidecars packed during the scoring pass" >"$LOGS/layer-$l3-reuse.json"
  layer_complete "$layer"
}
prefetch_layer() {
  local layer=$1 l3; printf -v l3 '%03d' "$layer"
  "$PYTHON" -m glm53_nvfp4.prefetch_partial_capture --capture-root "$CAPTURE" --roles "$ROLES" --layer "$layer" \
    --role fit --workers 8 --receipt "$CAPTURE/layers/layer-$l3/partial-fit-capture.json" >>"$LOGS/layer-$l3-prefetch.log" 2>&1
}
remove_capture() {
  local layer=$1 l3 name path; printf -v l3 '%03d' "$layer"
  for name in hidden.bf16.bin topk_ids.u16le.bin topk_weights.f32le.bin; do
    path=$CAPTURE/layers/layer-$l3/$name; [ ! -f "$path" ] || unlink -- "$path"
  done
}
chunk_path() { printf '%s/layer-%03d/p8-coupled-k%d-layer-%03d-experts-%03d-%03d.safetensors' "$WORK" "$1" "$2" "$1" "$3" "$4"; }
chunk_receipt() { printf '%s/chunks/layer-%03d-k%d-experts-%03d-%03d.json' "$RECEIPTS" "$1" "$2" "$3" "$4"; }

encode_layer() {
  local layer=$1 bits=$2 design=$3 l3 slot gpu start end codec receipt pid status temp
  printf -v l3 '%03d' "$layer"
  mkdir -p "$WORK/layer-$l3"
  local -a pids=(0 0 0 0) gpus=(0 0 0 0) stopped=(0 0 0 0)
  local wave=${#GPUS[@]} first=0
  while [ "$first" -lt 4 ]; do
    local last=$((first + wave)); [ "$last" -le 4 ] || last=4
    for slot in $(seq "$first" $((last - 1))); do
      gpu=${GPUS[$((slot - first))]}; start=$((slot * 72)); end=$((start + 72))
      codec=$(chunk_path "$layer" "$bits" "$start" "$end"); receipt=$(chunk_receipt "$layer" "$bits" "$start" "$end")
      if [ -f "$codec" ] && [ -f "$receipt" ]; then pids[$slot]=0; gpus[$slot]=$gpu; continue; fi
      if [ -e "$codec" ] || [ -e "$receipt" ]; then log "layer=$layer inconsistent partial chunk=$start:$end"; return 1; fi
      CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 "$PYTHON" -u -m glm53_nvfp4.quantize_p8_coupled_rate_layer \
        --source "$SOURCE" --source-index "$INDEX" --capture-root "$CAPTURE" --roles "$ROLES" --design "$design" \
        --exl3-scales "$EXL3" --codec-output "$codec" --receipt "$receipt" --layer "$layer" \
        --expert-start "$start" --expert-end "$end" --samples "$SAMPLES" --bits "$bits" --device cuda:0 \
        >"$LOGS/layer-$l3-k$bits-chunk-$start-$end.log" 2>&1 &
      pids[$slot]=$!; gpus[$slot]=$gpu
    done
    while :; do
      local alive=0
      for slot in $(seq "$first" $((last - 1))); do
        pid=${pids[$slot]}; [ "$pid" -ne 0 ] && kill -0 "$pid" 2>/dev/null || continue
        alive=1; gpu=${gpus[$slot]}
        temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
        if [ "${stopped[$slot]}" -eq 0 ] && [ "$temp" -ge "$THERMAL_PAUSE_C" ]; then
          kill -STOP "$pid"; stopped[$slot]=1
          printf '{"at":"%s","event":"pause","layer":%d,"bits":%d,"gpu":%d,"pid":%d,"temperature_c":%d}\n' "$(date --iso-8601=seconds)" "$layer" "$bits" "$gpu" "$pid" "$temp" >>"$THERMAL"
        elif [ "${stopped[$slot]}" -eq 1 ] && [ "$temp" -le "$THERMAL_RESUME_C" ]; then
          kill -CONT "$pid"; stopped[$slot]=0
          printf '{"at":"%s","event":"resume","layer":%d,"bits":%d,"gpu":%d,"pid":%d,"temperature_c":%d}\n' "$(date --iso-8601=seconds)" "$layer" "$bits" "$gpu" "$pid" "$temp" >>"$THERMAL"
        fi
      done
      [ "$alive" -eq 1 ] || break
      sleep 5
    done
    status=0
    for slot in $(seq "$first" $((last - 1))); do pid=${pids[$slot]}; [ "$pid" -eq 0 ] || wait "$pid" || status=1; done
    [ "$status" -eq 0 ] || { log "layer=$layer k$bits chunk encode failed"; return 1; }
    first=$last
  done
  local -a chunk_args=() sidecar_args=()
  for start in 0 72 144 216; do chunk_args+=(--chunk "$(chunk_path "$layer" "$bits" "$start" $((start + 72)))"); done
  for rank in 0 1 2 3; do sidecar_args+=(--sidecar "$SIDECARS/p8-layer-$l3-tp4-rank-$rank.safetensors"); done
  "$PYTHON" -u -m glm53_nvfp4.build_p8_coupled_rate_tp4_sidecars "${chunk_args[@]}" --output-dir "$SIDECARS" \
    --receipt "$RECEIPTS/layer-$l3-sidecars.json" --layer "$layer" --world-size 4 >"$LOGS/layer-$l3-pack.log" 2>&1
  "$PYTHON" -u -m glm53_nvfp4.verify_p8_coupled_rate_tp4_sidecars "${chunk_args[@]}" "${sidecar_args[@]}" \
    --packer-receipt "$RECEIPTS/layer-$l3-sidecars.json" --receipt "$RECEIPTS/layer-$l3-postwrite.json" --layer "$layer" \
    >"$LOGS/layer-$l3-postwrite.log" 2>&1
  layer_complete "$layer"
  for start in 0 72 144 216; do
    codec=$(chunk_path "$layer" "$bits" "$start" $((start + 72))); [ ! -f "$codec" ] || unlink -- "$codec"
  done
}

exec 9>"$LOCK"
flock -w 900 9
log "mixed-rate assembly started root=$ROOT allocation=$(sha256sum "$ALLOCATION" | cut -d' ' -f1) ceiling=$MAX_NEW_BYTES gpus=$(gpu_list)"
ledger "start"

for layer in $(seq 3 44); do
  printf -v l3 '%03d' "$layer"
  bits=$(rate_of "$layer")
  if layer_complete "$layer"; then log "layer=$layer already complete (K$bits)"; remove_capture "$layer"; continue; fi
  if [ "$bits" = "4" ]; then
    log "layer=$layer K4: linking the uniform coupled build"; link_k4_layer "$layer"; ledger "linked K4 layer $layer"; continue
  fi
  if link_candidate_layer "$layer" "$bits"; then
    log "layer=$layer K$bits: linked candidate sidecars from the scoring pass"; ledger "linked candidate layer $layer"; continue
  fi
  case "$bits" in 3) design=$DESIGN_K3 ;; 5) design=$DESIGN_K5 ;; *) log "layer=$layer invalid rate $bits"; exit 1 ;; esac
  need=$(( (WEIGHTS=288*3*4096*2048, WEIGHTS*(4*bits+1)/32) * 2 + CAPTURE_BYTES ))
  guard "$need"
  log "layer=$layer K$bits: prefetching fit capture"
  prefetch_layer "$layer"
  log "layer=$layer K$bits: encoding"
  encode_layer "$layer" "$bits" "$design"
  remove_capture "$layer"
  ledger "encoded K$bits layer $layer"
  log "layer=$layer K$bits complete"
done

manifest_args=()
[ "$ALLOW_LARGER" = 1 ] && manifest_args+=(--allow-larger)
mixed manifest --allocation "$ALLOCATION" --sidecars "$SIDECARS" --receipts "$RECEIPTS" --output "$ROOT/manifest.json" \
  --transform "$TRANSFORM" "${ALLOW[@]}" "${manifest_args[@]}" | tee -a "$LOG"
ledger "manifest written"
log "mixed-rate assembly complete manifest=$ROOT/manifest.json"
