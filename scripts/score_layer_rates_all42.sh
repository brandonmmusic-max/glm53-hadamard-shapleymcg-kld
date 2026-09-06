#!/usr/bin/env bash
# Phase 1 candidate pass: for every routed layer encode transient K3 and K5 coupled
# chunks, score K3/K4/K5 fit-role routed-output damage (K4 from the packed full
# coupled sidecars), keep only the damage receipt, and release the chunks.
#
# Required environment:
#   GLM53_RATE_DESIGN_K3 / GLM53_RATE_DESIGN_K5   rate preparation designs (prepare_p8_coupled_rate_design)
#   GLM53_MAX_NEW_BYTES                          owner-approved aggregate new-byte ceiling
set -euo pipefail

REPO=${GLM53_COUPLED_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CAMPAIGN=${GLM53_CAMPAIGN_ROOT:-/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6}
K4_ROOT=${GLM53_FULL_COUPLED_ROOT:-$CAMPAIGN/full-coupled-p8-all42-v1}
ROOT=${GLM53_RATE_SCORE_ROOT:-$CAMPAIGN/rate-candidates-v1}
KLCSTORE=${GLM53_KLCSTORE:-/media/brandonmusic/klcstore/bmxfp4-glm53}
SOURCE=${GLM53_BF16_SOURCE:-$KLCSTORE/downloads/GLM-5.3-Flash-BF16}
INDEX=$SOURCE/model.safetensors.index.json
CAPTURE_SOURCE=${GLM53_CAPTURE_SOURCE:-$KLCSTORE/teacher/calibration/main-ep4-full}
CAPTURE=${GLM53_RATE_CAPTURE_ROOT:-$CAMPAIGN/fit-capture-scoring-v1}
ROLES=${GLM53_FIT_ROLES:-$KLCSTORE/roles/roles-v5.json}
DESIGN_K3=${GLM53_RATE_DESIGN_K3:?set GLM53_RATE_DESIGN_K3}
DESIGN_K5=${GLM53_RATE_DESIGN_K5:?set GLM53_RATE_DESIGN_K5}
EXL3=${GLM53_EXL3_SCALES:-/home/brandonmusic/models/GLM-5.3-Flash-EXL3-4bpw}
MAX_NEW_BYTES=${GLM53_MAX_NEW_BYTES:?set GLM53_MAX_NEW_BYTES}
PYTHON=${GLM53_ENCODER_PYTHON:-/usr/bin/python3}
read -r -a GPUS <<<"${GLM53_ENCODER_GPUS:-0 1 2 3}"
SCORE_GPU=${GLM53_SCORE_GPU:-${GPUS[0]}}
THERMAL_PAUSE_C=${GLM53_P8_THERMAL_PAUSE_C:-90}
THERMAL_RESUME_C=${GLM53_P8_THERMAL_RESUME_C:-85}
SAMPLES=${GLM53_COUPLED_SAMPLES:-256}
TOKENS_PER_WINDOW=${GLM53_SCORE_TOKENS_PER_WINDOW:-48}
LAYERS=${GLM53_RATE_LAYERS:-$(seq 3 44 | tr '\n' ' ')}
LOCK=/run/lock/klc/model-stack.lock

WORK=$ROOT/work
RECEIPTS=$ROOT/receipts
LOGS=$ROOT/logs
LOG=$ROOT/score.log
LEDGER=$ROOT/storage-ledger.jsonl
THERMAL=$ROOT/thermal-events.jsonl
K3_CHUNK_BYTES=2947983360
K5_CHUNK_BYTES=4759922688
CAPTURE_BYTES=1080033280
NEED_BYTES=$((K3_CHUNK_BYTES + K5_CHUNK_BYTES + CAPTURE_BYTES))

[ "$THERMAL_RESUME_C" -lt "$THERMAL_PAUSE_C" ] && [ "$THERMAL_PAUSE_C" -le 90 ] || { echo "invalid thermal thresholds" >&2; exit 2; }
mkdir -p "$WORK" "$RECEIPTS/chunks" "$LOGS" "$CAPTURE/layers"
cd "$REPO"
export PYTHONPATH="$REPO"

log() { echo "[$(date --iso-8601=seconds)] $*" | tee -a "$LOG"; }
support() { "$PYTHON" -m glm53_nvfp4.full_coupled_build_support "$@"; }
gpu_list() { local IFS=,; echo "${GPUS[*]}"; }

for design in "$DESIGN_K3:3" "$DESIGN_K5:5"; do
  "$PYTHON" - "${design%%:*}" "${design##*:}" "$MAX_NEW_BYTES" "$CAPTURE_SOURCE/capture-manifest.json" "$ROLES" <<'PY'
import hashlib, json, sys
from pathlib import Path
design = json.loads(Path(sys.argv[1]).read_text())
bits = int(sys.argv[2])
if design.get("layers") != list(range(3, 45)) or design.get("bits") != bits:
    raise SystemExit(f"design is not the 42-layer K{bits} rate preparation")
if design.get("storage_budget", {}).get("max_new_bytes") != int(sys.argv[3]):
    raise SystemExit("design storage ceiling differs from GLM53_MAX_NEW_BYTES")
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()
if design["inputs"]["capture_manifest"]["sha256"] != sha(sys.argv[4]) or design["inputs"]["fit_roles"]["sha256"] != sha(sys.argv[5]):
    raise SystemExit("capture manifest or roles differ from the design")
PY
done
if [ ! -f "$CAPTURE/capture-manifest.json" ]; then
  cp "$CAPTURE_SOURCE/capture-manifest.json" "$CAPTURE/capture-manifest.json"
fi
cmp -s "$CAPTURE/capture-manifest.json" "$CAPTURE_SOURCE/capture-manifest.json" || { echo "transient capture manifest differs" >&2; exit 2; }

guard() {
  support guard --root "$ROOT" --campaign-path "$K4_ROOT" --campaign-path "$ROOT" --campaign-path "$CAPTURE" \
    --max-new-bytes "$MAX_NEW_BYTES" --need-bytes "$1" --gpus "" >>"$LOGS/guard.jsonl"
}
ledger() {
  support ledger --root "$ROOT" --campaign-path "$K4_ROOT" --campaign-path "$ROOT" --campaign-path "$CAPTURE" \
    --max-new-bytes "$MAX_NEW_BYTES" --output "$LEDGER" --note "$1" >/dev/null
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
chunk_path() { printf '%s/layer-%03d/k%d/p8-coupled-k%d-layer-%03d-experts-%03d-%03d.safetensors' "$WORK" "$1" "$2" "$2" "$1" "$3" "$4"; }
chunk_receipt() { printf '%s/chunks/layer-%03d-k%d-experts-%03d-%03d.json' "$RECEIPTS" "$1" "$2" "$3" "$4"; }

encode_rate() {
  local layer=$1 bits=$2 design=$3 l3 slot gpu start end codec receipt pid status temp
  printf -v l3 '%03d' "$layer"
  mkdir -p "$WORK/layer-$l3/k$bits"
  local -a pids=(0 0 0 0) gpus=(0 0 0 0) stopped=(0 0 0 0)
  local wave=${#GPUS[@]} first=0
  while [ "$first" -lt 4 ]; do
    local last=$((first + wave)); [ "$last" -le 4 ] || last=4
    for slot in $(seq "$first" $((last - 1))); do
      gpu=${GPUS[$((slot - first))]}; start=$((slot * 72)); end=$((start + 72))
      codec=$(chunk_path "$layer" "$bits" "$start" "$end"); receipt=$(chunk_receipt "$layer" "$bits" "$start" "$end")
      if [ -f "$codec" ] && [ -f "$receipt" ]; then pids[$slot]=0; gpus[$slot]=$gpu; continue; fi
      if [ -e "$codec" ] || [ -e "$receipt" ]; then log "layer=$layer k$bits inconsistent partial chunk=$start:$end"; return 1; fi
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
}

score_layer() {
  local layer=$1 l3; printf -v l3 '%03d' "$layer"
  local -a k3=() k5=() k4r=()
  for start in 0 72 144 216; do
    k3+=(--k3-chunk "$(chunk_path "$layer" 3 "$start" $((start + 72)))")
    k5+=(--k5-chunk "$(chunk_path "$layer" 5 "$start" $((start + 72)))")
    k4r+=(--k4-chunk-receipt "$K4_ROOT/receipts/chunks/layer-$l3-experts-$(printf '%03d' "$start")-$(printf '%03d' $((start + 72))).json")
  done
  CUDA_VISIBLE_DEVICES=$SCORE_GPU OMP_NUM_THREADS=8 "$PYTHON" -u -m glm53_nvfp4.p8_layer_rate_damage --layer "$layer" \
    --source "$SOURCE" --source-index "$INDEX" --exl3-scales "$EXL3" --capture-root "$CAPTURE" --roles "$ROLES" \
    --k4-rank-dir "$K4_ROOT/sidecars" "${k3[@]}" "${k5[@]}" "${k4r[@]}" --tokens-per-window "$TOKENS_PER_WINDOW" \
    --device cuda:0 --output "$RECEIPTS/damage-layer-$l3.json" >"$LOGS/layer-$l3-score.log" 2>&1
}

release_chunks() {
  local layer=$1 l3; printf -v l3 '%03d' "$layer"
  for bits in 3 5; do
    for start in 0 72 144 216; do
      codec=$(chunk_path "$layer" "$bits" "$start" $((start + 72)))
      [ ! -f "$codec" ] || unlink -- "$codec"
    done
  done
}

exec 9>"$LOCK"
flock -w 900 9
log "layer rate scoring started root=$ROOT k4_root=$K4_ROOT ceiling=$MAX_NEW_BYTES gpus=$(gpu_list) score_gpu=$SCORE_GPU thermal=$THERMAL_PAUSE_C/$THERMAL_RESUME_C"
guard "$NEED_BYTES"
ledger "start"

for layer in $LAYERS; do
  printf -v l3 '%03d' "$layer"
  if [ -f "$RECEIPTS/damage-layer-$l3.json" ]; then
    log "layer=$layer already scored"; release_chunks "$layer"; remove_capture "$layer"; continue
  fi
  for rank in 0 1 2 3; do
    [ -f "$K4_ROOT/sidecars/p8-layer-$l3-tp4-rank-$rank.safetensors" ] || { log "layer=$layer missing K4 sidecar rank $rank"; exit 1; }
  done
  guard "$NEED_BYTES"
  log "layer=$layer prefetching fit capture"
  prefetch_layer "$layer"
  log "layer=$layer encoding K3 candidate"
  encode_rate "$layer" 3 "$DESIGN_K3"
  log "layer=$layer encoding K5 candidate"
  encode_rate "$layer" 5 "$DESIGN_K5"
  log "layer=$layer scoring K3/K4/K5"
  score_layer "$layer"
  release_chunks "$layer"
  remove_capture "$layer"
  ledger "scored layer $layer"
  log "layer=$layer scored"
done
log "layer rate scoring complete receipts=$RECEIPTS"
