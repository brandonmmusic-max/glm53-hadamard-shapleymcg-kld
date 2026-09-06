#!/usr/bin/env bash
# Unattended Phase 0 -> Phase 1 campaign, candidate-screened variant.  After the full
# coupled K4 encode finishes:
#   1. single-arm CF32 KLD of the uniform coupled K4 model (v10 image)
#   2. K4-only Shapley damage for all 42 layers (one worker per GPU, no encodes)
#   3. preliminary allocation with assumed K3/K5 damage ratios -> candidate layer sets
#   4. exact K5 encodes+scores for the K5 candidates, then K3 for the K3 candidates
#      (candidate sidecars are packed and kept)
#   5. K3 and K5 M1 device closures on candidate sidecars (v11 image)
#   6. final allocation from measured damage (unscored layers stay K4)
#   7. mixed-rate checkpoint assembly (all layers hard-linked: K4 build or candidates)
#   8. single-arm CF32 KLD of the allocated model (v11 image)
# Every step logs under $CAMPAIGN/phase1-campaign-v1 and the script stops at the
# first failure, preserving receipts.  Production is never touched.
set -euo pipefail

REPO=${GLM53_COUPLED_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CAMPAIGN=${GLM53_CAMPAIGN_ROOT:-/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6}
KLCSTORE=${GLM53_KLCSTORE:-/media/brandonmusic/klcstore/bmxfp4-glm53}
K4_ROOT=$CAMPAIGN/full-coupled-p8-all42-v1
RATE_ROOT=$CAMPAIGN/rate-candidates-v1
MIXED_ROOT=$CAMPAIGN/mixed-rate-p8-all42-v1
SCORE_CAPTURE=$CAMPAIGN/fit-capture-scoring-v1
OUT=$CAMPAIGN/phase1-campaign-v1
PY=/usr/bin/python3
MAX_NEW_BYTES=${GLM53_MAX_NEW_BYTES:-400000000000}
V10_RECEIPT=$CAMPAIGN/p8-full-coupled-image-v10-build-b/receipt.json
V11_RECEIPT=$CAMPAIGN/p8-mixed-rate-image-v11b-build/receipt.json
V11_IMAGE=sha256:7676b8f53138886aa753c7bb66c00bb0b49811310db257917d0158be2a35a141
V11_MANIFEST_SHA=b2e7760471df45da672068aa3ec084ebf2224140cc266e01675d78e82b3b1bc8
RECIPE=$CAMPAIGN/p8-smallm-scheduler-v1/fc1-integrated-v1/container-final.private.json
MODEL=/home/brandonmusic/models/GLM-5.3-Flash-NVFP4
CARRIER_RECEIPT=$REPO/results/P8_STOCK_NVFP4_CARRIER_V1.json
CARRIER_EXTRAS=$CAMPAIGN/p8-coupled-image-v9-build/carrier-extra-audit/receipt.json
ROLES=$KLCSTORE/roles/roles-codec-conditional-fit32-v1.json
TEACHER=$KLCSTORE/teacher
TRANSFORM=$REPO/experiments/p8-coupled-transform-draw0-silu10-v1.json
DESIGN_V4=$REPO/results/P8_COUPLED_SCALE_FLASH_EVIDENCE_PREPARATION_V4.json
DESIGN_V3=$REPO/results/P8_COUPLED_SCALE_FLASH_EVIDENCE_PREPARATION_V3.json
DESIGN_K3=$REPO/results/P8_COUPLED_RATE_DESIGN_K3_V1.json
DESIGN_K5=$REPO/results/P8_COUPLED_RATE_DESIGN_K5_V1.json
ALLOCATION=$REPO/results/P8_LAYER_RATE_ALLOCATION_V1.json
PRELIMINARY=$OUT/preliminary-allocation.json
# Assumed damage ratios for the preliminary screen: the layer-4 smoke encodes measured
# transformed-weight NMSE of 3.76x (K3) and 0.285x (K5) relative to K4.
ASSUME_K3=${GLM53_ASSUME_K3_RATIO:-3.76}
ASSUME_K5=${GLM53_ASSUME_K5_RATIO:-0.285}
CANDIDATE_MARGIN=${GLM53_CANDIDATE_MARGIN:-3}
MIN_CANDIDATES=${GLM53_MIN_CANDIDATES:-4}
LOCK=/run/lock/klc/model-stack.lock

mkdir -p "$OUT"
cd "$REPO"
export PYTHONPATH="$REPO"
LOG=$OUT/campaign.log
log() { echo "[$(date --iso-8601=seconds)] $*" | tee -a "$LOG"; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
step_done() { [ -f "$OUT/step-$1.done" ]; }
mark_done() { date --iso-8601=seconds > "$OUT/step-$1.done"; }
wait_cool() {
  local gpu=$1 temp
  for _ in $(seq 1 120); do
    temp=$(nvidia-smi -i "$gpu" --query-gpu=temperature.gpu --format=csv,noheader,nounits | tr -d ' ')
    [ "$temp" -gt 75 ] || return 0
    sleep 5
  done
  return 1
}
common_paths=(--campaign-path "$K4_ROOT" --campaign-path "$RATE_ROOT" --campaign-path "$MIXED_ROOT" --campaign-path "$SCORE_CAPTURE" --campaign-path "$OUT")

kld_run() {
  # kld_run NAME IMAGE_RECEIPT COUPLED_MANIFEST COUPLED_SIDECARS DESIGN...
  local name=$1 receipt=$2 manifest=$3 sidecars=$4; shift 4
  local -a designs=()
  for d in "$@"; do designs+=(--coupled-design "$d"); done
  local runtime_manifest=$OUT/$name-runtime.json seal=$OUT/$name-seal.json
  if [ ! -f "$runtime_manifest" ]; then
    log "$name: preparing runtime manifest"
    "$PY" scripts/prepare_p8_full_coupled_cf32_runtime.py --output "$runtime_manifest" --image-build-receipt "$receipt" \
      --source-recipe "$RECIPE" --model-root "$MODEL" --stock-carrier-receipt "$CARRIER_RECEIPT" \
      --stock-carrier-extras-receipt "$CARRIER_EXTRAS" --coupled-manifest "$manifest" --coupled-sidecars "$sidecars" \
      "${designs[@]}" --transform "$TRANSFORM" --roles "$ROLES" --teacher-root "$TEACHER" "${common_paths[@]}" \
      --max-new-bytes "$MAX_NEW_BYTES" --arm coupled_full >>"$OUT/$name-prepare.log" 2>&1
  fi
  if [ ! -f "$seal" ]; then
    log "$name: sealing execution"
    "$PY" -m glm53_nvfp4.p8_full_coupled_cf32_executor seal --manifest "$runtime_manifest" --seal "$seal" \
      --global-root "$CAMPAIGN" >>"$OUT/$name-seal.log" 2>&1
  fi
  log "$name: executing single-arm CF32 KLD"
  "$PY" -m glm53_nvfp4.p8_full_coupled_cf32_executor execute --seal "$seal" >>"$OUT/$name-execute.log" 2>&1
  "$PY" - "$OUT/$name-runtime-captures/analysis.json" <<'PY' | tee -a "$LOG"
import json, sys
a = json.load(open(sys.argv[1]))
arm = a["arms"]["coupled_full"]
print(json.dumps({"true_decode_mean_kld": arm["true_decode_mean_kld"], "including_prefill_mean_kld": arm["including_prefill_mean_kld"],
                  "bca95": arm["true_decode_window_bca95"], "per_domain": arm["per_domain_true_decode_mean_kld"],
                  "conditions": arm["conditions"]}, indent=1))
PY
}
candidate_list() {
  # candidate_list RATE -> space-separated layer list from the preliminary allocation
  "$PY" - "$PRELIMINARY" "$1" <<'PY'
import json, sys
print(" ".join(str(x) for x in json.load(open(sys.argv[1]))["candidates"][sys.argv[2]]))
PY
}

log "phase 1 campaign v2 started repo=$(git -C "$REPO" rev-parse --short HEAD) assume_k3=$ASSUME_K3 assume_k5=$ASSUME_K5 margin=$CANDIDATE_MARGIN min_candidates=$MIN_CANDIDATES"

# 0. wait for the full coupled K4 build
until [ -f "$K4_ROOT/manifest.json" ] && grep -q "full coupled P8 build complete" "$K4_ROOT/build.log"; do sleep 60; done
exec 9>"$LOCK"; flock -w 7200 9; flock -u 9; exec 9>&-
log "K4 build complete; manifest $(sha "$K4_ROOT/manifest.json")"

# 1. uniform coupled KLD (v10 image)
if ! step_done 1; then
  kld_run uniform-coupled-cf32-v1 "$V10_RECEIPT" "$K4_ROOT/manifest.json" "$K4_ROOT/sidecars" "$DESIGN_V4" "$DESIGN_V3"
  mark_done 1
fi

# 2. K4-only damage for every layer, one worker per GPU
if ! step_done 2; then
  log "scoring K4 damage for all layers"
  GLM53_RATE_DESIGN_K3=$DESIGN_K3 GLM53_RATE_DESIGN_K5=$DESIGN_K5 GLM53_MAX_NEW_BYTES=$MAX_NEW_BYTES \
    bash scripts/score_k4_damage_parallel.sh >>"$OUT/score-k4-only.log" 2>&1
  mark_done 2
fi

# 3. preliminary allocation -> candidate sets
if ! step_done 3; then
  rm -f "$PRELIMINARY"
  "$PY" -m glm53_nvfp4.p8_layer_rate_allocation --receipt-dir "$RATE_ROOT/receipts" --k4-only-dir "$RATE_ROOT/receipts/k4-only" \
    --assume-ratio "3=$ASSUME_K3" --assume-ratio "5=$ASSUME_K5" --candidate-margin "$CANDIDATE_MARGIN" \
    --min-candidates "$MIN_CANDIDATES" --output "$PRELIMINARY" | tee -a "$LOG"
  mark_done 3
fi
K5_LAYERS=$(candidate_list 5)
K3_LAYERS=$(candidate_list 3)
log "candidates K5='$K5_LAYERS' K3='$K3_LAYERS'"

# 4. exact candidate encodes and scores (sidecars kept for reuse)
if ! step_done 4; then
  log "encoding and scoring K5 candidates"
  GLM53_RATE_LAYERS=$K5_LAYERS GLM53_RATE_CANDIDATES=5 GLM53_KEEP_CANDIDATE_LAYERS=all \
    GLM53_RATE_DESIGN_K3=$DESIGN_K3 GLM53_RATE_DESIGN_K5=$DESIGN_K5 GLM53_MAX_NEW_BYTES=$MAX_NEW_BYTES \
    bash scripts/score_layer_rates_all42.sh >>"$OUT/score-k5-candidates.log" 2>&1
  log "encoding and scoring K3 candidates"
  GLM53_RATE_LAYERS=$K3_LAYERS GLM53_RATE_CANDIDATES=3 GLM53_KEEP_CANDIDATE_LAYERS=all \
    GLM53_RATE_DESIGN_K3=$DESIGN_K3 GLM53_RATE_DESIGN_K5=$DESIGN_K5 GLM53_MAX_NEW_BYTES=$MAX_NEW_BYTES \
    bash scripts/score_layer_rates_all42.sh >>"$OUT/score-k3-candidates.log" 2>&1
  mark_done 4
fi

# 5. device closures for K3 and K5 on candidate sidecars (v11 image)
if ! step_done 5; then
  for bits in 3 5; do
    if [ "$bits" = 3 ]; then layer=${K3_LAYERS%% *}; design=$DESIGN_K3; else layer=${K5_LAYERS%% *}; design=$DESIGN_K5; fi
    sidecar=$RATE_ROOT/candidate-sidecars/layer-$(printf '%03d' "$layer")/k$bits/p8-layer-$(printf '%03d' "$layer")-tp4-rank-0.safetensors
    [ -f "$sidecar" ] || { log "no K$bits candidate sidecar for layer $layer"; exit 1; }
    rm -rf "$OUT/closure-k$bits"
    wait_cool 0 || { log "GPU 0 did not cool below 75 C for the closure"; exit 1; }
    log "K$bits M1 device closure on layer $layer"
    "$PY" scripts/run_p8_mixed_rate_m1_device_closure.py --execute --image "$V11_IMAGE" --gpu-device 0 \
      --sidecar "$sidecar" --sidecar-sha256 "$(sha "$sidecar")" --design "$design" --design-sha256 "$(sha "$design")" \
      --transform "$TRANSFORM" --transform-sha256 "$(sha "$TRANSFORM")" --runtime-manifest-sha256 "$V11_MANIFEST_SHA" \
      --output "$OUT/closure-k$bits" --bits "$bits" --layer "$layer" >>"$OUT/closure-k$bits.log" 2>&1
    log "K$bits closure: $(tail -1 "$OUT/closure-k$bits.log")"
  done
  mark_done 5
fi

# 6. final allocation from measured damage
if ! step_done 6; then
  rm -f "$ALLOCATION"
  "$PY" -m glm53_nvfp4.p8_layer_rate_allocation --receipt-dir "$RATE_ROOT/receipts" --k4-only-dir "$RATE_ROOT/receipts/k4-only" \
    --output "$ALLOCATION" | tee -a "$LOG"
  mark_done 6
fi

# 7. assembly (hard links only: K4 layers from the full build, K3/K5 from kept candidates)
if ! step_done 7; then
  log "assembling the mixed-rate checkpoint"
  GLM53_RATE_ALLOCATION=$ALLOCATION GLM53_RATE_DESIGN_K3=$DESIGN_K3 GLM53_RATE_DESIGN_K5=$DESIGN_K5 GLM53_MAX_NEW_BYTES=$MAX_NEW_BYTES \
    bash scripts/assemble_mixed_rate_p8_all42.sh >>"$OUT/assemble.log" 2>&1
  mark_done 7
fi

# 8. mixed-rate KLD (v11 image); designs exactly as installed
if ! step_done 8; then
  mapfile -t mixed_designs < <("$PY" - "$MIXED_ROOT/manifest.json" "$DESIGN_V4" "$DESIGN_V3" "$DESIGN_K3" "$DESIGN_K5" <<'PY'
import hashlib, json, sys
manifest = json.load(open(sys.argv[1]))
files = {}
for path in sys.argv[2:]:
    files[hashlib.sha256(open(path, "rb").read()).hexdigest()] = path
for digest in sorted(manifest["designs"]):
    print(files[digest])
PY
)
  kld_run mixed-rate-cf32-v1 "$V11_RECEIPT" "$MIXED_ROOT/manifest.json" "$MIXED_ROOT/sidecars" "${mixed_designs[@]}"
  mark_done 8
fi
log "phase 1 campaign complete"
