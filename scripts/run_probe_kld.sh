#!/usr/bin/env bash
# One sealed single-arm CF32 KLD measurement of an already-assembled mixed-rate checkpoint.
# Same 32-window protocol, same image lineage and same seal/execute path as the campaign; only
# the checkpoint under test differs.  Used by the diagnostic group probes.
#
# Required: GLM53_PROBE_NAME, GLM53_PROBE_MANIFEST, GLM53_PROBE_SIDECARS
set -euo pipefail

REPO=${GLM53_COUPLED_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CAMPAIGN=${GLM53_CAMPAIGN_ROOT:-/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6}
KLCSTORE=${GLM53_KLCSTORE:-/media/brandonmusic/klcstore/bmxfp4-glm53}
OUT=$CAMPAIGN/phase1-campaign-v1
K4_ROOT=$CAMPAIGN/full-coupled-p8-all42-v1
RATE_ROOT=$CAMPAIGN/rate-candidates-v1
MIXED_ROOT=$CAMPAIGN/mixed-rate-p8-all42-v1
PROBES=$CAMPAIGN/group-probes-v1
SCORE_CAPTURE=$CAMPAIGN/fit-capture-scoring-v1
PY=/usr/bin/python3
MAX_NEW_BYTES=${GLM53_MAX_NEW_BYTES:-400000000000}
V11_RECEIPT=$CAMPAIGN/p8-mixed-rate-image-v11b-build/receipt.json
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
NAME=${GLM53_PROBE_NAME:?set GLM53_PROBE_NAME}
MANIFEST=${GLM53_PROBE_MANIFEST:?set GLM53_PROBE_MANIFEST}
SIDECARS=${GLM53_PROBE_SIDECARS:?set GLM53_PROBE_SIDECARS}

cd "$REPO"
export PYTHONPATH="$REPO"
mkdir -p "$OUT"
log() { echo "[$(date --iso-8601=seconds)] $*"; }

mapfile -t designs < <("$PY" - "$MANIFEST" "$DESIGN_V4" "$DESIGN_V3" "$DESIGN_K3" "$DESIGN_K5" <<'PY'
import hashlib, json, sys
manifest = json.load(open(sys.argv[1]))
files = {hashlib.sha256(open(p, "rb").read()).hexdigest(): p for p in sys.argv[2:]}
for digest in sorted(manifest["designs"]):
    print(files[digest])
PY
)
design_args=()
for d in "${designs[@]}"; do design_args+=(--coupled-design "$d"); done

runtime_manifest=$OUT/$NAME-runtime.json
seal=$OUT/$NAME-seal.json
if [ ! -f "$runtime_manifest" ]; then
  log "$NAME: preparing runtime manifest"
  "$PY" scripts/prepare_p8_full_coupled_cf32_runtime.py --output "$runtime_manifest" --image-build-receipt "$V11_RECEIPT" \
    --source-recipe "$RECIPE" --model-root "$MODEL" --stock-carrier-receipt "$CARRIER_RECEIPT" \
    --stock-carrier-extras-receipt "$CARRIER_EXTRAS" --coupled-manifest "$MANIFEST" --coupled-sidecars "$SIDECARS" \
    "${design_args[@]}" --transform "$TRANSFORM" --roles "$ROLES" --teacher-root "$TEACHER" \
    --campaign-path "$K4_ROOT" --campaign-path "$RATE_ROOT" --campaign-path "$MIXED_ROOT" \
    --campaign-path "$PROBES" --campaign-path "$SCORE_CAPTURE" --campaign-path "$OUT" \
    --max-new-bytes "$MAX_NEW_BYTES" --arm coupled_full
fi
if [ ! -f "$seal" ]; then
  log "$NAME: sealing execution"
  "$PY" -m glm53_nvfp4.p8_full_coupled_cf32_executor seal --manifest "$runtime_manifest" --seal "$seal" --global-root "$CAMPAIGN"
fi
log "$NAME: executing single-arm CF32 KLD"
"$PY" -m glm53_nvfp4.p8_full_coupled_cf32_executor execute --seal "$seal"
"$PY" - "$OUT/$NAME-runtime-captures/analysis.json" <<'PY'
import json, sys
arm = json.load(open(sys.argv[1]))["arms"]["coupled_full"]
print(json.dumps({"true_decode_mean_kld": arm["true_decode_mean_kld"],
                  "bca95": arm["true_decode_window_bca95"]}, indent=1))
PY
