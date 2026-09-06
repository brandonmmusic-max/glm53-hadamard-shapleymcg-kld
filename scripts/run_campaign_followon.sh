#!/usr/bin/env bash
# Waits for the Phase 1 campaign to finish, then runs the diagnostic group probes, builds the
# rate-swap evidence table and writes the campaign report.  Resumable; stops at the first failure.
set -euo pipefail

REPO=${GLM53_COUPLED_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CAMPAIGN=${GLM53_CAMPAIGN_ROOT:-/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6}
OUT=$CAMPAIGN/phase1-campaign-v1
K4_ROOT=$CAMPAIGN/full-coupled-p8-all42-v1
RATE_ROOT=$CAMPAIGN/rate-candidates-v1
MIXED_ROOT=$CAMPAIGN/mixed-rate-p8-all42-v1
PROBES=$CAMPAIGN/group-probes-v1
ALLOCATION=$REPO/results/P8_LAYER_RATE_ALLOCATION_V1.json
EVIDENCE=$REPO/results/P8_RATE_SWAP_EVIDENCE_V1.json
REPORT=$REPO/results/P8_FULL_COUPLED_SHAPLEY_RESULT_V1.md
PY=/usr/bin/python3
SKIP_PROBES=${GLM53_SKIP_PROBES:-0}

cd "$REPO"
export PYTHONPATH="$REPO"
mkdir -p "$PROBES"
LOG=$PROBES/followon.log
log() { echo "[$(date --iso-8601=seconds)] $*" | tee -a "$LOG"; }
analysis() { echo "$OUT/$1-runtime-captures/analysis.json"; }

log "waiting for the phase 1 campaign to complete"
until grep -q "phase 1 campaign complete" "$OUT/campaign.log" 2>/dev/null; do
  # Bracket trick so the probe never matches its own command line.
  if ! pgrep -f "[r]un_phase1_campaign_v2.sh" >/dev/null; then
    sleep 5
    grep -q "phase 1 campaign complete" "$OUT/campaign.log" 2>/dev/null && break
    log "the campaign process is gone and the log has no completion line; stopping"
    exit 1
  fi
  sleep 120
done
log "campaign complete"

if [ "$SKIP_PROBES" != 1 ] && [ -f "$ALLOCATION" ]; then
  log "running the diagnostic group probes"
  bash scripts/run_group_probes.sh >>"$PROBES/probes-driver.log" 2>&1 || log "group probes failed; continuing to the report"
fi

log "building the rate-swap evidence table"
rm -f "$EVIDENCE"
args=(--output "$EVIDENCE" --uniform-analysis "$(analysis uniform-coupled-cf32-v1)"
      --allocated-analysis "$(analysis mixed-rate-cf32-v1)"
      --k3-only-analysis "$(analysis probe-k3-only)" --k5-only-analysis "$(analysis probe-k5-only)"
      --damage-dir "$RATE_ROOT/receipts" --k4-only-dir "$RATE_ROOT/receipts/k4-only")
[ -f "$ALLOCATION" ] && args+=(--allocation "$ALLOCATION")
"$PY" -m glm53_nvfp4.analyze_rate_swap_evidence "${args[@]}" | tee -a "$LOG"

log "writing the campaign report"
rm -f "$REPORT"
report_args=(--generated-at "$(date --iso-8601=seconds)" --output "$REPORT"
             --k4-manifest "$K4_ROOT/manifest.json" --uniform-analysis "$(analysis uniform-coupled-cf32-v1)"
             --damage-dir "$RATE_ROOT/receipts" --k4-only-dir "$RATE_ROOT/receipts/k4-only"
             --mixed-manifest "$MIXED_ROOT/manifest.json" --mixed-analysis "$(analysis mixed-rate-cf32-v1)"
             --closure-dir "$OUT")
[ -f "$ALLOCATION" ] && report_args+=(--allocation "$ALLOCATION")
"$PY" -m glm53_nvfp4.report_p8_campaign "${report_args[@]}" | tee -a "$LOG"
log "follow-on complete report=$REPORT evidence=$EVIDENCE"
