#!/usr/bin/env bash
# Diagnostic group probes for the same-size allocation, run AFTER the main campaign.
#
# The allocated model answers "is the swap worth it".  These two probes answer "why":
#   k3-only   the allocation's K3 layers downgraded, K5 layers left at K4  -> downgrade cost
#   k5-only   the allocation's K5 layers upgraded, K3 layers left at K4    -> upgrade benefit
# With the uniform K4 arm and the allocated arm already measured, the four points give the
# group-level interaction (the additivity retention measured on GLM-5.2 as 78.2%).
#
# Both probes reuse the candidate sidecars already packed and verified during scoring, so no
# encoding happens here.  k5-only is deliberately larger than the identity budget and is a
# diagnostic, never a shippable checkpoint; the same-size gate is waived for it only.
set -euo pipefail

REPO=${GLM53_COUPLED_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CAMPAIGN=${GLM53_CAMPAIGN_ROOT:-/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6}
OUT=$CAMPAIGN/phase1-campaign-v1
PROBES=$CAMPAIGN/group-probes-v1
ALLOCATION=${GLM53_RATE_ALLOCATION:-$REPO/results/P8_LAYER_RATE_ALLOCATION_V1.json}
PY=/usr/bin/python3
MAX_NEW_BYTES=${GLM53_MAX_NEW_BYTES:-400000000000}
DESIGN_K3=$REPO/results/P8_COUPLED_RATE_DESIGN_K3_V1.json
DESIGN_K5=$REPO/results/P8_COUPLED_RATE_DESIGN_K5_V1.json

mkdir -p "$PROBES"
cd "$REPO"
export PYTHONPATH="$REPO"
LOG=$PROBES/probes.log
log() { echo "[$(date --iso-8601=seconds)] $*" | tee -a "$LOG"; }

[ -f "$ALLOCATION" ] || { log "no allocation at $ALLOCATION"; exit 1; }
counts=$("$PY" - "$ALLOCATION" <<'PY'
import json, sys
a = json.load(open(sys.argv[1]))["assignment"]
print(sum(1 for v in a.values() if v == 3), sum(1 for v in a.values() if v == 5))
PY
)
read -r n3 n5 <<<"$counts"
log "allocation has $n3 K3 layers and $n5 K5 layers"
[ "$n3" -gt 0 ] || { log "no K3 layers in the allocation; nothing to probe"; exit 0; }

for probe in k3-only k5-only; do
  root=$PROBES/$probe
  alloc=$PROBES/$probe-allocation.json
  if [ "$probe" = k5-only ] && [ "$n5" -eq 0 ]; then log "allocation has no K5 layers; skipping k5-only"; continue; fi
  if [ ! -f "$alloc" ]; then
    "$PY" - "$ALLOCATION" "$alloc" "$probe" <<'PY'
import json, sys
src = json.load(open(sys.argv[1]))
keep = 3 if sys.argv[3] == "k3-only" else 5
assignment = {layer: (rate if rate == keep else 4) for layer, rate in src["assignment"].items()}
json.dump({"schema": src["schema"], "assignment": assignment, "derived_from": sys.argv[1],
           "probe": sys.argv[3],
           "note": "diagnostic single-arm probe of the allocation; not a shippable checkpoint"},
          open(sys.argv[2], "w"), indent=1, sort_keys=True)
PY
  fi
  if [ ! -f "$root/manifest.json" ]; then
    log "$probe: assembling (hard links only)"
    larger=0; [ "$probe" = k5-only ] && larger=1
    GLM53_RATE_ALLOCATION=$alloc GLM53_MIXED_RATE_ROOT=$root GLM53_ALLOW_LARGER=$larger \
      GLM53_RATE_DESIGN_K3=$DESIGN_K3 GLM53_RATE_DESIGN_K5=$DESIGN_K5 GLM53_MAX_NEW_BYTES=$MAX_NEW_BYTES \
      bash scripts/assemble_mixed_rate_p8_all42.sh >>"$PROBES/$probe-assemble.log" 2>&1
  fi
  if [ ! -f "$OUT/probe-$probe-runtime-captures/analysis.json" ]; then
    log "$probe: measuring CF32 KLD (32 windows)"
    GLM53_PROBE_NAME=probe-$probe GLM53_PROBE_MANIFEST=$root/manifest.json GLM53_PROBE_SIDECARS=$root/sidecars \
      bash scripts/run_probe_kld.sh >>"$PROBES/$probe-kld.log" 2>&1
  fi
  log "$probe complete"
done
log "group probes complete root=$PROBES"
