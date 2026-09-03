#!/usr/bin/env bash
set -euo pipefail

REPO=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53
CAMPAIGN=/media/brandonmusic/klcstore/bmxfp4-glm53
STOCK=/home/brandonmusic/models/GLM-5.3-Flash-NVFP4
ROLES=$CAMPAIGN/roles/roles-v3.json
RECORD=$CAMPAIGN/experiment-record-v3.json
WINNER=$CAMPAIGN/evidence-v3/rotation-wave1-winner.json
DESIGN=$CAMPAIGN/shapley-v3/design.json
CONTROL_DESIGN=$REPO/designs/shapley-equal-cost-uniform-depth-control-v1.json
CHUNKS=$CAMPAIGN/mxfp6-v3/chunks
SHAPLEY_ROOT=$CAMPAIGN/shapley-v3
MANIFEST=$SHAPLEY_ROOT/candidate-set.json
ANALYSIS=$SHAPLEY_ROOT/analysis.json
FINAL=$CAMPAIGN/candidates-v3/shapley-6bpw
UNIFORM_DEPTH=$CAMPAIGN/candidates-v3/uniform-depth-6bpw
FREEZE=$CAMPAIGN/evidence-v3/shapley-6bpw-freeze.json
REPORT=$CAMPAIGN/evidence-v3/shapley-6bpw-results.json
RUN_ROOT=$CAMPAIGN/kld-v3

cd "$REPO"
run_complete() {
  [ -f "$RUN_ROOT/run-$1.json" ] && grep -q '"status": "complete"' "$RUN_ROOT/run-$1.json"
}
stage_recorded() {
  python3 - "$RECORD" "$1" <<'PY'
import json, sys
record = json.load(open(sys.argv[1]))
raise SystemExit(0 if any(row.get("stage") == sys.argv[2] for row in record["stage_receipts"]) else 1)
PY
}

mapfile -t WINNER_FIELDS < <(python3 - "$WINNER" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
if d.get("status") != "pass" or not d.get("winner"):
    raise SystemExit("rotation wave 1 has no passing winner")
w=d["winner"]
print(w["rotation"])
print(w["model"])
print(w.get("rotation_file") or "")
PY
)
ROTATION=${WINNER_FIELDS[0]}
CARRIER=${WINNER_FIELDS[1]}
ROTATION_FILE=${WINNER_FIELDS[2]}

# Full MXFP6 production is resumable per layer/expert range.  Require a
# conservative headroom gate before creating the roughly 238 GB tensor set.
available=$(df -B1 --output=avail /media/brandonmusic/klcstore | tail -1 | tr -d ' ')
[ "$available" -ge 300000000000 ] || { echo "less than 300 GB free before MXFP6 production" >&2; exit 1; }
"$REPO/scripts/run_mxfp6_layer.sh" 3 "$ROTATION" "$ROTATION_FILE"
pilot_chunks=()
while IFS= read -r chunk; do pilot_chunks+=(--chunk "$chunk"); done \
  < <(find "$CHUNKS" -maxdepth 1 -type f -name 'mxfp6-layer-003-experts-*.safetensors' | sort)
# Each discovered shard contributes the option and its value, hence eight
# array elements for the four required expert-range chunks.
[ "${#pilot_chunks[@]}" -eq 8 ] || { echo "MXFP6 pilot layer is incomplete" >&2; exit 1; }
PILOT=$CAMPAIGN/candidates-v3/mxfp6-layer3-pilot
PILOT_SESSION=$CAMPAIGN/evidence-v3/mxfp6-layer3-pilot-runtime
if [ ! -f "$PILOT/MIXED_RECEIPT.json" ]; then
  python3 -m glm53_nvfp4.mixed_candidate --carrier "$CARRIER" --output "$PILOT" \
    --mxfp6-layers 3 "${pilot_chunks[@]}"
fi
if ! stage_recorded mxfp6-layer3-runtime-pilot; then
  "$REPO/scripts/run_candidate_canary.sh" "$PILOT" "$PILOT_SESSION" "$ROTATION" 3-44 "$ROTATION_FILE"
  python3 -m glm53_nvfp4.stage_receipt "$RECORD" --stage mxfp6-layer3-runtime-pilot \
    --evidence "$PILOT/MIXED_RECEIPT.json" --evidence "$PILOT_SESSION/canary.json" \
    --evidence "$PILOT_SESSION/server-ready.log" --evidence "$PILOT_SESSION/backend-proof.log" \
    --note 'A one-layer native MXFP6 mixed checkpoint passed exact image loading, mixed-method binding marker, rotation marker, and deterministic coherent generation before bulk MXFP6 production.'
fi
"$REPO/scripts/run_mxfp6_all.sh" "$ROTATION" "$ROTATION_FILE" 3
[ "$(find "$CHUNKS" -maxdepth 1 -type f -name 'mxfp6-layer-*-experts-*.safetensors' | wc -l)" -eq 168 ] || {
  echo "full MXFP6 chunk set is incomplete" >&2
  exit 1
}

if [ ! -f "$MANIFEST" ]; then
  python3 -m glm53_nvfp4.prepare_shapley_candidates \
    --design "$DESIGN" --carrier "$CARRIER" --chunk-root "$CHUNKS" \
    --output "$SHAPLEY_ROOT/candidates" --manifest "$MANIFEST"
fi
if ! stage_recorded shapley-candidate-set-frozen; then
  python3 -m glm53_nvfp4.stage_receipt "$RECORD" --stage shapley-candidate-set-frozen \
    --evidence "$DESIGN" --evidence "$MANIFEST" --evidence "$CONTROL_DESIGN" \
    --note 'All 84 unique end-to-end coalition endpoints and the equal-cost uniform-depth confirmation control were frozen before Shapley target scoring.'
fi

"$REPO/scripts/run_shapley_v3.sh" "$ROTATION" "$CARRIER" "$ROTATION_FILE"

mapfile -t SELECTED < <(python3 - "$ANALYSIS" <<'PY'
import json, sys
for layer in json.load(open(sys.argv[1]))["selected_mxfp6_layers"]:
    print(layer)
PY
)
mapfile -t CONTROL_SELECTED < <(python3 - "$CONTROL_DESIGN" <<'PY'
import json, sys
for layer in json.load(open(sys.argv[1]))["mxfp6_layers"]:
    print(layer)
PY
)

build_equal_cost() {
  local output=$1
  shift
  local layers=("$@") layer layer3
  local layer_spec chunk
  local args=()
  layer_spec=$(IFS=,; echo "${layers[*]}")
  for layer in "${layers[@]}"; do
    printf -v layer3 '%03d' "$layer"
    while IFS= read -r chunk; do args+=(--chunk "$chunk"); done \
      < <(find "$CHUNKS" -maxdepth 1 -type f -name "mxfp6-layer-$layer3-experts-*.safetensors" | sort)
  done
  [ "${#args[@]}" -eq 140 ] || { echo "$output: expected 140 chunks, found ${#args[@]}" >&2; exit 1; }
  python3 -m glm53_nvfp4.mixed_candidate --carrier "$CARRIER" --output "$output" \
    --mxfp6-layers "$layer_spec" "${args[@]}"
}
[ -f "$FINAL/MIXED_RECEIPT.json" ] || build_equal_cost "$FINAL" "${SELECTED[@]}"
[ -f "$UNIFORM_DEPTH/MIXED_RECEIPT.json" ] || build_equal_cost "$UNIFORM_DEPTH" "${CONTROL_SELECTED[@]}"

FULL_MXFP6=$(python3 - "$DESIGN" "$MANIFEST" <<'PY'
import json, sys
d=json.load(open(sys.argv[1])); m=json.load(open(sys.argv[2]))
full=next(cid for cid, layers in d["coalitions"].items() if len(layers) == len(d["layers"]))
print(next(row["path"] for row in m["candidates"] if row["coalition_id"] == full))
PY
)

if [ ! -f "$FREEZE" ]; then
  python3 -m glm53_nvfp4.freeze_6bpw \
    --candidate "$FINAL" --uniform-depth-control "$UNIFORM_DEPTH" \
    --uniform-rotated "$CARRIER" --stock "$STOCK" --full-mxfp6 "$FULL_MXFP6" \
    --analysis "$ANALYSIS" --design "$DESIGN" --candidate-set "$MANIFEST" \
    --control-design "$CONTROL_DESIGN" --rotation-winner "$WINNER" --roles "$ROLES" \
    --run-root "$RUN_ROOT" --output "$FREEZE"
fi
if ! stage_recorded shapley-6bpw-frozen-before-confirmation; then
  python3 -m glm53_nvfp4.stage_receipt "$RECORD" --stage shapley-6bpw-frozen-before-confirmation \
    --evidence "$ANALYSIS" --evidence "$FINAL/MIXED_RECEIPT.json" \
    --evidence "$UNIFORM_DEPTH/MIXED_RECEIPT.json" --evidence "$FREEZE" \
    --note 'Shapley allocation, equal-cost control, exact byte accounting, implementation, and all confirmation endpoints frozen before opening confirmation.'
fi

rotation_args=("$ROTATION" 3-44 "$ROTATION_FILE" '' "$FREEZE")
run_complete confirmation-stock || "$REPO/scripts/run_kld_v3.sh" confirmation confirmation-stock "$STOCK" glm53-v3-confirmation-stock identity 3-44 '' '' "$FREEZE"
run_complete confirmation-rotated || "$REPO/scripts/run_kld_v3.sh" confirmation confirmation-rotated "$CARRIER" glm53-v3-confirmation-rotated "${rotation_args[@]}"
run_complete confirmation-uniform-depth || "$REPO/scripts/run_kld_v3.sh" confirmation confirmation-uniform-depth "$UNIFORM_DEPTH" glm53-v3-confirmation-uniform-depth "${rotation_args[@]}"
run_complete confirmation-shapley || "$REPO/scripts/run_kld_v3.sh" confirmation confirmation-shapley "$FINAL" glm53-v3-confirmation-shapley "${rotation_args[@]}"
run_complete confirmation-full-mxfp6 || "$REPO/scripts/run_kld_v3.sh" confirmation confirmation-full-mxfp6 "$FULL_MXFP6" glm53-v3-confirmation-full-mxfp6 "${rotation_args[@]}"

for comparison in rotated uniform-depth stock; do
  python3 -m glm53_nvfp4.paired_role_analysis --role confirmation \
    --candidate-run "$RUN_ROOT/records/confirmation-shapley" \
    --stock-run "$RUN_ROOT/records/confirmation-$comparison" --roles "$ROLES" \
    --output "$CAMPAIGN/evidence-v3/confirmation-shapley-vs-$comparison.json"
done
python3 -m glm53_nvfp4.paired_role_analysis --role confirmation \
  --candidate-run "$RUN_ROOT/records/confirmation-full-mxfp6" \
  --stock-run "$RUN_ROOT/records/confirmation-rotated" --roles "$ROLES" \
  --output "$CAMPAIGN/evidence-v3/confirmation-full-mxfp6-vs-rotated.json"

python3 - "$REPORT" "$FREEZE" "$ANALYSIS" "$CAMPAIGN/evidence-v3" <<'PY'
import hashlib, json, sys
from pathlib import Path
output, freeze_path, analysis_path, evidence_root = map(Path, sys.argv[1:])
names = {
    "primary_shapley_vs_uniform_rotated": "confirmation-shapley-vs-rotated.json",
    "equal_cost_shapley_vs_uniform_depth": "confirmation-shapley-vs-uniform-depth.json",
    "secondary_shapley_vs_stock": "confirmation-shapley-vs-stock.json",
    "diagnostic_full_mxfp6_vs_uniform_rotated": "confirmation-full-mxfp6-vs-rotated.json",
}
comparisons = {}
for key, name in names.items():
    path = evidence_root / name
    value = json.loads(path.read_text())
    comparisons[key] = {
        "candidate_mean_kld": value["candidate_mean_kld"],
        "control_mean_kld": value["stock_mean_kld"],
        "mean_delta_kld": value["mean_delta_kld"],
        "relative_improvement": value["relative_improvement"],
        "delta_ci95_bca": value["delta_ci95_bca"],
        "decision": value["decision"],
        "evidence": str(path.resolve()),
        "evidence_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
freeze = json.loads(freeze_path.read_text())
analysis = json.loads(analysis_path.read_text())
payload = {
    "schema": "glm53-nvfp4-v3.shapley-6bpw-results.v1",
    "claim_boundary": "confirmation-level; no fresh final holdout exists",
    "payload_bpw": freeze["allocation"]["payload_bpw"],
    "selected_mxfp6_layers": freeze["allocation"]["mxfp6_layers"],
    "selected_nvfp4_layers": freeze["allocation"]["nvfp4_layers"],
    "shapley_efficiency_remainder": analysis["efficiency_remainder"],
    "shapley_top35_jaccard_two_paths": analysis["convergence"]["top35_jaccard_after_first_vs_final"],
    "comparisons": comparisons,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
if ! stage_recorded confirmation-complete; then
  python3 -m glm53_nvfp4.stage_receipt "$RECORD" --stage confirmation-complete \
    --evidence "$REPORT" --note 'All five frozen confirmation endpoints completed on the exact 28-window role with paired end-to-end teacher KLD.'
fi
python3 /home/brandonmusic/.codex/skills/local-inference-lab-running-sealed-experiments/scripts/experiment_record.py validate "$RECORD" --strict
