#!/usr/bin/env bash
set -euo pipefail

REPO=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53
CAMPAIGN=/media/brandonmusic/klcstore/bmxfp4-glm53
STOCK=/home/brandonmusic/models/GLM-5.3-Flash-NVFP4
CARRIER=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-v9-large/had16-all/candidate
ROLES=$CAMPAIGN/roles/roles-v3.json
WINNER=$CAMPAIGN/exact-v10/full-had16/evidence/rotation-winner.json
DESIGN=$CAMPAIGN/shapley-v3/design.json
CONTROL_DESIGN=$REPO/designs/shapley-equal-cost-uniform-depth-control-v1.json
MXROOT=$CAMPAIGN/mxfp6-v10-had16-all
CHUNKS=$MXROOT/chunks
ROOT=$CAMPAIGN/shapley-v10-had16-all
MANIFEST=$ROOT/candidate-set.json
ANALYSIS=$ROOT/analysis.json
FINAL=$ROOT/shapley-6bpw
UNIFORM_DEPTH=$ROOT/uniform-depth-6bpw
FREEZE=$ROOT/confirmation-freeze.json
REPORT=$ROOT/results.json
RUN_ROOT=$CAMPAIGN/kld-v3

cd "$REPO"
mkdir -p "$ROOT/evidence"
python3 - "$WINNER" <<'PY'
import json, sys
value=json.load(open(sys.argv[1]))
if value.get("status") != "pass" or value.get("rotation_scope") != "all":
    raise SystemExit("exact H16 rotation winner is not qualified")
PY

available=$(df -B1 --output=avail /media/brandonmusic/klcstore | tail -1 | tr -d ' ')
[ "$available" -ge 260000000000 ] || {
  echo "less than 260 GB free before corrected MXFP6 production" >&2
  exit 1
}
export GLM53_MXFP6_ROOT=$MXROOT

"$REPO/scripts/run_mxfp6_layer.sh" 3 had16
pilot_chunks=()
while IFS= read -r chunk; do pilot_chunks+=(--chunk "$chunk"); done \
  < <(find "$CHUNKS" -maxdepth 1 -type f -name 'mxfp6-layer-003-experts-*.safetensors' | sort)
[ "${#pilot_chunks[@]}" -eq 8 ] || { echo "MXFP6 pilot layer is incomplete" >&2; exit 1; }
PILOT=$ROOT/mxfp6-layer3-pilot
if [ ! -f "$PILOT/MIXED_RECEIPT.json" ]; then
  python3 -m glm53_nvfp4.mixed_candidate --carrier "$CARRIER" --output "$PILOT" \
    --mxfp6-layers 3 "${pilot_chunks[@]}"
fi
if [ ! -f "$ROOT/evidence/mxfp6-layer3-pilot-runtime/canary.json" ]; then
  "$REPO/scripts/run_candidate_canary.sh" "$PILOT" \
    "$ROOT/evidence/mxfp6-layer3-pilot-runtime" had16 3-44 '' all
fi

"$REPO/scripts/run_mxfp6_all.sh" had16 '' 4
[ "$(find "$CHUNKS" -maxdepth 1 -type f -name 'mxfp6-layer-*-experts-*.safetensors' | wc -l)" -eq 168 ] || {
  echo "corrected full MXFP6 chunk set is incomplete" >&2
  exit 1
}

if [ ! -f "$MANIFEST" ]; then
  python3 -m glm53_nvfp4.prepare_shapley_candidates \
    --design "$DESIGN" --carrier "$CARRIER" --chunk-root "$CHUNKS" \
    --output "$ROOT/candidates" --manifest "$MANIFEST"
fi
if [ ! -f "$ROOT/candidate-set-freeze.json" ]; then
  python3 -m glm53_nvfp4.freeze_shapley_inputs \
    --manifest "$MANIFEST" --design "$DESIGN" --control-design "$CONTROL_DESIGN" \
    --roles "$ROLES" --winner "$WINNER" --output "$ROOT/candidate-set-freeze.json"
fi
"$REPO/scripts/run_shapley_v10.sh"

mapfile -t SELECTED < <(python3 - "$ANALYSIS" <<'PY'
import json, sys
for layer in json.load(open(sys.argv[1]))["selected_mxfp6_layers"]: print(layer)
PY
)
mapfile -t CONTROL_SELECTED < <(python3 - "$CONTROL_DESIGN" <<'PY'
import json, sys
for layer in json.load(open(sys.argv[1]))["mxfp6_layers"]: print(layer)
PY
)

build_equal_cost() {
  local output=$1; shift
  local layers=("$@") layer layer3 layer_spec chunk
  local args=()
  layer_spec=$(IFS=,; echo "${layers[*]}")
  for layer in "${layers[@]}"; do
    printf -v layer3 '%03d' "$layer"
    while IFS= read -r chunk; do args+=(--chunk "$chunk"); done \
      < <(find "$CHUNKS" -maxdepth 1 -type f -name "mxfp6-layer-$layer3-experts-*.safetensors" | sort)
  done
  [ "${#args[@]}" -eq 280 ] || {
    echo "$output: expected 140 --chunk pairs, found ${#args[@]} array elements" >&2
    exit 1
  }
  python3 -m glm53_nvfp4.mixed_candidate --carrier "$CARRIER" --output "$output" \
    --mxfp6-layers "$layer_spec" "${args[@]}"
}
[ -f "$FINAL/MIXED_RECEIPT.json" ] || build_equal_cost "$FINAL" "${SELECTED[@]}"
[ -f "$UNIFORM_DEPTH/MIXED_RECEIPT.json" ] || build_equal_cost "$UNIFORM_DEPTH" "${CONTROL_SELECTED[@]}"

FULL_MXFP6=$(python3 - "$DESIGN" "$MANIFEST" <<'PY'
import json, sys
design=json.load(open(sys.argv[1])); manifest=json.load(open(sys.argv[2]))
full=next(cid for cid,layers in design["coalitions"].items() if len(layers)==len(design["layers"]))
print(next(row["path"] for row in manifest["candidates"] if row["coalition_id"]==full))
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

run_complete() {
  [ -f "$RUN_ROOT/run-$1.json" ] && grep -q '"status": "complete"' "$RUN_ROOT/run-$1.json"
}
run_confirmation() {
  local run_id=$1 model=$2 name=$3 rotation=$4 scope=$5
  if ! run_complete "$run_id"; then
    GLM53_LOAD_FORMAT=instanttensor GLM53_ROTATION_PLACEMENT=humming-inner \
      "$REPO/scripts/run_kld_v3.sh" confirmation "$run_id" "$model" "$name" \
        "$rotation" 3-44 '' '' "$FREEZE" humming "$scope"
  fi
}
run_confirmation exact-v10-confirmation-stock "$STOCK" glm53-confirmation-stock identity gate-up
run_confirmation exact-v10-confirmation-rotated "$CARRIER" glm53-confirmation-rotated had16 all
run_confirmation exact-v10-confirmation-uniform-depth "$UNIFORM_DEPTH" glm53-confirmation-uniform-depth had16 all
run_confirmation exact-v10-confirmation-shapley "$FINAL" glm53-confirmation-shapley had16 all
run_confirmation exact-v10-confirmation-full-mxfp6 "$FULL_MXFP6" glm53-confirmation-full-mxfp6 had16 all

for comparison in rotated uniform-depth stock; do
  python3 -m glm53_nvfp4.paired_role_analysis --role confirmation \
    --candidate-run "$RUN_ROOT/records/exact-v10-confirmation-shapley" \
    --stock-run "$RUN_ROOT/records/exact-v10-confirmation-$comparison" --roles "$ROLES" \
    --output "$ROOT/evidence/confirmation-shapley-vs-$comparison.json"
done
python3 -m glm53_nvfp4.paired_role_analysis --role confirmation \
  --candidate-run "$RUN_ROOT/records/exact-v10-confirmation-full-mxfp6" \
  --stock-run "$RUN_ROOT/records/exact-v10-confirmation-rotated" --roles "$ROLES" \
  --output "$ROOT/evidence/confirmation-full-mxfp6-vs-rotated.json"

python3 - "$REPORT" "$FREEZE" "$ANALYSIS" "$ROOT/evidence" <<'PY'
import hashlib, json, sys
from pathlib import Path
output, freeze_path, analysis_path, evidence_root = map(Path, sys.argv[1:])
names={
 "primary_shapley_vs_uniform_rotated":"confirmation-shapley-vs-rotated.json",
 "equal_cost_shapley_vs_uniform_depth":"confirmation-shapley-vs-uniform-depth.json",
 "secondary_shapley_vs_stock":"confirmation-shapley-vs-stock.json",
 "diagnostic_full_mxfp6_vs_uniform_rotated":"confirmation-full-mxfp6-vs-rotated.json",
}
comparisons={}
for key,name in names.items():
    path=evidence_root/name; value=json.loads(path.read_text())
    comparisons[key]={
      "candidate_mean_kld":value["candidate_mean_kld"], "control_mean_kld":value["stock_mean_kld"],
      "mean_delta_kld":value["mean_delta_kld"], "relative_improvement":value["relative_improvement"],
      "delta_ci95_bca":value["delta_ci95_bca"], "decision":value["decision"],
      "evidence":str(path.resolve()), "evidence_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
    }
freeze=json.loads(freeze_path.read_text()); analysis=json.loads(analysis_path.read_text())
payload={
 "schema":"glm53-nvfp4-v10.qwen-h16-shapley-6bpw-results.v1",
 "claim_boundary":"confirmation-level; no fresh final holdout exists",
 "payload_bpw":freeze["allocation"]["payload_bpw"],
 "selected_mxfp6_layers":freeze["allocation"]["mxfp6_layers"],
 "selected_nvfp4_layers":freeze["allocation"]["nvfp4_layers"],
 "shapley_efficiency_remainder":analysis["efficiency_remainder"],
 "shapley_top35_jaccard_two_paths":analysis["convergence"]["top35_jaccard_after_first_vs_final"],
 "comparisons":comparisons,
}
output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(payload,indent=2,sort_keys=True))
PY
