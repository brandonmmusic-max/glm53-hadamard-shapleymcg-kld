#!/usr/bin/env bash
set -euo pipefail

REPO=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53
CAMPAIGN=/media/brandonmusic/klcstore/bmxfp4-glm53
ROOT=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-v9-large/had16-all
CANDIDATE=$ROOT/candidate
STOCK=/home/brandonmusic/models/GLM-5.3-Flash-NVFP4
ROLES=$CAMPAIGN/roles/roles-v3.json
BUILD=$ROOT/evidence/full-build.json
EVIDENCE=$CAMPAIGN/exact-v10/full-had16/evidence
FREEZE=$EVIDENCE/selection-wave2-freeze.json
CONDITIONAL=$EVIDENCE/conditional-fit-vs-stock.json
SELECTION=$EVIDENCE/selection-wave2-vs-stock.json
WINNER=$EVIDENCE/rotation-winner.json
RUN_ROOT=$CAMPAIGN/kld-v3

mkdir -p "$EVIDENCE"
cd "$REPO"
[ -f "$BUILD" ] || { echo "full H16 build is incomplete" >&2; exit 1; }

run_complete() {
  [ -f "$RUN_ROOT/run-$1.json" ] && grep -q '"status": "complete"' "$RUN_ROOT/run-$1.json"
}
run_kld() {
  GLM53_LOAD_FORMAT=instanttensor GLM53_ROTATION_PLACEMENT=humming-inner \
    "$REPO/scripts/run_kld_v3.sh" "$@"
}

CF_CANDIDATE=exact-v10-full-had16-qwen256-indexed
if ! run_complete "$CF_CANDIDATE"; then
  run_kld conditional-fit "$CF_CANDIDATE" "$CANDIDATE" \
    glm53-had16-qwen256-full had16 3-44 '' '' '' humming all
fi
python3 -m glm53_nvfp4.paired_role_analysis --role conditional-fit \
  --candidate-run "$RUN_ROOT/records/$CF_CANDIDATE" \
  --stock-run "$RUN_ROOT/records/exact-v8-stock-humming-indexed" \
  --roles "$ROLES" --output "$CONDITIONAL"
python3 - "$CONDITIONAL" <<'PY'
import json, sys
value=json.load(open(sys.argv[1]))
if value.get("decision") != "pass":
    raise SystemExit("full exact-Qwen H16 candidate did not pass conditional-fit")
PY

if [ ! -f "$FREEZE" ]; then
  python3 -m glm53_nvfp4.freeze_qwen_exact_h16 \
    --candidate "$CANDIDATE" --stock "$STOCK" --full-build "$BUILD" \
    --conditional-analysis "$CONDITIONAL" --roles "$ROLES" \
    --selection-wave 2 --output "$FREEZE"
fi

WAVE=2
STOCK_RUN=exact-v10-wave2-stock-indexed
CANDIDATE_RUN=exact-v10-wave2-had16-qwen256-indexed
if ! run_complete "$STOCK_RUN"; then
  run_kld selection "$STOCK_RUN" "$STOCK" glm53-wave2-stock \
    identity 3-44 '' "$WAVE" "$FREEZE" humming gate-up
fi
if ! run_complete "$CANDIDATE_RUN"; then
  run_kld selection "$CANDIDATE_RUN" "$CANDIDATE" glm53-wave2-had16-qwen256 \
    had16 3-44 '' "$WAVE" "$FREEZE" humming all
fi
python3 -m glm53_nvfp4.paired_role_analysis --role selection --selection-wave "$WAVE" \
  --candidate-run "$RUN_ROOT/records/$CANDIDATE_RUN" \
  --stock-run "$RUN_ROOT/records/$STOCK_RUN" --roles "$ROLES" --output "$SELECTION"

python3 - "$WINNER" "$CANDIDATE" "$FREEZE" "$SELECTION" <<'PY'
import hashlib, json, sys
from pathlib import Path
output, candidate, freeze_path, analysis_path = map(Path, sys.argv[1:])
analysis=json.loads(analysis_path.read_text())
payload={
    "schema":"glm53-nvfp4-v10.qwen-exact-h16-winner.v1",
    "status":"pass" if analysis.get("decision") == "pass" else "fail",
    "rotation":"had16",
    "rotation_scope":"all",
    "calibration_max_samples_per_expert":256,
    "model":str(candidate.resolve()),
    "required_load_format":"instanttensor",
    "runtime_rotation_placement":"humming-inner",
    "selection_wave":2,
    "analysis":str(analysis_path.resolve()),
    "analysis_sha256":hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
    "freeze":str(freeze_path.resolve()),
    "freeze_sha256":hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
    "mean_kld":analysis["candidate_mean_kld"],
    "stock_mean_kld":analysis["stock_mean_kld"],
    "mean_delta_kld":analysis["mean_delta_kld"],
    "delta_ci95_bca":analysis["delta_ci95_bca"],
}
output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(payload,indent=2,sort_keys=True))
if payload["status"] != "pass": raise SystemExit(3)
PY
