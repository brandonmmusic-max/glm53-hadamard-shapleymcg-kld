#!/usr/bin/env bash
set -euo pipefail

REPO=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53
CAMPAIGN=/media/brandonmusic/klcstore/bmxfp4-glm53
STOCK=/home/brandonmusic/models/GLM-5.3-Flash-NVFP4
HAD16=$CAMPAIGN/candidates-v3/had16-full
LEARNED=$CAMPAIGN/candidates-v3/learned-full
ROTATIONS=$CAMPAIGN/rotations-v3/learned-full.safetensors
ROLES=$CAMPAIGN/roles/roles-v3.json
EVIDENCE=$CAMPAIGN/evidence-v3
RECORD=$CAMPAIGN/experiment-record-v3.json
FREEZE=$EVIDENCE/rotation-wave1-freeze-v2.json
WINNER=$EVIDENCE/rotation-wave1-winner.json

cd "$REPO"
run_complete() {
  [ -f "$CAMPAIGN/kld-v3/run-$1.json" ] && grep -q '"status": "complete"' "$CAMPAIGN/kld-v3/run-$1.json"
}
stage_recorded() {
  python3 - "$RECORD" "$1" <<'PY'
import json, sys
record = json.load(open(sys.argv[1]))
raise SystemExit(0 if any(row.get("stage") == sys.argv[2] for row in record["stage_receipts"]) else 1)
PY
}
count_passes() {
  find "$EVIDENCE/$1" -maxdepth 1 -type f -name "$1-layer-*-validation.json" \
    -exec grep -l '"status": "pass"' {} + | wc -l
}
[ "$(count_passes had16)" -eq 42 ] || { echo "H16 validation set is incomplete" >&2; exit 1; }
[ "$(count_passes learned)" -eq 42 ] || { echo "learned validation set is incomplete" >&2; exit 1; }
for path in "$HAD16/model.safetensors.index.json" "$LEARNED/model.safetensors.index.json" "$ROTATIONS"; do
  [ -f "$path" ] || { echo "missing full candidate artifact: $path" >&2; exit 1; }
done

if [ ! -f "$FREEZE" ]; then
  python3 -m glm53_nvfp4.freeze_rotation_wave \
    --stock "$STOCK" --had16 "$HAD16" --learned "$LEARNED" \
    --learned-rotations "$ROTATIONS" --roles "$ROLES" --campaign "$CAMPAIGN" \
    --output "$FREEZE"
fi
if ! stage_recorded rotation-wave1-v2-frozen; then
  python3 -m glm53_nvfp4.stage_receipt "$RECORD" --stage rotation-wave1-v2-frozen \
    --evidence "$FREEZE" --note 'Stock, H16, and learned full-model endpoints frozen before end-to-end scoring.'
fi

# Conditional-fit is development evidence: it catches runtime/recipe faults but
# cannot establish the primary win.  Every candidate uses the same 16 windows.
run_complete cf-stock || "$REPO/scripts/run_kld_v3.sh" conditional-fit cf-stock "$STOCK" glm53-v3-cf-stock identity 3-44
run_complete cf-had16 || "$REPO/scripts/run_kld_v3.sh" conditional-fit cf-had16 "$HAD16" glm53-v3-cf-had16 had16 3-44
run_complete cf-learned || "$REPO/scripts/run_kld_v3.sh" conditional-fit cf-learned "$LEARNED" glm53-v3-cf-learned learned 3-44 "$ROTATIONS"
python3 -m glm53_nvfp4.paired_role_analysis --role conditional-fit \
  --candidate-run "$CAMPAIGN/kld-v3/records/cf-had16" --stock-run "$CAMPAIGN/kld-v3/records/cf-stock" \
  --roles "$ROLES" --output "$EVIDENCE/conditional-fit-had16-vs-stock.json"
python3 -m glm53_nvfp4.paired_role_analysis --role conditional-fit \
  --candidate-run "$CAMPAIGN/kld-v3/records/cf-learned" --stock-run "$CAMPAIGN/kld-v3/records/cf-stock" \
  --roles "$ROLES" --output "$EVIDENCE/conditional-fit-learned-vs-stock.json"
if ! stage_recorded rotation-conditional-fit-complete; then
  python3 -m glm53_nvfp4.stage_receipt "$RECORD" --stage rotation-conditional-fit-complete \
    --evidence "$EVIDENCE/conditional-fit-had16-vs-stock.json" \
    --evidence "$EVIDENCE/conditional-fit-learned-vs-stock.json" \
    --note 'Conditional-fit completed before opening selection wave 1.'
fi

# Wave 1 is the first protected evidence for the rotation decision.  The stock
# endpoint is scored once and paired with each frozen candidate.
run_complete wave1-stock || "$REPO/scripts/run_kld_v3.sh" selection wave1-stock "$STOCK" glm53-v3-wave1-stock identity 3-44 '' 1
run_complete wave1-had16 || "$REPO/scripts/run_kld_v3.sh" selection wave1-had16 "$HAD16" glm53-v3-wave1-had16 had16 3-44 '' 1
run_complete wave1-learned || "$REPO/scripts/run_kld_v3.sh" selection wave1-learned "$LEARNED" glm53-v3-wave1-learned learned 3-44 "$ROTATIONS" 1
python3 -m glm53_nvfp4.paired_role_analysis --role selection --selection-wave 1 \
  --candidate-run "$CAMPAIGN/kld-v3/records/wave1-had16" --stock-run "$CAMPAIGN/kld-v3/records/wave1-stock" \
  --roles "$ROLES" --output "$EVIDENCE/selection-wave1-had16-vs-stock.json"
python3 -m glm53_nvfp4.paired_role_analysis --role selection --selection-wave 1 \
  --candidate-run "$CAMPAIGN/kld-v3/records/wave1-learned" --stock-run "$CAMPAIGN/kld-v3/records/wave1-stock" \
  --roles "$ROLES" --output "$EVIDENCE/selection-wave1-learned-vs-stock.json"

python3 - "$WINNER" "$HAD16" "$LEARNED" "$ROTATIONS" \
  "$EVIDENCE/selection-wave1-had16-vs-stock.json" \
  "$EVIDENCE/selection-wave1-learned-vs-stock.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output, had_path, learned_path, rotations_path, had_analysis, learned_analysis = map(Path, sys.argv[1:])
rows = []
for rotation, model, rotation_file, analysis_path in (
    ("had16", had_path, None, had_analysis),
    ("learned", learned_path, rotations_path, learned_analysis),
):
    analysis = json.loads(analysis_path.read_text())
    rows.append({
        "rotation": rotation,
        "model": str(model.resolve()),
        "rotation_file": str(rotation_file.resolve()) if rotation_file else None,
        "analysis": str(analysis_path.resolve()),
        "analysis_sha256": hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
        "decision": analysis["decision"],
        "mean_kld": analysis["candidate_mean_kld"],
        "mean_delta_kld": analysis["mean_delta_kld"],
        "delta_ci95_bca": analysis["delta_ci95_bca"],
    })
passing = [row for row in rows if row["decision"] == "pass"]
winner = min(passing, key=lambda row: (row["mean_kld"], row["rotation"])) if passing else None
payload = {
    "schema": "glm53-nvfp4-v3.rotation-wave1-winner.v1",
    "status": "pass" if winner else "no-winner",
    "rule": "mean paired KLD delta < 0 and 20000-resample BCa 95% upper delta < 0; lowest candidate mean wins",
    "candidates": rows,
    "winner": winner,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
if ! stage_recorded rotation-selection-wave1-complete; then
  python3 -m glm53_nvfp4.stage_receipt "$RECORD" --stage rotation-selection-wave1-complete \
    --evidence "$EVIDENCE/selection-wave1-had16-vs-stock.json" \
    --evidence "$EVIDENCE/selection-wave1-learned-vs-stock.json" --evidence "$WINNER" \
    --note 'Wave 1 decision uses the preregistered paired BCa rule; failed rows remain preserved.'
fi

python3 - "$WINNER" <<'PY'
import json, sys
if json.load(open(sys.argv[1]))["status"] != "pass":
    raise SystemExit(3)
PY
