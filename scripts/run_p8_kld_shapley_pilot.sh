#!/usr/bin/env bash
set -euo pipefail

REPO=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53
CAMPAIGN=/media/brandonmusic/klcstore/bmxfp4-glm53
PLAN=$REPO/experiments/p8-kld-shapley-interaction-pilot-execution-v1.json
DESIGN=$REPO/experiments/p8-kld-shapley-interaction-pilot-v1.json
MANIFEST=$CAMPAIGN/codec-v2/kld-shapley-pilot-v1/overlay-manifest.json
RUN_ROOT=$CAMPAIGN/kld-v3
OUTPUT=$CAMPAIGN/codec-v2/kld-shapley-pilot-v1/analysis.json

mapfile -t ROWS < <(PYTHONPATH="$REPO" python3 - "$PLAN" "$DESIGN" "$MANIFEST" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

plan_path, design_path, manifest_path = map(Path, sys.argv[1:])
plan = json.loads(plan_path.read_text())
design = json.loads(design_path.read_text())
manifest = json.loads(manifest_path.read_text())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
if sha(design_path) != plan["inputs"]["design"]["sha256"]:
    raise SystemExit("pilot design hash mismatch")
if sha(manifest_path) != plan["inputs"]["overlay_manifest"]["sha256"]:
    raise SystemExit("overlay manifest hash mismatch")
addendum = plan_path.parent.parent / plan["inputs"]["preflight_addendum"]["path"]
if sha(addendum) != plan["inputs"]["preflight_addendum"]["sha256"]:
    raise SystemExit("preflight addendum hash mismatch")
addendum_v2 = plan_path.parent.parent / plan["inputs"]["preflight_addendum_v2"]["path"]
if sha(addendum_v2) != plan["inputs"]["preflight_addendum_v2"]["sha256"]:
    raise SystemExit("preflight addendum v2 hash mismatch")
if manifest["design_sha256"] != sha(design_path):
    raise SystemExit("overlay/design identity mismatch")
models = {row["coalition_id"]: row["model"] for row in manifest["coalitions"]}
if set(models) != set(design["coalitions"]):
    raise SystemExit("overlay matrix is incomplete")
for cid in plan["run_order"]:
    print(f"{cid}\t{models[cid]}")
PY
)

for ROW in "${ROWS[@]}"; do
  CID=${ROW%%$'\t'*}
  MODEL=${ROW#*$'\t'}
  RUN_ID=p8-kld-shapley-pilot-v1-$CID
  if [ -f "$RUN_ROOT/run-$RUN_ID.json" ] && grep -q '"status": "complete"' "$RUN_ROOT/run-$RUN_ID.json"; then
    continue
  fi
  GLM53_LOAD_FORMAT=instanttensor \
  GLM53_ROLES="$CAMPAIGN/roles/roles-codec-conditional-fit32-v1.json" \
  GLM53_RUNTIME_IMAGE=klc/glm53-flash-nvfp4:r19-sm120-tp4-ep4-dcp4-v79-dflash2-packed-aux-candidate \
  GLM53_RUNTIME_IMAGE_ID=sha256:ed027a3a2ff93b9cf60c95f7adfaf676cabc8e040a28cffa7486a262c82fdfbe \
  GLM53_RUNTIME_PATCH_MANIFEST="$CAMPAIGN/codec-v2/p8-joint-policy-runtime-manifest-v1.json" \
    "$REPO/scripts/run_kld_v3.sh" conditional-fit "$RUN_ID" "$MODEL" \
    "glm53-$RUN_ID" identity 3-44 "" "" "" humming
done

if [ ! -e "$OUTPUT" ]; then
  PYTHONPATH="$REPO" python3 -m glm53_nvfp4.p8_kld_shapley_analysis \
    --design "$DESIGN" --run-root "$RUN_ROOT" \
    --run-prefix p8-kld-shapley-pilot-v1 --output "$OUTPUT" \
    --bootstrap-replicates 50000 --bootstrap-seed 20260966
fi
