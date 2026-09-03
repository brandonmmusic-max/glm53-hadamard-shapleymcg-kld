#!/usr/bin/env bash
set -euo pipefail

REPO=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53
CAMPAIGN=/media/brandonmusic/klcstore/bmxfp4-glm53
DESIGN=$CAMPAIGN/shapley-v3/design.json
ROOT=$CAMPAIGN/shapley-v10-had16-all
MANIFEST=$ROOT/candidate-set.json
RUN_ROOT=$CAMPAIGN/kld-v3
PREFIX=shapley-v10-had16-all

[ -f "$MANIFEST" ] || { echo "missing frozen V10 Shapley candidate set" >&2; exit 1; }
mapfile -t ROWS < <(PYTHONPATH="$REPO" /usr/bin/python3 - "$DESIGN" "$MANIFEST" <<'PY'
import json, sys
design=json.load(open(sys.argv[1])); manifest=json.load(open(sys.argv[2]))
paths={row["coalition_id"]:row["path"] for row in manifest["candidates"]}
seen=set()
for permutation in design["permutations"]:
    for coalition_id in permutation["coalition_ids"]:
        if coalition_id not in seen:
            print(coalition_id+"\t"+paths[coalition_id])
            seen.add(coalition_id)
PY
)
for ROW in "${ROWS[@]}"; do
  CID=${ROW%%$'\t'*}
  MODEL=${ROW#*$'\t'}
  RUN_ID=$PREFIX-$CID
  if [ -f "$RUN_ROOT/run-$RUN_ID.json" ] && grep -q '"status": "complete"' "$RUN_ROOT/run-$RUN_ID.json"; then
    continue
  fi
  GLM53_LOAD_FORMAT=instanttensor GLM53_ROTATION_PLACEMENT=humming-inner \
    "$REPO/scripts/run_kld_v3.sh" conditional-fit "$RUN_ID" "$MODEL" \
      "glm53-$RUN_ID" had16 3-44 '' '' '' humming all
done
PYTHONPATH="$REPO" /usr/bin/python3 -m glm53_nvfp4.shapley_analysis \
  --design "$DESIGN" --run-root "$RUN_ROOT" --run-prefix "$PREFIX" \
  --output "$ROOT/analysis.json"
