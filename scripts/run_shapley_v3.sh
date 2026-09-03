#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "usage: $0 ROTATION CARRIER [ROTATION_FILE]" >&2
  exit 2
fi
ROTATION=$1
CARRIER=$(readlink -f "$2")
ROTATION_FILE=${3:-}
REPO=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53
CAMPAIGN=/media/brandonmusic/klcstore/bmxfp4-glm53
DESIGN=$CAMPAIGN/shapley-v3/design.json
MANIFEST=$CAMPAIGN/shapley-v3/candidate-set.json
RUN_ROOT=$CAMPAIGN/kld-v3

[ -f "$MANIFEST" ] || { echo "missing frozen Shapley candidate set" >&2; exit 1; }
mapfile -t ROWS < <(PYTHONPATH="$REPO" /usr/bin/python3 - <<PY
import json
d=json.load(open('$DESIGN'))
m=json.load(open('$MANIFEST'))
paths={x['coalition_id']:x['path'] for x in m['candidates']}
seen=set()
for p in d['permutations']:
  for cid in p['coalition_ids']:
    if cid not in seen:
      print(cid+'\t'+paths[cid])
      seen.add(cid)
PY
)
for ROW in "${ROWS[@]}"; do
  CID=${ROW%%$'\t'*}
  MODEL=${ROW#*$'\t'}
  RUN_ID=shapley-$CID
  if [ -f "$RUN_ROOT/run-$RUN_ID.json" ] && grep -q '"status": "complete"' "$RUN_ROOT/run-$RUN_ID.json"; then
    continue
  fi
  if [ "$ROTATION" = learned ]; then
    "$REPO/scripts/run_kld_v3.sh" conditional-fit "$RUN_ID" "$MODEL" "glm53-v3-$RUN_ID" learned 3-44 "$ROTATION_FILE"
  else
    "$REPO/scripts/run_kld_v3.sh" conditional-fit "$RUN_ID" "$MODEL" "glm53-v3-$RUN_ID" "$ROTATION" 3-44
  fi
done
PYTHONPATH="$REPO" /usr/bin/python3 -m glm53_nvfp4.shapley_analysis \
  --design "$DESIGN" --run-root "$RUN_ROOT" --output "$CAMPAIGN/shapley-v3/analysis.json"
