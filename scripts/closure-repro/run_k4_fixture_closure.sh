#!/usr/bin/env bash
# K4 grouped M64 control on the exact synthetic fixture the pinned K4 closure passed on (v9).
# If this passes on the rebuilt image, the rate parameterization did not regress K4 and the
# real-sidecar failures are a pre-existing gap in what that closure was ever validated against.
set -euo pipefail
REPO=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53/.claude/worktrees/fable-p8-full-coupled-v1
CAMP=/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6
FIX=$CAMP/p8-coupled-fixture-v1/rank0-v1
IMAGE=sha256:9846694e7dd3546f6ef3d259ecd2c129a5d54147662a76d32c5a3d15673f5e25
MANIFEST_SHA=acd55382242b3d79d6bdae242ff5268fb298f41c5f88aace121d4d9531afa89a
S=$FIX/p8-synthetic-layer-003-tp4-rank-0.safetensors
D=$FIX/synthetic-design.json
T=$REPO/experiments/p8-coupled-transform-draw0-silu10-v1.json
O=$CAMP/phase1-campaign-v1/grouped-k4-fixture
cd "$REPO"
export PYTHONPATH="$REPO"
sha() { sha256sum "$1" | cut -d' ' -f1; }
rm -rf "$O"
echo "fixture design sha: $(sha "$D")"
/usr/bin/python3 scripts/run_p8_mixed_rate_prefill_device_closure.py --execute \
  --image "$IMAGE" --gpu-device 0 \
  --sidecar "$S" --sidecar-sha256 "$(sha "$S")" \
  --design "$D" --design-sha256 "$(sha "$D")" \
  --transform "$T" --transform-sha256 "$(sha "$T")" \
  --runtime-manifest-sha256 "$MANIFEST_SHA" \
  --bits 4 --layer 3 --output "$O" >"$O-run.log" 2>&1 || true
/usr/bin/python3 - "$O/result.json" <<'PY'
import json, os, sys
d = json.load(open(sys.argv[1])) if os.path.exists(sys.argv[1]) else {}
print("K4 grouped on the synthetic fixture:", d.get("decision"),
      str((d.get("error") or {}).get("message"))[:140])
PY
