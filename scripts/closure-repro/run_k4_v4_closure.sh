#!/usr/bin/env bash
# K4 grouped M64 control on a layer encoded with the main V4 design (layer 4), rather than the
# pilot-design layer 3. This isolates whether the earlier K4 failure is about the kernels or
# about that one hard-linked pilot sidecar.
set -euo pipefail
REPO=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53/.claude/worktrees/fable-p8-full-coupled-v1
CAMP=/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6
IMAGE=sha256:9846694e7dd3546f6ef3d259ecd2c129a5d54147662a76d32c5a3d15673f5e25
MANIFEST_SHA=acd55382242b3d79d6bdae242ff5268fb298f41c5f88aace121d4d9531afa89a
S=$CAMP/full-coupled-p8-all42-v1/sidecars/p8-layer-004-tp4-rank-0.safetensors
D=$REPO/results/P8_COUPLED_SCALE_FLASH_EVIDENCE_PREPARATION_V4.json
T=$REPO/experiments/p8-coupled-transform-draw0-silu10-v1.json
O=$CAMP/phase1-campaign-v1/grouped-k4-v4design
cd "$REPO"
export PYTHONPATH="$REPO"
sha() { sha256sum "$1" | cut -d' ' -f1; }
rm -rf "$O"
/usr/bin/python3 scripts/run_p8_mixed_rate_prefill_device_closure.py --execute \
  --image "$IMAGE" --gpu-device 0 \
  --sidecar "$S" --sidecar-sha256 "$(sha "$S")" \
  --design "$D" --design-sha256 "$(sha "$D")" \
  --transform "$T" --transform-sha256 "$(sha "$T")" \
  --runtime-manifest-sha256 "$MANIFEST_SHA" \
  --bits 4 --layer 4 --output "$O" >"$O.log" 2>&1 || true
/usr/bin/python3 - "$O/result.json" <<'PY'
import json, os, sys
p = sys.argv[1]
d = json.load(open(p)) if os.path.exists(p) else {}
print("K4 grouped (V4 design, layer 4):", d.get("decision"),
      str((d.get("error") or {}).get("message"))[:140])
PY
