#!/usr/bin/env bash
set -euo pipefail

REPO=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-rotation-v6-blocklocal-kld
CAMPAIGN=/media/brandonmusic/klcstore/bmxfp4-glm53
SOURCE=$CAMPAIGN/downloads/GLM-5.3-Flash-BF16
SOURCE_INDEX=$SOURCE/model.safetensors.index.json
CAPTURE=$CAMPAIGN/teacher/calibration/main-ep4-full
ROLES=$CAMPAIGN/roles/roles-v3.json
PLAN=${GLM53_BLOCKLOCAL_PLAN:-$REPO/experiments/rotation-v6-blocklocal-h16-layer3-cf32.json}
ROOT=${GLM53_BLOCKLOCAL_ROOT:-/media/brandonmusic/nvme1n1p3/glm53-rotation-v6/blocklocal-h16-layer3/full-all288}
CHUNKS=$ROOT/chunks
PHYSICAL=$ROOT/physical
RECEIPTS=$ROOT/receipts
LOGS=$ROOT/logs
CANDIDATE=$ROOT/candidate
LOCK=/run/lock/klc/model-stack.lock
PRODUCTION=glm53-flash-exl3-k4-tp4-vision-mtp3
PLAN_SHA=${GLM53_BLOCKLOCAL_PLAN_SHA:-00d2841389cca9f87aee67b7c20ead677e7c9e64d69dba0e68eae4507d634359}

[ "$(sha256sum "$PLAN" | cut -d' ' -f1)" = "$PLAN_SHA" ] || {
  echo "sealed plan hash mismatch" >&2
  exit 1
}
for required in "$SOURCE_INDEX" "$CAPTURE/capture-manifest.json" "$ROLES"; do
  [ -f "$required" ] || { echo "missing input: $required" >&2; exit 1; }
done
available=$(df -B1 --output=avail /media/brandonmusic/nvme1n1p3 | tail -1 | tr -d ' ')
[ "$available" -ge 30000000000 ] || { echo "less than 30 GB available on artifact volume" >&2; exit 1; }
mkdir -p "$CHUNKS" "$PHYSICAL" "$RECEIPTS" "$LOGS"

exec 9>"$LOCK"
flock -w 900 9
timer_was_active=false
backend_was_active=false
production_was_running=false
systemctl is-active --quiet klc-model-stack.timer && timer_was_active=true
systemctl --user is-active --quiet klc-backend.service && backend_was_active=true
[ "$(docker inspect -f '{{.State.Running}}' "$PRODUCTION" 2>/dev/null || true)" = true ] && production_was_running=true
restore() {
  set +e
  if [ "$production_was_running" = true ] && [ "$(docker inspect -f '{{.State.Running}}' "$PRODUCTION" 2>/dev/null || true)" != true ]; then
    docker start "$PRODUCTION" >/dev/null
  fi
  [ "$backend_was_active" = false ] || systemctl --user start klc-backend.service
  [ "$timer_was_active" = false ] || sudo -n systemctl start klc-model-stack.timer
}
trap restore EXIT
[ "$timer_was_active" = false ] || sudo -n systemctl stop klc-model-stack.timer
[ "$backend_was_active" = false ] || systemctl --user stop klc-backend.service
[ "$production_was_running" = false ] || docker stop --time 120 "$PRODUCTION" >/dev/null

cd "$REPO"
pids=()
for gpu in 0 1 2 3; do
  first=$((gpu * 72))
  last=$((first + 72))
  printf -v range '%03d-%03d' "$first" "$last"
  CUDA_VISIBLE_DEVICES=$gpu /home/brandonmusic/klc-env/bin/python -m glm53_nvfp4.quantize_blocklocal_h16_layer \
    --plan "$PLAN" --source "$SOURCE" --source-index "$SOURCE_INDEX" \
    --capture-root "$CAPTURE" --roles "$ROLES" \
    --expert-start "$first" --expert-end "$last" --device cuda:0 \
    --dense-output "$CHUNKS/blocklocal-h16-layer-003-experts-$range.safetensors" \
    --physical-output "$PHYSICAL/blocklocal-h16-layer-003-experts-$range.safetensors" \
    --receipt "$RECEIPTS/blocklocal-h16-layer-003-experts-$range.json" \
    >"$LOGS/gpu-$gpu.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[ "$failed" -eq 0 ] || { echo "one or more materialization workers failed" >&2; exit 1; }

chunk_args=()
for gpu in 0 1 2 3; do
  first=$((gpu * 72))
  last=$((first + 72))
  printf -v range '%03d-%03d' "$first" "$last"
  dense=$CHUNKS/blocklocal-h16-layer-003-experts-$range.safetensors
  physical=$PHYSICAL/blocklocal-h16-layer-003-experts-$range.safetensors
  receipt=$RECEIPTS/blocklocal-h16-layer-003-experts-$range.json
  /usr/bin/python3 -m glm53_nvfp4.verify_blocklocal_h16_chunk \
    --dense "$dense" --physical "$physical" --receipt "$receipt" \
    --output "$RECEIPTS/verification-$range.json" \
    >"$LOGS/verify-$gpu.log"
  chunk_args+=(--chunk "$dense")
done

/usr/bin/python3 -m glm53_nvfp4.bf16_layer_candidate \
  --carrier /home/brandonmusic/models/GLM-5.3-Flash-NVFP4 \
  --output "$CANDIDATE" --bf16-layers 3 "${chunk_args[@]}" \
  >"$LOGS/candidate-overlay.log"

/usr/bin/python3 - "$ROOT" "$PLAN" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, plan = map(Path, sys.argv[1:])
receipts=[]
verifications=[]
for first in (0,72,144,216):
    last=first+72
    receipts.append(json.loads((root/'receipts'/f'blocklocal-h16-layer-003-experts-{first:03d}-{last:03d}.json').read_text()))
    verifications.append(json.loads((root/'receipts'/f'verification-{first:03d}-{last:03d}.json').read_text()))
if any(item['status'] != 'pass' for item in verifications): raise SystemExit('verification did not pass')
if sum(item['logical_elements'] for item in receipts) != 288 * 25165824: raise SystemExit('logical coverage mismatch')
physical_bytes=sum(item['physical_bytes'] for item in receipts)
logical=sum(item['logical_elements'] for item in receipts)
candidate=root/'candidate'
payload={
  'schema':'glm53-rotation-v6.blocklocal-h16-layer3-full-build.v1',
  'status':'pass',
  'plan_sha256':hashlib.sha256(plan.read_bytes()).hexdigest(),
  'experts':288,
  'projections':['gate_proj','up_proj','down_proj'],
  'physical_bytes':physical_bytes,
  'logical_elements':logical,
  'physical_bpw':8*physical_bytes/logical,
  'descriptor_bytes':sum(item['descriptor_bytes'] for item in receipts),
  'candidate_receipt_sha256':hashlib.sha256((candidate/'BF16_LAYER_RECEIPT.json').read_bytes()).hexdigest(),
  'receipts':[hashlib.sha256((root/'receipts'/f'blocklocal-h16-layer-003-experts-{first:03d}-{first+72:03d}.json').read_bytes()).hexdigest() for first in (0,72,144,216)],
  'verifications':[hashlib.sha256((root/'receipts'/f'verification-{first:03d}-{first+72:03d}.json').read_bytes()).hexdigest() for first in (0,72,144,216)],
  'runtime_status':'BF16 pseudoquant overlay ready; indexed native prologue not implemented',
  'protected_roles_opened':[],
}
(root/'FULL_BUILD.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(json.dumps(payload,sort_keys=True))
PY
