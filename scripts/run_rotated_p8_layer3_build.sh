#!/usr/bin/env bash
set -euo pipefail

REPO=${GLM53_ENCODER_REPO:-/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-rotation-v6-blocklocal-kld}
CAMPAIGN=${GLM53_CAMPAIGN_ROOT:-/media/brandonmusic/klcstore/bmxfp4-glm53}
ROOT=${GLM53_ROTATED_P8_ROOT:-/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/rotated-p8-mid-butterfly-l3-v1}
SOURCE=${GLM53_BF16_SOURCE:-$CAMPAIGN/downloads/GLM-5.3-Flash-BF16}
INDEX=$SOURCE/model.safetensors.index.json
CAPTURE=${GLM53_CAPTURE_ROOT:-$CAMPAIGN/teacher/calibration/main-ep4-full}
ROLES=${GLM53_FIT_ROLES:-$CAMPAIGN/roles/roles-v3.json}
DESIGN=${GLM53_ROTATED_P8_DESIGN:-$REPO/experiments/trellismx-p8-mid-butterfly-l3-interaction-v1.json}
CARRIER=${GLM53_NVFP4_CARRIER:-/home/brandonmusic/models/GLM-5.3-Flash-NVFP4}
PYTHON_BIN=${GLM53_ENCODER_PYTHON:-/home/brandonmusic/klc-env/bin/python}
LOCK=/run/lock/klc/model-stack.lock
THERMAL_PAUSE_C=${GLM53_P8_THERMAL_PAUSE_C:-94}
THERMAL_RESUME_C=${GLM53_P8_THERMAL_RESUME_C:-89}

[ -x "$PYTHON_BIN" ] || { echo "missing encoder Python: $PYTHON_BIN" >&2; exit 2; }
[ -f "$DESIGN" ] || { echo "missing rotated P8 design: $DESIGN" >&2; exit 2; }
[ ! -e "$ROOT/FULL_BUILD.json" ] || { echo "refusing completed destination: $ROOT" >&2; exit 2; }
[[ "$THERMAL_PAUSE_C" =~ ^[0-9]+$ ]] || exit 2
[[ "$THERMAL_RESUME_C" =~ ^[0-9]+$ ]] || exit 2
[ "$THERMAL_RESUME_C" -lt "$THERMAL_PAUSE_C" ] || exit 2
mkdir -p "$ROOT"/{codec,dense,receipts,logs,sidecars}
cd "$REPO"

exec 9>"$LOCK"
flock -w 900 9
backend_was_active=false
timer_was_active=false
systemctl --user is-active --quiet klc-backend.service && backend_was_active=true
systemctl is-active --quiet klc-model-stack.timer && timer_was_active=true
restore() {
  set +e
  [ "$backend_was_active" = false ] || systemctl --user start klc-backend.service
  [ "$timer_was_active" = false ] || sudo -n systemctl start klc-model-stack.timer
}
trap restore EXIT
[ "$timer_was_active" = false ] || sudo -n systemctl stop klc-model-stack.timer
[ "$backend_was_active" = false ] || systemctl --user stop klc-backend.service

pids=()
stopped=(0 0 0 0)
for slot in 0 1 2 3; do
  start=$((slot * 72)); end=$((start + 72))
  printf -v s3 '%03d' "$start"; printf -v e3 '%03d' "$end"
  codec=$ROOT/codec/rotated-p8-layer-003-experts-$s3-$e3.safetensors
  dense=$ROOT/dense/rotated-p8-layer-003-experts-$s3-$e3.safetensors
  receipt=$ROOT/receipts/rotated-p8-layer-003-experts-$s3-$e3.json
  for path in "$codec" "$dense" "$receipt"; do
    [ ! -e "$path" ] || { echo "refusing partial destination: $path" >&2; exit 2; }
  done
  CUDA_VISIBLE_DEVICES=$slot PYTHONPATH="$REPO" "$PYTHON_BIN" \
    -m glm53_nvfp4.quantize_rotated_hessian_trellis_layer \
    --source "$SOURCE" --source-index "$INDEX" --capture-root "$CAPTURE" \
    --roles "$ROLES" --design "$DESIGN" --layer 3 \
    --expert-start "$start" --expert-end "$end" --samples 256 --device cuda:0 \
    --dense-output "$dense" --codec-output "$codec" --receipt "$receipt" \
    >"$ROOT/logs/chunk-$s3-$e3.log" 2>&1 &
  pids+=("$!")
done

while :; do
  alive=0
  for slot in 0 1 2 3; do
    pid=${pids[$slot]}
    kill -0 "$pid" 2>/dev/null || continue
    alive=1
    temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits -i "$slot" | tr -d ' ')
    if [ "${stopped[$slot]}" -eq 0 ] && [ "$temp" -ge "$THERMAL_PAUSE_C" ]; then
      kill -STOP "$pid"
      stopped[$slot]=1
      printf '{"event":"pause","gpu":%d,"pid":%d,"temperature_c":%d}\n' "$slot" "$pid" "$temp" >>"$ROOT/thermal-events.jsonl"
    elif [ "${stopped[$slot]}" -eq 1 ] && [ "$temp" -le "$THERMAL_RESUME_C" ]; then
      kill -CONT "$pid"
      stopped[$slot]=0
      printf '{"event":"resume","gpu":%d,"pid":%d,"temperature_c":%d}\n' "$slot" "$pid" "$temp" >>"$ROOT/thermal-events.jsonl"
    fi
  done
  [ "$alive" -eq 1 ] || break
  sleep 5
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[ "$status" -eq 0 ] || { echo "one or more rotated P8 chunks failed" >&2; exit 1; }

chunk_args=()
overlay_args=()
for start in 0 72 144 216; do
  end=$((start + 72)); printf -v s3 '%03d' "$start"; printf -v e3 '%03d' "$end"
  chunk_args+=(--chunk "$ROOT/codec/rotated-p8-layer-003-experts-$s3-$e3.safetensors")
  overlay_args+=(--chunk "$ROOT/dense/rotated-p8-layer-003-experts-$s3-$e3.safetensors")
done
PYTHONPATH="$REPO" "$PYTHON_BIN" -m glm53_nvfp4.build_p8_tp4_sidecars \
  "${chunk_args[@]}" --layer 3 --world-size 4 --output-dir "$ROOT/sidecars" \
  --receipt "$ROOT/sidecars/receipt.json" >"$ROOT/logs/sidecars.log"
PYTHONPATH="$REPO" "$PYTHON_BIN" -m glm53_nvfp4.bf16_layer_candidate \
  --carrier "$CARRIER" --output "$ROOT/candidate" --bf16-layers 3 \
  "${overlay_args[@]}" >"$ROOT/logs/overlay.log"

PYTHONPATH="$REPO" "$PYTHON_BIN" - "$ROOT" "$DESIGN" "$REPO" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
root, design, repo = map(Path, sys.argv[1:])
def item(path):
    h=hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path":str(path.resolve()),"bytes":path.stat().st_size,"sha256":h}
receipts=[item(path) for path in sorted((root/'receipts').glob('*.json'))]
chunks=[item(path) for path in sorted((root/'codec').glob('*.safetensors'))]
dense=[item(path) for path in sorted((root/'dense').glob('*.safetensors'))]
if len(receipts)!=4 or len(chunks)!=4 or len(dense)!=4: raise SystemExit('incomplete chunk set')
rows=[json.loads(Path(row['path']).read_text()) for row in receipts]
if any(row['physical_payload_bpw'] != 4.25 for row in rows): raise SystemExit('rate mismatch')
if sum(row['expert_range'][1]-row['expert_range'][0] for row in rows) != 288: raise SystemExit('expert mismatch')
payload={
 "schema":"glm53-trellismx-p8.shared-mid-butterfly-layer-build.v1",
 "status":"complete",
 "layer":3,
 "experts":288,
 "git_commit":subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
 "design":item(design),
 "codec_chunks":chunks,
 "dense_chunks":dense,
 "chunk_receipts":receipts,
 "sidecar_receipt":item(root/'sidecars/receipt.json'),
 "overlay_receipt":item(root/'candidate/BF16_LAYER_RECEIPT.json'),
 "physical_payload_bpw":4.25,
 "rotation":{"id":"p00625","angle_pi":0.0625,"table_bytes":0},
 "protected_roles_opened":[],
 "ldlq":False,
}
(root/'FULL_BUILD.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(item(root/'FULL_BUILD.json')['sha256'])
PY
