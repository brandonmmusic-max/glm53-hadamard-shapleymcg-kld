#!/usr/bin/env bash
set -euo pipefail

REPO=${GLM53_REPO:-/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53}
RUNTIME_REPO=${GLM53_P8_RUNTIME_REPO:-/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-all42-runtime}
CAMPAIGN=${GLM53_CAMPAIGN_ROOT:-/media/brandonmusic/klcstore/bmxfp4-glm53}
ROOT=${GLM53_UNIFORM_P8_ROOT:-/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/uniform-p8-all42-v1}
BUILDER=${GLM53_UNIFORM_P8_SERVICE:-glm53-uniform-p8-all42-v1-r2.service}
PLAN=$REPO/experiments/p8-uniform-all42-fullmodel-kld-v1.json
PLAN_SEAL=$REPO/experiments/p8-uniform-all42-fullmodel-kld-v1.sha256
DESIGN=$RUNTIME_REPO/experiments/p8-kld-shapley-native6-v2.json
RUNTIME_PATCH=$RUNTIME_REPO/runtime_patch
RUNTIME_MANIFEST=$ROOT/runtime-patch-manifest-bb45c5b.json
ROLES=$CAMPAIGN/roles/roles-codec-conditional-fit32-v1.json
MODEL=$CAMPAIGN/codec-v2/p8-h128/identity-mcg-layer3-v1/candidate-bf16-pseudoquant
RUN_ID=p8-uniform-all42-native-cf32-v1
MODEL_NAME=glm53-p8-uniform-all42-native
CONTROL=$CAMPAIGN/kld-v3/records/p8-scaled-mcg-l3-control-cf32-v1
RECORDS=$CAMPAIGN/kld-v3/records/$RUN_ID
ANALYSIS=$ROOT/fullmodel-kld-vs-decoded-gptq-context.json
EXECUTION=$ROOT/fullmodel-kld-execution.json
LOG=$ROOT/fullmodel-kld-followon.log
LAYERS=$(seq -s, 3 44)

log() { echo "[$(date --iso-8601=seconds)] $*" | tee -a "$LOG"; }

log "waiting for builder=$BUILDER"
while systemctl --user is-active --quiet "$BUILDER"; do
  sleep 30
done
[ "$(systemctl --user show "$BUILDER" -p Result --value)" = success ] || {
  log "builder did not finish successfully; KLD will not start"
  exit 1
}
(cd "$(dirname "$PLAN")" && sha256sum --check --strict "$(basename "$PLAN_SEAL")") | tee -a "$LOG"

PYTHONPATH="$REPO" python3 - "$PLAN" "$DESIGN" "$RUNTIME_MANIFEST" "$ROLES" "$ROOT/manifest.json" "$EXECUTION" "$REPO/glm53_nvfp4/analyze_uniform_p8_fullmodel.py" "$RUNTIME_REPO" "$MODEL/BF16_LAYER_RECEIPT.json" "$CONTROL" <<'PY'
import hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

(
    plan_path, design, runtime_manifest, roles, checkpoint, execution, analyzer,
    runtime_repo, carrier_receipt, control,
) = map(Path, sys.argv[1:])
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

plan = json.loads(plan_path.read_text())
for path, expected, label in (
    (roles, plan["role"]["sha256"], "role"),
    (design, plan["runtime"]["design_sha256"], "design"),
    (runtime_manifest, plan["runtime"]["patch_manifest_sha256"], "runtime manifest"),
    (analyzer, plan["analysis"]["code_sha256"], "analysis code"),
    (carrier_receipt, plan["carrier"]["receipt_sha256"], "carrier receipt"),
):
    if sha(path) != expected:
        raise SystemExit(f"{label} identity mismatch")
runtime_commit = subprocess.check_output(
    ["git", "-C", str(runtime_repo), "rev-parse", "HEAD"], text=True
).strip()
if runtime_commit != plan["runtime"]["commit"]:
    raise SystemExit("runtime worktree commit mismatch")
role_payload = json.loads(roles.read_text())
expected_windows = [row["id"] for row in role_payload["roles"]["conditional-fit"]]
if len(expected_windows) != plan["role"]["windows"]:
    raise SystemExit("conditional-fit role count mismatch")
import pyarrow.parquet as pq
control_rows = []
control_values = {}
for path in sorted(control.glob("*.parquet")):
    frame = pq.read_table(path, columns=["window_id", "kld"]).to_pandas()
    ids = frame["window_id"].unique()
    if len(ids) != 1 or str(ids[0]) in control_values:
        raise SystemExit(f"invalid contextual control record: {path}")
    control_values[str(ids[0])] = float(frame["kld"].mean())
    control_rows.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha(path)})
control_canonical = hashlib.sha256(
    json.dumps(control_rows, separators=(",", ":"), sort_keys=True).encode()
).hexdigest()
if control_canonical != plan["paired_context_control"]["canonical_file_list_sha256"]:
    raise SystemExit("contextual control inventory mismatch")
if set(control_values) != set(expected_windows):
    raise SystemExit("contextual control does not contain the exact conditional-fit role")
if abs(sum(control_values.values()) / len(control_values) - plan["paired_context_control"]["mean_kld"]) > 1e-15:
    raise SystemExit("contextual control mean mismatch")
manifest = json.loads(checkpoint.read_text())
if (manifest.get("schema") != "glm53-p8-uniform-all42-tp4-checkpoint.v1"
        or manifest.get("layer_range") != [3, 44]
        or manifest.get("world_size") != 4
        or manifest.get("payload", {}).get("bpw") != 4.25
        or manifest.get("design", {}).get("sha256") != plan["runtime"]["design_sha256"]
        or [row.get("layer") for row in manifest.get("layers", [])] != list(range(3, 45))):
    raise SystemExit("all-layer P8 checkpoint contract mismatch")
for layer in manifest["layers"]:
    receipt = Path(layer["receipt"]["path"])
    if sha(receipt) != layer["receipt"]["sha256"]:
        raise SystemExit(f"receipt hash mismatch: {receipt}")
    if [rank.get("rank") for rank in layer.get("ranks", [])] != [0, 1, 2, 3]:
        raise SystemExit(f"TP4 inventory mismatch: layer {layer['layer']}")
    for rank in layer["ranks"]:
        sidecar = Path(rank["path"])
        if (sidecar.stat().st_size != rank["bytes"] or sha(sidecar) != rank["sha256"]
                or rank.get("payload_bpw") != 4.25):
            raise SystemExit(f"sidecar mismatch: {sidecar}")
static = {
    "schema": "glm53-p8-uniform-all42-kld-execution.v1",
    "plan_sha256": sha(plan_path),
    "checkpoint_manifest_sha256": sha(checkpoint),
    "runtime_manifest_sha256": sha(runtime_manifest),
    "runtime_commit": runtime_commit,
    "roles_sha256": sha(roles),
    "analysis_code_sha256": sha(analyzer),
    "context_control_inventory_sha256": control_canonical,
    "run_id": plan["run_id"],
    "sidecars_verified": 168,
    "protected_roles_opened": [],
}
if execution.exists():
    previous = json.loads(execution.read_text())
    if any(previous.get(key) != value for key, value in static.items()):
        raise SystemExit("existing execution receipt does not match this plan")
else:
    execution.write_text(json.dumps({**static, "started_at": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
print(json.dumps(static, sort_keys=True))
PY

PYTHONPATH="$REPO" python3 -m glm53_nvfp4.hash_tree \
  --root "$RUNTIME_PATCH" --output "$RUNTIME_MANIFEST" --verify | tee -a "$LOG"

log "starting full-model conditional-fit n=32 KLD run_id=$RUN_ID"
GLM53_ROLES="$ROLES" \
GLM53_REPO="$REPO" \
GLM53_RUNTIME_PATCH_ROOT="$RUNTIME_PATCH" \
GLM53_RUNTIME_PATCH_MANIFEST="$RUNTIME_MANIFEST" \
GLM53_RUNTIME_IMAGE_ID=sha256:ed027a3a2ff93b9cf60c95f7adfaf676cabc8e040a28cffa7486a262c82fdfbe \
GLM53_LOAD_FORMAT=instanttensor \
GLM53_P8_NATIVE=1 \
GLM53_P8_NATIVE_SIDECAR_DIR="$ROOT/sidecars" \
GLM53_P8_NATIVE_LAYERS="$LAYERS" \
GLM53_P8_NATIVE_DESIGN="$DESIGN" \
GLM53_DISABLE_EP=1 \
GLM53_DCP_SIZE=1 \
GLM53_B12X_DETERMINISTIC_OUTPUT=1 \
  "$REPO/scripts/run_kld_v3.sh" conditional-fit "$RUN_ID" "$MODEL" "$MODEL_NAME" identity "$LAYERS"

[ ! -e "$ANALYSIS" ] || { log "refusing to overwrite analysis=$ANALYSIS"; exit 1; }
PYTHONPATH="$REPO" python3 -m glm53_nvfp4.analyze_uniform_p8_fullmodel \
  --plan "$PLAN" --roles "$ROLES" --candidate-run "$RECORDS" \
  --context-control-run "$CONTROL" --output "$ANALYSIS" | tee -a "$LOG"
log "full-model KLD complete analysis=$ANALYSIS"
