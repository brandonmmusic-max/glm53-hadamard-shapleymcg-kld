"""Two-arm full-model CF32 executor: coupled_full first, identity_full second.

``seal`` is CPU-only and binds the prepared runtime manifest, the executor
sources, the retained capture inventory and the storage ceiling.  ``execute`` is
the sole GPU/container entry point.  Raw logits are streamed one window at a
time and retired only after a durable score receipt; production is never
started or restored.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import time
from urllib.request import urlopen

import numpy as np

from . import analyze_p8_full_coupled_cf32 as analysis
from . import full_coupled_build_support as support
from . import p8_coupled_three_layer_runtime as carrier_receipts
from . import p8_decode_launcher as capture
from . import p8_decode_protocol as protocol
from . import p8_fc1_cold_compare as cold
from . import p8_full_coupled_runtime as runtime
from . import tail_v2_product_runner as product
from . import tail_v2_product_validation as validation
from .audit_speed_graphs import verify as verify_graphs
from .p8_coupled_cf32_executor import CAPTURE_SUFFIXES, _owned_argv, global_capture_inventory

ARMS = runtime.ARMS
LOCK = Path("/run/lock/klc/model-stack.lock")
PORT = runtime.PORT
RAW_BYTES = validation.RAW_BYTES
MIN_FREE_BYTES = 10_000_000_000
REPO = Path(__file__).resolve().parents[1]
MANIFEST_SCHEMA = "glm53.p8-full-coupled-cf32-runtime.v1"
SEAL_SCHEMA = "glm53.p8-full-coupled-cf32-execution-seal.v1"
EXECUTION_SCHEMA = "glm53.p8-full-coupled-cf32-execution.v1"
ARM_EXECUTION_SCHEMA = "glm53.p8-full-coupled-cf32-arm-execution.v1"
SCORE_SCHEMA = "glm53.p8-full-coupled-cf32-window-score.v1"
RETIREMENT_SCHEMA = "glm53.p8-full-coupled-cf32-window-retirement.v1"
SOURCE_FILES = (
    "glm53_nvfp4/p8_full_coupled_cf32_executor.py",
    "glm53_nvfp4/p8_full_coupled_runtime.py",
    "glm53_nvfp4/analyze_p8_full_coupled_cf32.py",
    "glm53_nvfp4/full_coupled_build_support.py",
    "scripts/prepare_p8_full_coupled_cf32_runtime.py",
    "experiments/p8-full-coupled-k4-cf32-v1.json",
)
RUNTIME_LABELS = {"tp": 4, "dcp": 1, "ep": False, "mtp": False, "graphs": True,
                  "attention_backend": "B12X_MLA_SPARSE", "kv_dtype": "nvfp4_ds_mla",
                  "forced_decode_rows": 2047, "capture_is_speed_valid": False,
                  "sitecustomize": "/usr/lib/python3.12/sitecustomize.py",
                  "pythonpath": runtime.V10_PYTHONPATH}
RESTORATION_POLICY = "never start or restore production, including on error or signal"


def _save(path: Path, value) -> None:
    cold.pilot.save(path, value)


def _private_save(path: Path, value) -> None:
    cold.pilot.private_save(path, value)


# ------------------------------------------------------------- authentication


def authenticate_runtime_manifest(path: Path, *, docker_inspect=None) -> tuple[dict, list[dict]]:
    path = Path(path)
    if path != path.resolve() or not path.is_file() or not path.with_suffix(".sha256").is_file():
        raise ValueError("canonical prepared manifest and seal required")
    if runtime.sha(path) != path.with_suffix(".sha256").read_text().split()[0]:
        raise ValueError("prepared runtime manifest seal differs")
    value = json.loads(path.read_text())
    arms = tuple(value.get("arms_in_order") or ())
    if (value.get("schema") != MANIFEST_SCHEMA or value.get("status") != "sealed-before-execution"
            or value.get("execution_authority") is not False or not arms or arms != ARMS[: len(arms)]
            or set(value.get("arms", {})) != set(arms)
            or (value.get("runtime") or {}).get("kv_dtype") not in runtime.KV_DTYPES
            or value.get("runtime") != {**RUNTIME_LABELS, "kv_dtype": value["runtime"]["kv_dtype"]}):
        raise ValueError("prepared runtime protocol differs")
    prereg = Path(value["preregistration"]["path"])
    if runtime.sha(prereg) != value["preregistration"]["sha256"] or json.loads(prereg.read_text()).get(
            "arms_in_order") != list(ARMS):
        raise ValueError("preregistration identity differs")
    image = runtime.validate_image_receipt(Path(value["image"]["receipt"]["path"]), docker_inspect=docker_inspect)
    if value.get("image") != image:
        raise ValueError("embedded image receipt differs")
    carrier = Path(value["stock_carrier"]["path"])
    if (runtime.sha(carrier / "config.json") != value["stock_carrier"]["config_sha256"]
            or runtime.sha(carrier / "model.safetensors.index.json") != value["stock_carrier"]["index_sha256"]):
        raise ValueError("stock carrier identity differs")
    stock_receipt = Path(value["stock_carrier"]["receipt"]["path"])
    extras_receipt = Path(value["stock_carrier"]["extras_receipt"]["path"])
    if (runtime.sha(stock_receipt) != value["stock_carrier"]["receipt"]["sha256"]
            or runtime.sha(extras_receipt) != value["stock_carrier"]["extras_receipt"]["sha256"]):
        raise ValueError("stock carrier receipt identity differs")
    stock = carrier_receipts.validate_stock_carrier_receipt(stock_receipt, carrier)
    extras = carrier_receipts.validate_stock_carrier_extras(extras_receipt, carrier)
    if (value["stock_carrier"].get("referenced_shard_count") != len(stock["shards"])
            or value["stock_carrier"].get("resolved_total_bytes") != stock["index"]["resolved_total_bytes"]
            or value["stock_carrier"].get("unreferenced_safetensors") != [row["name"] for row in extras["files"]]):
        raise ValueError("stock carrier shard inventory differs")
    identity_input, coupled_input = value.get("identity_inputs"), value["coupled_inputs"]
    identity = None
    if "identity_full" in arms:
        if not identity_input:
            raise ValueError("identity_full arm without identity inputs")
        identity = runtime.validate_identity_manifest(
            Path(identity_input["manifest"]["path"]), sidecar_dir=Path(identity_input["sidecar_dir"]),
            design=Path(identity_input["design"]["path"]))
        if (identity["manifest_sha256"] != identity_input["manifest"]["sha256"]
                or identity["files"] != identity_input["files"]
                or runtime.sha(Path(identity_input["design"]["path"])) != identity["design_sha256"]):
            raise ValueError("prepared identity checkpoint inventory differs")
    coupled = runtime.validate_full_coupled_manifest(
        Path(coupled_input["manifest"]["path"]), sidecar_dir=Path(coupled_input["sidecar_dir"]),
        transform=Path(coupled_input["transform"]["path"]))
    if (coupled["manifest_sha256"] != coupled_input["manifest"]["sha256"]
            or coupled["files"] != coupled_input["files"] or coupled["payload"] != coupled_input["payload"]):
        raise ValueError("prepared coupled checkpoint inventory differs")
    designs = {row["sha256"]: Path(row["path"]) for row in coupled_input["designs"]}
    if (set(designs) != set(coupled["design_by_layer"].values())
            or any(runtime.sha(path) != digest for digest, path in designs.items())):
        raise ValueError("coupled design files differ from the manifest")
    roles = Path(value["roles"]["path"])
    windows = protocol.load_role_inputs(roles, Path(value["roles"]["teacher_root"]), verify_teacher_bytes=True)
    if runtime.sha(roles) != runtime.ROLE_SHA256 or [row["id"] for row in windows] != value["roles"]["window_ids"]:
        raise ValueError("CF32 role/window identity differs")
    if runtime.sha(Path(value["source_recipe"]["path"])) != value["source_recipe"]["sha256"]:
        raise ValueError("source recipe identity differs")
    recipe = json.loads(Path(value["source_recipe"]["path"]).read_text())
    storage = value.get("storage", {})
    if (not isinstance(storage.get("max_new_bytes"), int) or storage["max_new_bytes"] <= 0
            or storage.get("one_window_raw_bytes") != RAW_BYTES
            or any(Path(p) != Path(p).resolve() for p in storage.get("campaign_paths", []))):
        raise ValueError("storage policy differs")
    expected_ids = [row["id"] for row in windows]
    roots = set()
    for arm in arms:
        entry = value["arms"][arm]
        if arm == "coupled_full":
            sidecars = Path(coupled_input["sidecar_dir"])
            arm_designs = [designs[digest] for digest in sorted(designs)]
            transform = Path(coupled_input["transform"]["path"])
            design_by_layer = coupled["design_by_layer"]
            bits_by_layer = coupled["bits_by_layer"]
        else:
            sidecars = Path(identity_input["sidecar_dir"])
            arm_designs = [Path(identity_input["design"]["path"])]
            transform = None
            design_by_layer = identity["design_by_layer"]
            bits_by_layer = {layer: 4 for layer in runtime.LAYERS}
        environment = runtime.arm_environment(arm, expected_ids, design_count=len(arm_designs))
        if entry["environment"] != environment:
            raise ValueError(f"{arm}: environment differs")
        if entry["design_by_layer"] != {str(layer): digest for layer, digest in design_by_layer.items()}:
            raise ValueError(f"{arm}: per-layer design map differs")
        if entry.get("bits_by_layer") != {str(layer): int(bits) for layer, bits in bits_by_layer.items()}:
            raise ValueError(f"{arm}: per-layer rate map differs")
        if any(bits != 4 for bits in bits_by_layer.values()) and image.get("lineage") != "v11":
            raise ValueError(f"{arm}: mixed-rate sidecars require the v11 runtime image")
        arm_root = Path(entry["capture_root"])
        if arm_root != arm_root.resolve() or arm_root.name != arm:
            raise ValueError(f"{arm}: capture root must be canonical and arm-named")
        expected_argv = runtime.launch_argv(recipe, image["image_id"], arm, environment, arm_root, carrier,
                                            sidecars, arm_designs, transform, kv_dtype=value["runtime"]["kv_dtype"])
        if entry["launch_argv"] != expected_argv:
            raise ValueError(f"{arm}: launch argv differs from authenticated source recipe")
        roots.add(str(arm_root.parent))
    if len(roots) != 1:
        raise ValueError("arm capture roots do not share one campaign root")
    return value, windows


def _storage_state(manifest: dict, output: Path) -> dict:
    storage = manifest["storage"]
    paths = [Path(p) for p in storage["campaign_paths"]]
    if output not in paths:
        paths.append(output)
    charged = sum(support.paths_apparent_bytes(paths).values())
    return {"campaign_paths": [str(p) for p in paths], "campaign_apparent_bytes": charged,
            "max_new_bytes": storage["max_new_bytes"], "one_window_raw_bytes": RAW_BYTES,
            "filesystem_free_bytes": shutil.disk_usage(output.parent if not output.exists() else output).free}


def _require_storage(state: dict, *, reserve_raw: bool) -> None:
    reservation = RAW_BYTES if reserve_raw else 0
    if state["campaign_apparent_bytes"] + reservation > state["max_new_bytes"]:
        raise ValueError("owner-approved campaign ceiling would be exceeded")
    if state["filesystem_free_bytes"] < RAW_BYTES + MIN_FREE_BYTES:
        raise ValueError("one raw capture plus the free-space margin does not fit")


def make_execution_seal(manifest_path: Path, seal_path: Path, global_root: Path, *, docker_inspect=None) -> dict:
    manifest, windows = authenticate_runtime_manifest(manifest_path, docker_inspect=docker_inspect)
    seal_path, global_root = Path(seal_path), Path(global_root)
    output = Path(manifest["arms"][ARMS[0]]["capture_root"]).parent
    if (seal_path != seal_path.resolve() or seal_path.exists() or seal_path.with_suffix(".sha256").exists()
            or output.exists() or not output.is_relative_to(global_root)):
        raise ValueError("fresh canonical seal/output below global capture root required")
    production = support.production_off()
    gpus = support.gpu_idle([0, 1, 2, 3])
    inventory = global_capture_inventory(global_root, output)
    state = _storage_state(manifest, output)
    _require_storage(state, reserve_raw=True)
    value = {"schema": SEAL_SCHEMA, "status": "authorized-before-gpu-execution",
             "runtime_manifest": str(manifest_path), "runtime_manifest_sha256": runtime.sha(manifest_path),
             "output": str(output), "global_capture_root": str(global_root),
             "global_retained_capture_inventory": inventory, "storage": state,
             "one_window_raw_bytes": RAW_BYTES, "arm_order": list(manifest["arms_in_order"]),
             "window_order": [row["id"] for row in windows], "image_id": manifest["image"]["image_id"],
             "production": production, "gpus_idle": gpus,
             "capture_policy": "one window raw at a time; durable score/hash receipt before exact raw unlink; never dense32",
             "restoration_policy": RESTORATION_POLICY,
             "source_sha256": {name: runtime.sha(REPO / name) for name in SOURCE_FILES},
             "protected_roles_opened": []}
    _save(seal_path, value)
    _save(seal_path.with_suffix(".sha256"), runtime.sha(seal_path) + "  " + seal_path.name + "\n")
    return value


def authenticate_execution_seal(path: Path, *, docker_inspect=None) -> tuple[dict, dict, list[dict]]:
    path = Path(path)
    if path != path.resolve() or not path.is_file() or runtime.sha(path) != path.with_suffix(".sha256").read_text().split()[0]:
        raise ValueError("execution seal identity differs")
    seal = json.loads(path.read_text())
    if (seal.get("schema") != SEAL_SCHEMA or seal.get("status") != "authorized-before-gpu-execution"
            or seal.get("one_window_raw_bytes") != RAW_BYTES
            or seal.get("protected_roles_opened") != [] or seal.get("restoration_policy") != RESTORATION_POLICY):
        raise ValueError("execution protocol differs")
    manifest_path = Path(seal["runtime_manifest"])
    if runtime.sha(manifest_path) != seal["runtime_manifest_sha256"]:
        raise ValueError("runtime manifest changed after execution authorization")
    manifest, windows = authenticate_runtime_manifest(manifest_path, docker_inspect=docker_inspect)
    if (seal["window_order"] != [row["id"] for row in windows] or seal["image_id"] != manifest["image"]["image_id"]
            or seal.get("arm_order") != list(manifest["arms_in_order"])):
        raise ValueError("window order, arm order or image differs")
    if set(seal["source_sha256"]) != set(SOURCE_FILES):
        raise ValueError("executor source inventory differs")
    for name, expected in seal["source_sha256"].items():
        if runtime.sha(REPO / name) != expected:
            raise ValueError(f"executor source differs: {name}")
    output, global_root = Path(seal["output"]), Path(seal["global_capture_root"])
    if output.exists() or not output.is_relative_to(global_root):
        raise ValueError("execution output must remain fresh and inside global root")
    if global_capture_inventory(global_root, output) != seal["global_retained_capture_inventory"]:
        raise ValueError("retained global capture inventory changed after authorization")
    _require_storage(_storage_state(manifest, output), reserve_raw=True)
    return seal, manifest, windows


def _capture_budget(seal: dict, manifest: dict, output: Path, *, reserve_next_raw: bool = True) -> dict:
    retained = global_capture_inventory(Path(seal["global_capture_root"]), output)
    if retained != seal["global_retained_capture_inventory"]:
        raise ValueError("retained capture inventory changed during execution")
    active = [path for path in output.rglob("*")
              if path.is_file() and path.name.endswith(CAPTURE_SUFFIXES)] if output.exists() else []
    active_bytes = sum(path.stat().st_size for path in active)
    if active_bytes > RAW_BYTES:
        raise ValueError("more than one fixed raw capture peak is active")
    if reserve_next_raw and active_bytes:
        raise ValueError("a prior raw/partial capture remains before the next request")
    if not reserve_next_raw and active_bytes != RAW_BYTES:
        raise ValueError("completed request did not produce exactly one fixed raw capture")
    state = _storage_state(manifest, output)
    _require_storage(state, reserve_raw=reserve_next_raw)
    return {"retained_capture_bytes": retained["bytes"], "active_capture_bytes": active_bytes,
            "campaign_apparent_bytes": state["campaign_apparent_bytes"],
            "ceiling_bytes": state["max_new_bytes"], "next_window_raw_bytes": RAW_BYTES}


# ------------------------------------------------------------------- auditing


def audit_runtime_log(text: str, arm: str, *, design_by_layer: dict[int, str], image_id: str, bpw: float,
                      completed: list[str] | None = None, bits_by_layer: dict[int, int] | None = None,
                      kv_dtype: str = "nvfp4_ds_mla") -> dict:
    if kv_dtype not in runtime.KV_DTYPES:
        raise ValueError(f"undeclared KV-cache dtype: {kv_dtype!r}")
    for marker in ("Using V2 Model Runner", "tensor_parallel_size=4", "decode_context_parallel_size=1",
                   "speculative_config=None", f"kv_cache_dtype={kv_dtype}", "quantization=modelopt_mixed"):
        if marker not in text:
            raise ValueError(f"runtime marker missing: {marker}")
    if "'enable_expert_parallel': True" in text or "enable_expert_parallel=True" in text:
        raise ValueError("unexpected expert parallel runtime")
    graph = verify_graphs(text)
    ready = re.findall(r"GLM53_P8_DECODE_CAPTURE_V2_READY tp_rank=(\d+) max_num_reqs=(\d+) real_vocab=(\d+) expected_outputs=(\d+)", text)
    closed = re.findall(r"GLM53_P8_DECODE_CAPTURE_V2_WARMUP_SCOPE_CLOSED tp_rank=(\d+) registrations=(\d+) samples=(\d+)", text)
    if set(ready) != {(str(rank), "1", str(protocol.VOCAB_LIMIT), "2047") for rank in range(4)} or len(ready) != 4:
        raise ValueError("capture ready-rank inventory differs")
    if set(closed) != {(str(rank), "1", "2") for rank in range(4)} or len(closed) != 4:
        raise ValueError("capture warmup closure differs")
    native = runtime.verify_runtime_log(text, arm, design_by_layer=design_by_layer, bits_by_layer=bits_by_layer)
    rates = sorted(set(bits_by_layer.values())) if bits_by_layer else [4]
    conditions = {"attention": "B12X_MLA_SPARSE", "kv_dtype": kv_dtype,
                  "moe_backend": "native-p8-mxf8f6f4-n128-" + native["boundary"],
                  "activation_precision": "E4M3 UE8M0_K32 at native P8 MMA boundaries",
                  "bpw": bpw, "layers": "3-44", "image_id": image_id,
                  "stored_rates": "/".join(f"K{b}" for b in rates)}
    if completed is not None:
        observed = re.findall(r"GLM53_P8_DECODE_CAPTURE_V2_COMPLETE window=(conditional-fit-\d{4}) rows=(\d+) tp_rank=(\d+)", text)
        if observed != [(window, "2047", "0") for window in completed]:
            raise ValueError("ordered 2047-row capture completion inventory differs")
    return {"arm": arm, "graph": graph, "native": native, "conditions": conditions,
            "capture_rows": 2047, "true_decode_mask": {"exclude_rows": [0], "include": [1, 2047]},
            "completed_window_ids": list(completed) if completed is not None else None}


def score_retire_window(*, arm: str, window: dict, teacher_root: Path, capture_root: Path,
                        receipt_root: Path, runtime_audit_sha256: str) -> dict:
    if arm not in ARMS or not re.fullmatch(r"[0-9a-f]{64}", runtime_audit_sha256):
        raise ValueError("arm/runtime audit differs")
    wid = window["id"]
    if validation.storage_inventory(capture_root)["active_window_ids"] != [wid]:
        raise ValueError("exactly one requested raw capture required")
    tokens = np.load(Path(window["token_path"]), allow_pickle=False)
    input_sha256 = runtime.sha(Path(window["token_path"]))
    token_values_sha256 = protocol.tokens_sha(tokens)
    if input_sha256 != window["input_sha256"] or token_values_sha256 != window["token_values_sha256"]:
        raise ValueError("token file/value identity changed before scoring")
    student, metadata = protocol.load_capture(capture_root, wid, tokens)
    teacher_path = Path(teacher_root) / window["teacher_path"]
    if validation.sha(teacher_path) != window["teacher_sha256"]:
        raise ValueError("teacher identity changed")
    teacher = validation.metric._load_teacher(teacher_path, window)
    scores = protocol.score_aligned(teacher, student, tokens)
    if len(scores.kld) != 2047 or not np.isfinite(scores.kld).all():
        raise ValueError("score rows/finite gate failed")
    target = receipt_root / arm
    target.mkdir(parents=True, mode=0o700, exist_ok=True)
    score_npz, score_json, retire_json = (target / f"{wid}.{suffix}" for suffix in
                                          ("scores.npz", "score.json", "retirement.json"))
    if any(path.exists() for path in (score_npz, score_json, retire_json)):
        raise ValueError("immutable score receipt already exists")
    validation._save_scores(score_npz, scores)
    raw_path, metadata_path = capture_root / f"{wid}.logits.f32", capture_root / f"{wid}.capture.json"
    record = {"schema": SCORE_SCHEMA, "status": "complete", "arm": arm,
              "window_id": wid, "domain": window["domain"], "prediction_rows": 2047,
              "one_token_prefill_rows": 1, "true_decode_rows": 2046,
              "mean_kld": float(np.mean(scores.kld)), "true_decode_mean_kld": float(np.mean(scores.kld[1:])),
              "one_token_prefill_kld": float(scores.kld[0]), "raw_sha256": metadata["raw_sha256"],
              "input_sha256": input_sha256, "token_values_sha256": token_values_sha256,
              "raw_bytes": RAW_BYTES, "capture_metadata_sha256": validation.sha(metadata_path),
              "scores_sha256": validation.sha(score_npz), "runtime_audit_sha256": runtime_audit_sha256,
              "raw_retired": False, "metric": "KL(teacher || student), FP64 CPU; true decode rows 1..2046"}
    validation._durable_json(score_json, record)
    del student, teacher, scores
    if (raw_path.is_symlink() or raw_path.resolve() != raw_path or raw_path.stat().st_size != RAW_BYTES
            or validation.sha(raw_path) != record["raw_sha256"]):
        raise ValueError("raw changed before authorized retirement")
    raw_path.unlink()
    descriptor = os.open(capture_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if validation.storage_inventory(capture_root)["raw_files"] != 0:
        raise ValueError("raw retirement did not close")
    retirement = {"schema": RETIREMENT_SCHEMA, "status": "complete",
                  "score_receipt_sha256": validation.sha(score_json), "retired_raw_sha256": record["raw_sha256"],
                  "retired_raw_bytes": RAW_BYTES, "capture_metadata_retained": str(metadata_path)}
    validation._durable_json(retire_json, retirement)
    return {**record, "raw_retired": True, "retirement_sha256": validation.sha(retire_json)}


def _wait_ready(cid: str, model: str, tick) -> dict:
    deadline = time.monotonic() + 3600
    while time.monotonic() < deadline:
        tick()
        try:
            with urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5) as response:
                value = json.load(response)
            if [row["id"] for row in value["data"]] == [model]:
                return value
        except OSError:
            pass
        time.sleep(2)
    raise RuntimeError("server readiness timeout")


# -------------------------------------------------------------------- running


def run_arm(seal: dict, manifest: dict, windows: list[dict], arm: str, owner: str,
            *, request=product._request, wait_ready=_wait_ready) -> dict:
    entry = manifest["arms"][arm]
    out = Path(entry["capture_root"])
    out.mkdir(parents=True, mode=0o700)
    (out / "requests").mkdir(); (out / "captures").mkdir(); (out / "scores").mkdir()
    name = runtime.SERVED_NAME.format(arm=arm)
    image = manifest["image"]["image_id"]
    design_by_layer = {int(layer): digest for layer, digest in entry["design_by_layer"].items()}
    bits_by_layer = {int(layer): int(bits) for layer, bits in entry["bits_by_layer"].items()}
    argv = _owned_argv(entry["launch_argv"], out, owner)
    _private_save(out / "launch.private.json", {"argv": argv})
    record = {"schema": ARM_EXECUTION_SCHEMA, "arm": arm, "exit_code": 1, "windows": [],
              "cleanup": {"ok": False}, "restoration_attempted": False, "production_checks": []}
    cid = None
    try:
        record["production_checks"].append(support.production_off())
        _capture_budget(seal, manifest, Path(seal["output"]))
        cold.pilot.command(argv)
        cid = (out / "container.cid").read_text().strip()
        if not re.fullmatch(r"[0-9a-f]{64}", cid):
            raise ValueError("created container ID differs")
        cold.pilot.command(["docker", "start", cid])

        def tick():
            if max(cold.pilot.temperatures()) >= 90:
                raise RuntimeError("thermal abort at 90C")
            state = json.loads(cold.pilot.command(["docker", "inspect", "-f", "{{json .State}}", cid]).stdout)
            if not state["Running"]:
                raise RuntimeError("owned server exited")

        models = wait_ready(cid, name, tick)
        _save(out / "models.json", models)
        ready = cold.pilot.command(["docker", "logs", cid])
        _private_save(out / "server-ready.private.log", ready.stdout + ready.stderr)
        runtime_path = out / "runtime-audit.json"
        for index, window in enumerate(windows):
            record["production_checks"].append(support.production_off())
            _capture_budget(seal, manifest, Path(seal["output"]))
            if validation.storage_inventory(out / "captures")["raw_files"]:
                raise RuntimeError("prior raw was not retired")
            tokens = np.load(Path(window["token_path"]), allow_pickle=False)
            input_sha256 = runtime.sha(Path(window["token_path"]))
            token_values_sha256 = protocol.tokens_sha(tokens)
            if input_sha256 != window["input_sha256"] or token_values_sha256 != window["token_values_sha256"]:
                raise ValueError("token file/value identity changed before request")
            request_value = protocol.completion_request(tokens, window["id"], name)
            _private_save(out / "requests" / f"{window['id']}.request.json", request_value)
            response = request(request_value, PORT, tick, 900)
            _private_save(out / "requests" / f"{window['id']}.response.json", response)
            response_audit = protocol.verify_response(response, request_value)
            _capture_budget(seal, manifest, Path(seal["output"]), reserve_next_raw=False)
            capture.normalize_capture_ownership(out / "captures", [window], cid, image, name, owner,
                                                {"uid": os.getuid(), "gid": os.getgid()})
            if index == 0:
                logs = cold.pilot.command(["docker", "logs", cid])
                proof_log = logs.stdout + logs.stderr
                _private_save(out / "runtime-proof.private.log", proof_log)
                _save(runtime_path, audit_runtime_log(proof_log, arm, kv_dtype=manifest["runtime"]["kv_dtype"],
                                                      design_by_layer=design_by_layer,
                                                      image_id=image, bpw=entry["bpw"], bits_by_layer=bits_by_layer))
            score = score_retire_window(arm=arm, window=window, teacher_root=Path(manifest["roles"]["teacher_root"]),
                                        capture_root=out / "captures", receipt_root=out / "scores",
                                        runtime_audit_sha256=runtime.sha(runtime_path))
            record["windows"].append({"window_id": window["id"], "response": response_audit,
                                      "score_sha256": runtime.sha(out / "scores" / arm / f"{window['id']}.score.json"),
                                      "raw_sha256": score["raw_sha256"], "raw_retired": True,
                                      "input_sha256": input_sha256, "token_values_sha256": token_values_sha256,
                                      "mean_kld": score["mean_kld"],
                                      "true_decode_mean_kld": score["true_decode_mean_kld"]})
            print(json.dumps({"arm": arm, "window": window["id"], "completed": index + 1, "of": len(windows),
                              "true_decode_mean_kld": score["true_decode_mean_kld"]}), flush=True)
        record["exit_code"] = 0
    except BaseException as error:
        record["error_type"] = type(error).__name__
        _private_save(out / "error.private.txt", str(error))
    finally:
        ok, errors = cold.cleanup_owned(out, name, image, owner)
        record["cleanup"] = {"ok": ok, "errors": errors}
        if not ok:
            record["exit_code"] = 1
        if record["exit_code"] == 0:
            try:
                final_path = out / "server-final.private.log"
                if not final_path.is_file():
                    raise ValueError("owned cleanup did not preserve the mandatory final runtime log")
                final = final_path.read_text(errors="replace")
                _save(out / "runtime-final-audit.json",
                      audit_runtime_log(final, arm, kv_dtype=manifest["runtime"]["kv_dtype"],
                                        design_by_layer=design_by_layer, image_id=image,
                                        bpw=entry["bpw"], completed=[w["id"] for w in windows],
                                        bits_by_layer=bits_by_layer))
            except BaseException as error:
                record["final_audit_error_type"] = type(error).__name__
                record["exit_code"] = 1
        record["completed_windows"] = len(record["windows"])
        _save(out / "execution.json", record)
    if record["exit_code"]:
        raise RuntimeError(f"{arm} failed; no next arm")
    return record


def _analyze(manifest: dict, windows: list[dict]) -> dict:
    arms = {}
    output = Path(manifest["arms"][ARMS[0]]["capture_root"]).parent
    expected_ids = [window["id"] for window in windows]
    for arm in manifest["arms_in_order"]:
        root = output / arm
        audit = json.loads((root / "runtime-audit.json").read_text())
        final_audit = json.loads((root / "runtime-final-audit.json").read_text())
        execution = json.loads((root / "execution.json").read_text())
        if (execution.get("schema") != ARM_EXECUTION_SCHEMA or execution.get("arm") != arm
                or execution.get("exit_code") != 0 or execution.get("cleanup") != {"ok": True, "errors": []}
                or execution.get("restoration_attempted") is not False
                or execution.get("completed_windows") != len(windows)
                or [row.get("window_id") for row in execution.get("windows", [])] != expected_ids
                or audit.get("arm") != arm or final_audit.get("arm") != arm
                or audit.get("conditions") != final_audit.get("conditions")
                or final_audit.get("completed_window_ids") != expected_ids
                or final_audit.get("true_decode_mask") != {"exclude_rows": [0], "include": [1, 2047]}):
            raise ValueError(f"{arm}: arm execution/runtime closure differs")
        execution_rows = {row["window_id"]: row for row in execution["windows"]}
        rows = []
        for window in windows:
            wid = window["id"]
            score_root = root / "scores" / arm
            score_path, score_npz, retirement_path = (score_root / f"{wid}.{suffix}" for suffix in
                                                      ("score.json", "scores.npz", "retirement.json"))
            score = json.loads(score_path.read_text())
            retirement = json.loads(retirement_path.read_text())
            metadata_path = Path(retirement.get("capture_metadata_retained", ""))
            raw_path = root / "captures" / f"{wid}.logits.f32"
            if (score.get("schema") != SCORE_SCHEMA or score.get("status") != "complete" or score.get("arm") != arm
                    or score.get("window_id") != wid or score.get("domain") != window["domain"]
                    or score.get("prediction_rows") != 2047 or score.get("one_token_prefill_rows") != 1
                    or score.get("true_decode_rows") != 2046 or score.get("raw_retired") is not False
                    or score.get("input_sha256") != window["input_sha256"]
                    or score.get("token_values_sha256") != window["token_values_sha256"]
                    or score.get("runtime_audit_sha256") != runtime.sha(root / "runtime-audit.json")
                    or score.get("scores_sha256") != runtime.sha(score_npz)
                    or score.get("capture_metadata_sha256") != runtime.sha(metadata_path)
                    or execution_rows[wid].get("score_sha256") != runtime.sha(score_path)
                    or execution_rows[wid].get("raw_sha256") != score.get("raw_sha256")
                    or execution_rows[wid].get("raw_retired") is not True
                    or retirement.get("schema") != RETIREMENT_SCHEMA or retirement.get("status") != "complete"
                    or retirement.get("score_receipt_sha256") != runtime.sha(score_path)
                    or retirement.get("retired_raw_sha256") != score.get("raw_sha256")
                    or retirement.get("retired_raw_bytes") != RAW_BYTES or raw_path.exists()):
                raise ValueError(f"{arm}/{wid}: score/retirement/hash closure differs")
            with np.load(score_npz, allow_pickle=False) as stored:
                kld = stored["kld"]
            if (kld.shape != (2047,) or not np.isfinite(kld).all()
                    or score["mean_kld"] != float(np.mean(kld))
                    or score["one_token_prefill_kld"] != float(kld[0])
                    or score["true_decode_mean_kld"] != float(np.mean(kld[1:]))):
                raise ValueError(f"{arm}/{wid}: persisted score statistics do not replay")
            rows.append(score)
        arms[arm] = {"conditions": audit["conditions"], "windows": rows}
    return analysis.analyze(windows, arms)


def execute(seal_path: Path, *, arm_runner=run_arm, docker_inspect=None) -> dict:
    seal, manifest, windows = authenticate_execution_seal(seal_path, docker_inspect=docker_inspect)
    with LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        production_before = support.production_off()
        output = Path(seal["output"])
        output.mkdir(mode=0o700)
        record = {"schema": EXECUTION_SCHEMA, "seal_sha256": runtime.sha(seal_path),
                  "arm_order": list(manifest["arms_in_order"]),
                  "completed_arms": [], "exit_code": 1, "restoration_attempted": False,
                  "protected_roles_opened": [], "production_before": production_before,
                  "root_lock": str(LOCK), "root_lock_mode": "LOCK_EX|LOCK_NB"}
        handlers = {}
        try:
            def interrupted(signum, _frame):
                raise RuntimeError(f"execution interrupted by signal {signum}")
            for sig in (signal.SIGTERM, signal.SIGHUP):
                handlers[sig] = signal.signal(sig, interrupted)
            image = manifest["image"]["image_id"]
            inspect = docker_inspect or (lambda name: cold.pilot.command(
                ["docker", "image", "inspect", name, "--format", "{{.Id}}"]).stdout.strip())
            if inspect(image) != image:
                raise ValueError("immutable v10 image unavailable")
            _save(output / "hardware-inventory.json", cold.inventory())
            for arm in manifest["arms_in_order"]:
                arm_runner(seal, manifest, windows, arm, runtime.sha(seal_path) + ":" + arm)
                record["completed_arms"].append(arm)
            final_manifest, final_windows = authenticate_runtime_manifest(Path(seal["runtime_manifest"]),
                                                                          docker_inspect=docker_inspect)
            if final_manifest != manifest or final_windows != windows:
                raise ValueError("runtime/model/sidecar/role identities changed across arms")
            result = _analyze(manifest, windows)
            _save(output / "analysis.json", result)
            record["analysis_sha256"] = runtime.sha(output / "analysis.json")
            record["exit_code"] = 0
        except BaseException as error:
            record["error_type"] = type(error).__name__
            _private_save(output / "error.private.txt", str(error))
        finally:
            # Observation only. There is deliberately no stop/start/restore call.
            try:
                record["production_after"] = support.production_off()
            except BaseException as error:
                record["production_after_error"] = type(error).__name__
                record["exit_code"] = 1
            _save(output / "execution.json", record)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    if record["exit_code"]:
        raise RuntimeError("full-model two-arm CF32 execution failed; partial evidence preserved, production remains off")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    seal = sub.add_parser("seal")
    seal.add_argument("--manifest", type=Path, required=True)
    seal.add_argument("--seal", type=Path, required=True)
    seal.add_argument("--global-root", type=Path, required=True)
    run = sub.add_parser("execute")
    run.add_argument("--seal", type=Path, required=True)
    args = parser.parse_args()
    value = (make_execution_seal(args.manifest, args.seal, args.global_root)
             if args.command == "seal" else execute(args.seal))
    print(json.dumps(value, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
