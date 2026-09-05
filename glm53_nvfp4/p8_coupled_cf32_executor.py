"""Three-arm CF32 executor with no production restoration path.

Import and ``seal`` are CPU-only. ``execute`` is the sole GPU/container entry
point and consumes both the prepared runtime manifest and an explicit seal.
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

from . import analyze_p8_coupled_cf32 as analysis
from . import p8_coupled_three_layer_runtime as runtime
from . import p8_decode_launcher as capture
from . import p8_decode_protocol as protocol
from . import p8_fc1_cold_compare as cold
from . import tail_v2_product_runner as product
from . import tail_v2_product_validation as validation
from .audit_speed_graphs import verify as verify_graphs


ARMS = ("stock", "identity_p8", "coupled_p8")
LOCK = Path("/run/lock/klc/model-stack.lock")
PORT = 8032
GLOBAL_BUDGET = 30_000_000_000
RAW_BYTES = validation.RAW_BYTES
CAPTURE_SUFFIXES = (".logits.f32", ".logits.f32.partial", ".capture.json.partial",
                    ".capture.inprogress.json", ".capture.failed.json")
SOURCE_FILES = (
    "glm53_nvfp4/p8_coupled_cf32_executor.py",
    "glm53_nvfp4/p8_coupled_three_layer_runtime.py",
    "glm53_nvfp4/analyze_p8_coupled_cf32.py",
    "glm53_nvfp4/p8_decode_protocol.py",
    "glm53_nvfp4/p8_decode_launcher.py",
    "glm53_nvfp4/tail_v2_product_validation.py",
    "glm53_nvfp4/p8_fc1_cold_compare.py",
    "scripts/prepare_p8_coupled_three_layer_cf32_runtime.py",
)
REPO = Path(__file__).resolve().parents[1]


def _save(path: Path, value) -> None:
    cold.pilot.save(path, value)


def _private_save(path: Path, value) -> None:
    cold.pilot.private_save(path, value)


def _production_off() -> dict:
    module = __import__("scripts.prepare_p8_coupled_three_layer_cf32_runtime",
                        fromlist=["production_off"])
    return module.production_off()


def global_capture_inventory(root: Path, exclude: Path | None = None) -> dict:
    root = Path(root)
    if root != root.resolve() or not root.is_dir():
        raise ValueError("global capture root must be a canonical directory")
    exclude = Path(exclude).resolve() if exclude is not None else None
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not path.name.endswith(CAPTURE_SUFFIXES):
            continue
        resolved = path.resolve()
        if path.is_symlink() or resolved != path:
            raise ValueError("global capture inventory refuses aliases")
        if exclude is not None and (resolved == exclude or resolved.is_relative_to(exclude)):
            continue
        stat = path.stat()
        rows.append({"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                     "inode": stat.st_ino, "device": stat.st_dev})
    return {"root": str(root), "bytes": sum(row["bytes"] for row in rows), "files": rows}


def authenticate_runtime_manifest(path: Path) -> tuple[dict, list[dict]]:
    path = Path(path)
    if path != path.resolve() or not path.is_file() or not path.with_suffix(".sha256").is_file():
        raise ValueError("canonical prepared manifest and seal required")
    if runtime.sha(path) != path.with_suffix(".sha256").read_text().split()[0]:
        raise ValueError("prepared runtime manifest seal differs")
    value = json.loads(path.read_text())
    if (value.get("schema") != "glm53.p8-coupled-three-layer-cf32-runtime.v1"
            or value.get("status") != "sealed-before-execution" or value.get("execution_authority") is not False
            or value.get("arms_in_order") != list(ARMS)
            or value.get("runtime") != {"tp": 4, "dcp": 1, "ep": False, "mtp": False, "graphs": True,
                "attention_backend": "B12X_MLA_SPARSE", "kv_dtype": "nvfp4_ds_mla",
                "forced_decode_rows": 2047, "capture_is_speed_valid": False,
                "sitecustomize": "/usr/lib/python3.12/sitecustomize.py", "pythonpath": runtime.V9_PYTHONPATH}):
        raise ValueError("prepared runtime protocol differs")
    attestation = Path(value["capture_attestation"]["path"])
    if runtime.sha(attestation) != value["capture_attestation"]["sha256"]:
        raise ValueError("capture attestation changed")
    image = runtime.validate_capture_image_receipt(attestation)
    if value.get("image") != image:
        raise ValueError("embedded capture attestation differs")
    carrier = Path(value["stock_carrier"]["path"])
    if (runtime.sha(carrier / "config.json") != value["stock_carrier"]["config_sha256"]
            or runtime.sha(carrier / "model.safetensors.index.json") != value["stock_carrier"]["index_sha256"]):
        raise ValueError("stock carrier identity differs")
    identity_input, coupled_input = value["identity_inputs"], value["coupled_inputs"]
    identity = runtime.validate_identity_sidecars(
        Path(identity_input["root"]), Path(identity_input["manifest"]["path"]), Path(identity_input["design"]["path"]))
    postwrites = {int(layer): Path(row["postwrite"]["path"]) for layer, row in value["coupled_evidence"].items()}
    loaders = {int(layer): Path(row["real_loader"]["path"]) for layer, row in value["coupled_evidence"].items()}
    coupled = runtime.validate_coupled_sidecars(
        Path(coupled_input["root"]), Path(coupled_input["design"]["path"]),
        Path(coupled_input["transform"]["path"]), postwrites, loaders)
    if identity != value["identity_sidecars"] or coupled != value["coupled_sidecars"]:
        raise ValueError("prepared sidecar inventory differs")
    for layer, row in value["coupled_evidence"].items():
        for key, source in (("postwrite", postwrites[int(layer)]), ("real_loader", loaders[int(layer)])):
            if runtime.sha(source) != row[key]["sha256"]:
                raise ValueError("coupled evidence identity differs")
    roles = Path(value["roles"]["path"])
    windows = protocol.load_role_inputs(roles, Path(value["roles"]["teacher_root"]), verify_teacher_bytes=True)
    if runtime.sha(roles) != runtime.ROLE_SHA256 or [row["id"] for row in windows] != value["roles"]["window_ids"]:
        raise ValueError("CF32 role/window identity differs")
    if runtime.sha(Path(value["source_recipe"]["path"])) != value["source_recipe"]["sha256"]:
        raise ValueError("source recipe identity differs")
    expected_ids = [row["id"] for row in windows]
    roots = set()
    for arm in ARMS:
        entry = value["arms"][arm]
        if entry["environment"] != runtime.arm_environment(arm, expected_ids):
            raise ValueError(f"{arm}: environment differs")
        argv_text = "\n".join(entry["launch_argv"])
        if (runtime.V9_IMAGE not in entry["launch_argv"] or "/runtime-patch" in argv_text
                or f"PYTHONPATH={runtime.V9_PYTHONPATH}" not in entry["launch_argv"]):
            raise ValueError(f"{arm}: image/runtime path differs")
        roots.add(str(Path(entry["capture_root"]).parent))
    if len(roots) != 1:
        raise ValueError("arm capture roots do not share one campaign root")
    return value, windows


def make_execution_seal(manifest_path: Path, seal_path: Path, global_root: Path) -> dict:
    manifest, windows = authenticate_runtime_manifest(manifest_path)
    seal_path, global_root = Path(seal_path), Path(global_root)
    output = Path(manifest["arms"]["stock"]["capture_root"]).parent
    if (seal_path != seal_path.resolve() or seal_path.exists() or seal_path.with_suffix(".sha256").exists()
            or output.exists() or not output.is_relative_to(global_root)):
        raise ValueError("fresh canonical seal/output below global capture root required")
    production = _production_off()
    inventory = global_capture_inventory(global_root, output)
    if shutil.disk_usage(global_root).free < GLOBAL_BUDGET:
        raise ValueError("global future-capture budget unavailable at seal time")
    value = {"schema": "glm53.p8-coupled-three-layer-cf32-execution-seal.v1",
        "status": "authorized-before-gpu-execution", "runtime_manifest": str(manifest_path),
        "runtime_manifest_sha256": runtime.sha(manifest_path), "output": str(output),
        "global_capture_root": str(global_root), "global_retained_capture_inventory": inventory,
        "global_new_capture_budget_bytes": GLOBAL_BUDGET, "one_window_raw_bytes": RAW_BYTES,
        "arm_order": list(ARMS), "window_order": [row["id"] for row in windows],
        "image_id": runtime.V9_IMAGE, "production": production,
        "capture_policy": "one window raw at a time; durable score/hash receipt before exact raw unlink; never dense32",
        "restoration_policy": "never start or restore production, including on error or signal",
        "source_sha256": {name: runtime.sha(REPO / name) for name in SOURCE_FILES},
        "analysis_source_sha256": runtime.sha(REPO / "glm53_nvfp4/analyze_p8_coupled_cf32.py"),
        "protected_roles_opened": []}
    _save(seal_path, value)
    _save(seal_path.with_suffix(".sha256"), runtime.sha(seal_path) + "  " + seal_path.name + "\n")
    return value


def authenticate_execution_seal(path: Path) -> tuple[dict, dict, list[dict]]:
    path = Path(path)
    if path != path.resolve() or not path.is_file() or runtime.sha(path) != path.with_suffix(".sha256").read_text().split()[0]:
        raise ValueError("execution seal identity differs")
    seal = json.loads(path.read_text())
    if (seal.get("schema") != "glm53.p8-coupled-three-layer-cf32-execution-seal.v1"
            or seal.get("status") != "authorized-before-gpu-execution" or seal.get("arm_order") != list(ARMS)
            or seal.get("global_new_capture_budget_bytes") != GLOBAL_BUDGET
            or seal.get("one_window_raw_bytes") != RAW_BYTES or seal.get("protected_roles_opened") != []
            or seal.get("image_id") != runtime.V9_IMAGE
            or seal.get("restoration_policy") != "never start or restore production, including on error or signal"):
        raise ValueError("execution protocol differs")
    manifest_path = Path(seal["runtime_manifest"])
    if runtime.sha(manifest_path) != seal["runtime_manifest_sha256"]:
        raise ValueError("runtime manifest changed after execution authorization")
    manifest, windows = authenticate_runtime_manifest(manifest_path)
    if seal["window_order"] != [row["id"] for row in windows]:
        raise ValueError("window order differs")
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
    if shutil.disk_usage(global_root).free < GLOBAL_BUDGET:
        raise ValueError("global future-capture budget unavailable")
    return seal, manifest, windows


def _capture_budget(seal: dict, output: Path) -> dict:
    retained = global_capture_inventory(Path(seal["global_capture_root"]), output)
    if retained != seal["global_retained_capture_inventory"]:
        raise ValueError("retained capture inventory changed during execution")
    active = []
    if output.exists():
        active = [path for path in output.rglob("*") if path.is_file() and path.name.endswith(CAPTURE_SUFFIXES)]
    active_bytes = sum(path.stat().st_size for path in active)
    if active_bytes > RAW_BYTES or retained["bytes"] + active_bytes + RAW_BYTES > retained["bytes"] + GLOBAL_BUDGET:
        raise ValueError("global capture budget exceeded")
    return {"retained_bytes": retained["bytes"], "active_bytes": active_bytes,
            "next_window_reserved_bytes": RAW_BYTES, "budget_bytes": GLOBAL_BUDGET}


def _owned_argv(template: list[str], out: Path, owner: str) -> list[str]:
    argv = list(template)
    if argv[:2] != ["docker", "create"] or "--cidfile" in argv or "--label" in argv:
        raise ValueError("launch template ownership fields differ")
    insert = argv.index("--network")
    argv[insert:insert] = ["--cidfile", str(out / "container.cid"), "--label", f"{cold.LABEL}={owner}"]
    return argv


def audit_runtime_log(text: str, arm: str, *, completed: list[str] | None = None) -> dict:
    for marker in ("Using V2 Model Runner", "tensor_parallel_size=4", "decode_context_parallel_size=1",
                   "speculative_config=None", "kv_cache_dtype=nvfp4_ds_mla", "quantization=modelopt_mixed"):
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
    if arm == "stock":
        runtime.verify_runtime_log(text, arm)
        backends = set(re.findall(r"Using '([^']+)' NvFp4 MoE backend", text))
        markers = sorted(set(re.findall(r"Detected ModelOpt ([A-Z0-9_]+) checkpoint", text)))
        if len(backends) != 1 or not markers or "NVFP4" not in markers:
            raise ValueError("stock MoE/activation log evidence missing or ambiguous")
        conditions = {"attention": "B12X_MLA_SPARSE", "kv_dtype": "nvfp4_ds_mla",
            "moe_backend": next(iter(backends)),
            "activation_precision": "runtime-emitted ModelOpt markers: " + ",".join(markers),
            "bpw": "stock NVFP4 checkpoint; aggregate physical bpw not emitted"}
    else:
        native = runtime.verify_runtime_log(text, arm)
        conditions = {"attention": "B12X_MLA_SPARSE", "kv_dtype": "nvfp4_ds_mla",
            "moe_backend": "native-p8-mxf8f6f4-n128-" + native["boundary"],
            "activation_precision": "E4M3 UE8M0_K32 at native P8 MMA boundaries",
            "bpw": 4.25 if arm == "identity_p8" else 4.253993422896774}
    if completed is not None:
        observed = re.findall(r"GLM53_P8_DECODE_CAPTURE_V2_COMPLETE window=(conditional-fit-\d{4}) rows=(\d+) tp_rank=(\d+)", text)
        if observed != [(window, "2047", "0") for window in completed]:
            raise ValueError("ordered 2047-row capture completion inventory differs")
    return {"arm": arm, "graph": graph, "conditions": conditions,
            "capture_rows": 2047, "true_decode_mask": {"exclude_rows": [0], "include": [1, 2047]}}


def score_retire_window(*, arm: str, window: dict, teacher_root: Path, capture_root: Path,
                        receipt_root: Path, runtime_audit_sha256: str) -> dict:
    if arm not in ARMS or not re.fullmatch(r"[0-9a-f]{64}", runtime_audit_sha256):
        raise ValueError("arm/runtime audit differs")
    wid = window["id"]
    if validation.storage_inventory(capture_root)["active_window_ids"] != [wid]:
        raise ValueError("exactly one requested raw capture required")
    tokens = np.load(Path(window["token_path"]), allow_pickle=False)
    student, metadata = protocol.load_capture(capture_root, wid, tokens)
    teacher_path = Path(teacher_root) / window["teacher_path"]
    if validation.sha(teacher_path) != window["teacher_sha256"]:
        raise ValueError("teacher identity changed")
    teacher = validation.metric._load_teacher(teacher_path, window)
    scores = protocol.score_aligned(teacher, student, tokens)
    if len(scores.kld) != 2047 or not np.isfinite(scores.kld).all():
        raise ValueError("score rows/finite gate failed")
    target = receipt_root / arm; target.mkdir(parents=True, mode=0o700, exist_ok=True)
    score_npz, score_json, retire_json = (target / f"{wid}.{suffix}" for suffix in
                                          ("scores.npz", "score.json", "retirement.json"))
    if any(path.exists() for path in (score_npz, score_json, retire_json)):
        raise ValueError("immutable score receipt already exists")
    validation._save_scores(score_npz, scores)
    raw_path, metadata_path = capture_root / f"{wid}.logits.f32", capture_root / f"{wid}.capture.json"
    record = {"schema": "glm53.p8-coupled-cf32-window-score.v1", "status": "complete", "arm": arm,
        "window_id": wid, "domain": window["domain"], "prediction_rows": 2047,
        "one_token_prefill_rows": 1, "true_decode_rows": 2046,
        "mean_kld": float(np.mean(scores.kld)), "true_decode_mean_kld": float(np.mean(scores.kld[1:])),
        "one_token_prefill_kld": float(scores.kld[0]), "raw_sha256": metadata["raw_sha256"],
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
    retirement = {"schema": "glm53.p8-coupled-cf32-window-retirement.v1", "status": "complete",
        "score_receipt_sha256": validation.sha(score_json), "retired_raw_sha256": record["raw_sha256"],
        "retired_raw_bytes": RAW_BYTES, "capture_metadata_retained": str(metadata_path)}
    validation._durable_json(retire_json, retirement)
    return {**record, "raw_retired": True, "retirement_sha256": validation.sha(retire_json)}


def _wait_ready(cid: str, model: str, tick) -> dict:
    deadline = time.monotonic() + 2400
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


def run_arm(seal: dict, manifest: dict, windows: list[dict], arm: str, owner: str,
            *, request=product._request, wait_ready=_wait_ready) -> dict:
    out = Path(manifest["arms"][arm]["capture_root"])
    out.mkdir(parents=True, mode=0o700)
    (out / "requests").mkdir(); (out / "captures").mkdir(); (out / "scores").mkdir()
    name = f"glm53-p8-three-layer-cf32-{arm}"
    image = manifest["image"]["image_id"]
    argv = _owned_argv(manifest["arms"][arm]["launch_argv"], out, owner)
    _private_save(out / "launch.private.json", {"argv": argv})
    record = {"schema": "glm53.p8-coupled-cf32-arm-execution.v1", "arm": arm, "exit_code": 1,
              "windows": [], "cleanup": {"ok": False}, "restoration_attempted": False,
              "production_checks": []}
    cid = None
    try:
        record["production_checks"].append(_production_off())
        _capture_budget(seal, Path(seal["output"]))
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
        models = wait_ready(cid, f"glm53-p8-three-layer-cf32-{arm}", tick)
        _save(out / "models.json", models)
        ready = cold.pilot.command(["docker", "logs", cid])
        ready_log = ready.stdout + ready.stderr
        _private_save(out / "server-ready.private.log", ready_log)
        runtime_path = out / "runtime-audit.json"
        for index, window in enumerate(windows):
            record["production_checks"].append(_production_off())
            _capture_budget(seal, Path(seal["output"]))
            if validation.storage_inventory(out / "captures")["raw_files"]:
                raise RuntimeError("prior raw was not retired")
            tokens = np.load(Path(window["token_path"]), allow_pickle=False)
            request_value = protocol.completion_request(tokens, window["id"], f"glm53-p8-three-layer-cf32-{arm}")
            _private_save(out / "requests" / f"{window['id']}.request.json", request_value)
            response = request(request_value, PORT, tick, 900)
            _private_save(out / "requests" / f"{window['id']}.response.json", response)
            response_audit = protocol.verify_response(response, request_value)
            capture.normalize_capture_ownership(out / "captures", [window], cid, image, name, owner,
                                                {"uid": os.getuid(), "gid": os.getgid()})
            if index == 0:
                logs = cold.pilot.command(["docker", "logs", cid]); proof_log = logs.stdout + logs.stderr
                _private_save(out / "runtime-proof.private.log", proof_log)
                _save(runtime_path, audit_runtime_log(proof_log, arm))
            score = score_retire_window(arm=arm, window=window, teacher_root=Path(manifest["roles"]["teacher_root"]),
                capture_root=out / "captures", receipt_root=out / "scores",
                runtime_audit_sha256=runtime.sha(runtime_path))
            record["windows"].append({"window_id": window["id"], "response": response_audit,
                                      "score_sha256": runtime.sha(out / "scores" / arm / f"{window['id']}.score.json"),
                                      "raw_sha256": score["raw_sha256"], "raw_retired": True})
        record["exit_code"] = 0
    except BaseException as error:
        record["error_type"] = type(error).__name__; _private_save(out / "error.private.txt", str(error))
    finally:
        ok, errors = cold.cleanup_owned(out, name, image, owner)
        record["cleanup"] = {"ok": ok, "errors": errors}
        if not ok:
            record["exit_code"] = 1
        if (out / "server-final.private.log").is_file() and record["exit_code"] == 0:
            try:
                final = (out / "server-final.private.log").read_text(errors="replace")
                _save(out / "runtime-final-audit.json", audit_runtime_log(final, arm, completed=[w["id"] for w in windows]))
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
    output = Path(manifest["arms"]["stock"]["capture_root"]).parent
    for arm in ARMS:
        root = output / arm
        audit = json.loads((root / "runtime-audit.json").read_text())
        rows = [json.loads((root / "scores" / arm / f"{window['id']}.score.json").read_text()) for window in windows]
        arms[arm] = {"conditions": audit["conditions"], "windows": rows}
    return analysis.analyze(windows, arms)


def execute(seal_path: Path, *, arm_runner=run_arm) -> dict:
    seal, manifest, windows = authenticate_execution_seal(seal_path)
    with LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        production_before = _production_off()
        output = Path(seal["output"]); output.mkdir(mode=0o700)
        record = {"schema": "glm53.p8-coupled-three-layer-cf32-execution.v1",
            "seal_sha256": runtime.sha(seal_path), "arm_order": list(ARMS), "completed_arms": [],
            "exit_code": 1, "restoration_attempted": False, "protected_roles_opened": [],
            "production_before": production_before, "root_lock": str(LOCK),
            "root_lock_mode": "LOCK_EX|LOCK_NB"}
        handlers = {}
        try:
            def interrupted(signum, _frame):
                raise RuntimeError(f"execution interrupted by signal {signum}")
            for sig in (signal.SIGTERM, signal.SIGHUP):
                handlers[sig] = signal.signal(sig, interrupted)
            if cold.pilot.command(["docker", "image", "inspect", runtime.V9_IMAGE, "--format", "{{.Id}}"]).stdout.strip() != runtime.V9_IMAGE:
                raise ValueError("immutable v9 image unavailable")
            _save(output / "hardware-inventory.json", cold.inventory())
            for arm in ARMS:
                arm_runner(seal, manifest, windows, arm, runtime.sha(seal_path) + ":" + arm)
                record["completed_arms"].append(arm)
            result = _analyze(manifest, windows); _save(output / "analysis.json", result)
            record["analysis_sha256"] = runtime.sha(output / "analysis.json"); record["exit_code"] = 0
        except BaseException as error:
            record["error_type"] = type(error).__name__; _private_save(output / "error.private.txt", str(error))
        finally:
            # Observation only. There is deliberately no stop/start/restore call.
            try:
                record["production_after"] = _production_off()
            except BaseException as error:
                record["production_after_error"] = type(error).__name__; record["exit_code"] = 1
            _save(output / "execution.json", record)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    if record["exit_code"]:
        raise RuntimeError("three-arm CF32 execution failed; partial evidence preserved, production remains off")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    seal = sub.add_parser("seal"); seal.add_argument("--manifest", type=Path, required=True); seal.add_argument("--seal", type=Path, required=True); seal.add_argument("--global-root", type=Path, required=True)
    run = sub.add_parser("execute"); run.add_argument("--seal", type=Path, required=True)
    args = parser.parse_args()
    value = make_execution_seal(args.manifest, args.seal, args.global_root) if args.command == "seal" else execute(args.seal)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
