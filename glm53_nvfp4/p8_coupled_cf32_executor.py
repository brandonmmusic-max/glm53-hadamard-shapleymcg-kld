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
import subprocess
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
LEDGER_SCHEMA = "glm53.p8-coupled-campaign-storage-ledger.v1"
LEDGER_CUTOFF = 1788632160
QUALITY_PEAK = 1_310_510_246
LEDGER_COMPONENTS = {
    "coupled_worktrees": "tree-apparent",
    "common_git_since_cutoff": "files-modified-since-cutoff",
    "coupled_outside_since_cutoff": "coupled-files-outside-since-cutoff",
    "docker_containerd_since_cutoff": "privileged-files-modified-since-cutoff",
    "image_build_directories": "tree-apparent",
    "fixture": "tree-apparent",
    "three_layer_outputs": "tree-apparent",
    "quality_root": "tree-apparent",
    "capture_output": "tree-apparent",
}
WORKSPACE = Path("/home/brandonmusic/KLC_SANDBOXES")
CAMPAIGN_ROOT = Path("/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6")
WORKTREES = tuple(WORKSPACE / name for name in (
    "bmxfp4-glm53-p8-coupled-image-v2", "bmxfp4-glm53-p8-coupled-input-order-v1",
    "bmxfp4-glm53-p8-coupled-integration-v1", "bmxfp4-glm53-p8-coupled-prefill-v1",
    "bmxfp4-glm53-p8-coupled-scale-kernel-v1", "bmxfp4-glm53-p8-coupled-scale-v1",
    "bmxfp4-glm53-p8-storage-ledger-refresh-v1"))
IMAGE_DIRS = tuple(CAMPAIGN_ROOT / name for name in (
    "p8-coupled-image-preparation-v1", *(f"p8-coupled-image-v{i}-build" for i in range(2, 10))))
FIXED_LEDGER_PATHS = {
    "coupled_worktrees": WORKTREES,
    "common_git_since_cutoff": (WORKSPACE / "bmxfp4-glm53/.git",),
    "coupled_outside_since_cutoff": (WORKSPACE,),
    "docker_containerd_since_cutoff": (Path("/var/lib/docker"), Path("/var/lib/containerd")),
    "image_build_directories": IMAGE_DIRS,
    "fixture": (CAMPAIGN_ROOT / "p8-coupled-fixture-v1",),
    "three_layer_outputs": (CAMPAIGN_ROOT / "p8-coupled-three-layer-v1",),
    "quality_root": (CAMPAIGN_ROOT / "tail-v2-p8-exl3-cf32-product-validation-v1b",),
}
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
    "scripts/prepare_p8_coupled_cf32_storage_ledger.py",
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


def _preparer():
    return __import__("scripts.prepare_p8_coupled_three_layer_cf32_runtime",
                      fromlist=["launch_argv"])


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


def _apparent_bytes(path: Path, *, cutoff_ns: int | None = None, missing_ok: bool = False) -> int:
    path = Path(path)
    if path != path.resolve() or path.is_symlink():
        raise ValueError("storage ledger paths must be canonical and not symlinks")
    if not path.exists():
        if missing_ok:
            return 0
        raise ValueError(f"storage ledger path missing: {path}")
    if path.is_file():
        info = path.stat()
        return info.st_size if cutoff_ns is None or info.st_mtime_ns >= cutoff_ns else 0
    total = 0 if cutoff_ns is not None else path.lstat().st_size
    def fail(error):
        raise error
    for base, directories, files in os.walk(path, followlinks=False, onerror=fail):
        base_path = Path(base)
        for name in directories + files:
            child = base_path / name
            info = child.lstat()
            if cutoff_ns is None or (child.is_file() and info.st_mtime_ns >= cutoff_ns):
                total += info.st_size
    return total


def _privileged_modified_bytes(paths: list[Path]) -> int:
    command = ["sudo", "-n", "find", *map(str, paths), "-xdev", "-type", "f",
               "-newermt", f"@{LEDGER_CUTOFF}", "-printf", "%s\\n"]
    result = subprocess.run(command, check=True, text=True, capture_output=True, timeout=180)
    values = result.stdout.splitlines()
    if any(not value.isdigit() for value in values):
        raise ValueError("privileged storage inventory emitted non-numeric size")
    return sum(map(int, values))


def _coupled_outside_bytes() -> int:
    exclusions = [*WORKTREES, WORKSPACE / "bmxfp4-glm53/.git"]
    command = ["sudo", "-n", "find", str(WORKSPACE), "-xdev", "("]
    for index, path in enumerate(exclusions):
        if index:
            command.append("-o")
        command += ["-path", str(path)]
    command += [")", "-prune", "-o", "-type", "f", "-newermt", f"@{LEDGER_CUTOFF}",
                "-ipath", "*coupled*", "-printf", "%s\\n"]
    result = subprocess.run(command, check=True, text=True, capture_output=True, timeout=180)
    values = result.stdout.splitlines()
    if any(not value.isdigit() for value in values):
        raise ValueError("privileged outside-coupled inventory emitted non-numeric size")
    return sum(map(int, values))


def _measure_component(name: str, mode: str, paths: list[Path], output: Path) -> int:
    expected_paths = ((Path(output),) if name == "capture_output" else FIXED_LEDGER_PATHS[name])
    if tuple(paths) != expected_paths:
        raise ValueError(f"storage ledger path scope differs: {name}")
    cutoff_ns = LEDGER_CUTOFF * 1_000_000_000 if mode == "files-modified-since-cutoff" else None
    if mode == "privileged-files-modified-since-cutoff":
        return _privileged_modified_bytes(paths)
    if mode == "coupled-files-outside-since-cutoff":
        return _coupled_outside_bytes()
    return sum(_apparent_bytes(path, cutoff_ns=cutoff_ns,
                               missing_ok=name == "capture_output") for path in paths)


def storage_ledger_snapshot(value: dict, output: Path) -> dict:
    if value.get("cutoff_unix") != LEDGER_CUTOFF:
        raise ValueError("storage ledger cutoff differs")
    rows = value.get("components")
    if not isinstance(rows, list) or {row.get("name") for row in rows} != set(LEDGER_COMPONENTS):
        raise ValueError("storage ledger component scope differs")
    measured = []
    for row in rows:
        name, mode = row.get("name"), row.get("mode")
        if (mode != LEDGER_COMPONENTS[name] or not isinstance(row.get("baseline_bytes"), int)
                or not isinstance(row.get("projected_bytes"), int)
                or row["projected_bytes"] < row["baseline_bytes"]):
            raise ValueError(f"storage ledger component contract differs: {name}")
        paths = [Path(path) for path in row.get("paths", [])]
        if not paths or len(set(paths)) != len(paths):
            raise ValueError(f"storage ledger component paths differ: {name}")
        current = _measure_component(name, mode, paths, output)
        measured.append({"name": name, "mode": mode, "baseline_bytes": row["baseline_bytes"],
                         "projected_bytes": row["projected_bytes"], "current_bytes": current,
                         "positive_growth_bytes": max(0, current - row["projected_bytes"])})
    return {"components": measured,
            "positive_growth_bytes": sum(row["positive_growth_bytes"] for row in measured)}


def validate_storage_ledger(path: Path, output: Path, *, active_capture_bytes: int = 0,
                            require_fresh: bool = True, reserve_next_raw: bool = True) -> tuple[dict, dict]:
    path = Path(path)
    seal_path = path.with_suffix(".sha256")
    if (path != path.resolve() or not path.is_file() or not seal_path.is_file()
            or runtime.sha(path) != seal_path.read_text().split()[0]):
        raise ValueError("canonical sealed external storage ledger required")
    value = json.loads(path.read_text())
    if (value.get("schema") != LEDGER_SCHEMA or value.get("status") != "pass"
            or value.get("ceiling_bytes") != GLOBAL_BUDGET
            or value.get("cutoff_unix") != LEDGER_CUTOFF
            or value.get("retained_prior_quality_peak_charge_bytes") != QUALITY_PEAK
            or value.get("capture_peak_already_charged") is not True
            or value.get("pinned_raw_peak_bytes") != RAW_BYTES
            or value.get("future_jit_log_allowance_in_baseline_projection") is not True
            or value.get("future_jit_log_allowance_bytes", 0) <= 0
            or value.get("unallocated_reserve_bytes", -1) < 0
            or value.get("measurement_method") != "fixed source-owned paths; apparent lstat trees; original-cutoff file sums; privileged read-only Docker find"
            or value.get("baseline_projected_aggregate_bytes", GLOBAL_BUDGET + 1) > GLOBAL_BUDGET):
        raise ValueError("external storage ledger policy differs")
    if (not isinstance(value.get("measured_unix"), int)
            or not isinstance(value.get("valid_until_unix"), int)
            or value["measured_unix"] < LEDGER_CUTOFF
            or value["valid_until_unix"] < value["measured_unix"]
            or (require_fresh and not value["measured_unix"] <= int(time.time()) <= value["valid_until_unix"])):
        raise ValueError("external storage ledger is not fresh")
    snapshot = storage_ledger_snapshot(value, output)
    by_name = {row["name"]: row for row in snapshot["components"]}
    if (by_name["quality_root"]["projected_bytes"] != QUALITY_PEAK
            or by_name["capture_output"]["projected_bytes"] != value["future_jit_log_allowance_bytes"]):
        raise ValueError("quality-peak/capture-output projection differs")
    minimum_projection = (sum(row["projected_bytes"] for row in snapshot["components"])
                          + value["unallocated_reserve_bytes"])
    if value["baseline_projected_aggregate_bytes"] < minimum_projection:
        raise ValueError("external storage ledger omits projected component or reserve")
    available_quality_credit = max(0, QUALITY_PEAK - by_name["quality_root"]["current_bytes"])
    capture_credit = min(active_capture_bytes, available_quality_credit,
                         by_name["capture_output"]["positive_growth_bytes"])
    gross = value["baseline_projected_aggregate_bytes"] + snapshot["positive_growth_bytes"]
    aggregate = gross - capture_credit
    capture = by_name["capture_output"]
    next_growth = (max(0, capture["current_bytes"] + RAW_BYTES - capture["projected_bytes"])
                   - capture["positive_growth_bytes"])
    next_credit = min(RAW_BYTES, next_growth, max(0, available_quality_credit - capture_credit))
    prospective = aggregate + next_growth - next_credit if reserve_next_raw else aggregate
    if aggregate > GLOBAL_BUDGET or prospective > GLOBAL_BUDGET:
        raise ValueError("external aggregate campaign ceiling exceeded")
    snapshot["aggregate_before_capture_credit_bytes"] = gross
    snapshot["aggregate_after_capture_credit_bytes"] = aggregate
    snapshot["capture_peak_credit_bytes"] = capture_credit
    snapshot["available_quality_peak_credit_bytes"] = available_quality_credit
    snapshot["prospective_next_raw_growth_bytes"] = next_growth
    snapshot["prospective_next_raw_credit_bytes"] = next_credit
    snapshot["prospective_next_raw_aggregate_bytes"] = prospective
    return value, snapshot


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
    stock_receipt = Path(value["stock_carrier"]["receipt"]["path"])
    if runtime.sha(stock_receipt) != value["stock_carrier"]["receipt"]["sha256"]:
        raise ValueError("stock carrier receipt identity differs")
    stock = runtime.validate_stock_carrier_receipt(stock_receipt, carrier)
    extras_receipt = Path(value["stock_carrier"]["extras_receipt"]["path"])
    if runtime.sha(extras_receipt) != value["stock_carrier"]["extras_receipt"]["sha256"]:
        raise ValueError("stock carrier extra-file receipt identity differs")
    extras = runtime.validate_stock_carrier_extras(extras_receipt, carrier)
    if (value["stock_carrier"].get("referenced_shard_count") != len(stock["shards"])
            or value["stock_carrier"].get("resolved_total_bytes") != stock["index"]["resolved_total_bytes"]
            or value["stock_carrier"].get("unreferenced_safetensors")
            != [row["name"] for row in extras["files"]]):
        raise ValueError("stock carrier shard inventory differs")
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
    recipe = json.loads(Path(value["source_recipe"]["path"]).read_text())
    expected_ids = [row["id"] for row in windows]
    roots = set()
    for arm in ARMS:
        entry = value["arms"][arm]
        environment = runtime.arm_environment(arm, expected_ids)
        if entry["environment"] != environment:
            raise ValueError(f"{arm}: environment differs")
        arm_root = Path(entry["capture_root"])
        if arm_root != arm_root.resolve() or arm_root.name != arm:
            raise ValueError(f"{arm}: capture root must be canonical and arm-named")
        expected_argv = _preparer().launch_argv(
            recipe, image["image_id"], arm, environment, arm_root, carrier,
            Path(identity_input["root"]), Path(coupled_input["root"]),
            Path(identity_input["design"]["path"]), Path(coupled_input["design"]["path"]),
            Path(coupled_input["transform"]["path"]))
        if entry["launch_argv"] != expected_argv:
            raise ValueError(f"{arm}: launch argv differs from authenticated source recipe")
        roots.add(str(arm_root.parent))
    if len(roots) != 1:
        raise ValueError("arm capture roots do not share one campaign root")
    return value, windows


def make_execution_seal(manifest_path: Path, seal_path: Path, global_root: Path,
                        storage_ledger_path: Path) -> dict:
    manifest, windows = authenticate_runtime_manifest(manifest_path)
    seal_path, global_root = Path(seal_path), Path(global_root)
    output = Path(manifest["arms"]["stock"]["capture_root"]).parent
    if (seal_path != seal_path.resolve() or seal_path.exists() or seal_path.with_suffix(".sha256").exists()
            or output.exists() or not output.is_relative_to(global_root)):
        raise ValueError("fresh canonical seal/output below global capture root required")
    production = _production_off()
    inventory = global_capture_inventory(global_root, output)
    ledger, ledger_snapshot = validate_storage_ledger(storage_ledger_path, output)
    if shutil.disk_usage(global_root).free < RAW_BYTES:
        raise ValueError("one raw capture does not fit at seal time")
    value = {"schema": "glm53.p8-coupled-three-layer-cf32-execution-seal.v1",
        "status": "authorized-before-gpu-execution", "runtime_manifest": str(manifest_path),
        "runtime_manifest_sha256": runtime.sha(manifest_path), "output": str(output),
        "global_capture_root": str(global_root), "global_retained_capture_inventory": inventory,
        "storage_ledger": {"path": str(storage_ledger_path), "sha256": runtime.sha(storage_ledger_path),
                           "snapshot": ledger_snapshot},
        "aggregate_campaign_ceiling_bytes": GLOBAL_BUDGET,
        "retained_prior_quality_peak_charge_bytes": ledger["retained_prior_quality_peak_charge_bytes"],
        "one_window_raw_bytes": RAW_BYTES,
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
            or seal.get("aggregate_campaign_ceiling_bytes") != GLOBAL_BUDGET
            or seal.get("retained_prior_quality_peak_charge_bytes") != QUALITY_PEAK
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
    ledger_path = Path(seal["storage_ledger"]["path"])
    if runtime.sha(ledger_path) != seal["storage_ledger"]["sha256"]:
        raise ValueError("external storage ledger changed after authorization")
    _ledger, snapshot = validate_storage_ledger(ledger_path, output)
    sealed_rows = {row["name"]: row for row in seal["storage_ledger"]["snapshot"]["components"]}
    if (set(sealed_rows) != set(LEDGER_COMPONENTS)
            or any(row["positive_growth_bytes"] < sealed_rows[row["name"]]["positive_growth_bytes"]
                   for row in snapshot["components"])):
        raise ValueError("campaign storage ledger growth is not monotonic after authorization")
    if shutil.disk_usage(global_root).free < RAW_BYTES:
        raise ValueError("one raw capture does not fit")
    return seal, manifest, windows


def _capture_budget(seal: dict, output: Path, *, reserve_next_raw: bool = True) -> dict:
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
    ledger_path = Path(seal["storage_ledger"]["path"])
    ledger, snapshot = validate_storage_ledger(
        ledger_path, output, active_capture_bytes=active_bytes, require_fresh=False,
        reserve_next_raw=reserve_next_raw)
    if shutil.disk_usage(Path(seal["global_capture_root"])).free < RAW_BYTES:
        raise ValueError("external aggregate campaign ceiling exceeded")
    return {"retained_capture_bytes": retained["bytes"], "active_capture_bytes": active_bytes,
            "capture_peak_credit_bytes": snapshot["capture_peak_credit_bytes"],
            "aggregate_charged_bytes": snapshot["aggregate_after_capture_credit_bytes"],
            "prospective_next_raw_aggregate_bytes": snapshot["prospective_next_raw_aggregate_bytes"],
            "next_window_raw_bytes": RAW_BYTES, "ceiling_bytes": GLOBAL_BUDGET,
            "future_jit_log_allowance_bytes": ledger["future_jit_log_allowance_bytes"]}


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
            "bpw": 4.25 if arm == "identity_p8" else 4.2539798595}
    if completed is not None:
        observed = re.findall(r"GLM53_P8_DECODE_CAPTURE_V2_COMPLETE window=(conditional-fit-\d{4}) rows=(\d+) tp_rank=(\d+)", text)
        if observed != [(window, "2047", "0") for window in completed]:
            raise ValueError("ordered 2047-row capture completion inventory differs")
    return {"arm": arm, "graph": graph, "conditions": conditions,
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
    if (input_sha256 != window["input_sha256"]
            or token_values_sha256 != window["token_values_sha256"]):
        raise ValueError("token file/value identity changed before scoring")
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
            input_sha256 = runtime.sha(Path(window["token_path"]))
            token_values_sha256 = protocol.tokens_sha(tokens)
            if (input_sha256 != window["input_sha256"]
                    or token_values_sha256 != window["token_values_sha256"]):
                raise ValueError("token file/value identity changed before request")
            request_value = protocol.completion_request(tokens, window["id"], f"glm53-p8-three-layer-cf32-{arm}")
            _private_save(out / "requests" / f"{window['id']}.request.json", request_value)
            response = request(request_value, PORT, tick, 900)
            _private_save(out / "requests" / f"{window['id']}.response.json", response)
            response_audit = protocol.verify_response(response, request_value)
            _capture_budget(seal, Path(seal["output"]), reserve_next_raw=False)
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
                                      "raw_sha256": score["raw_sha256"], "raw_retired": True,
                                      "input_sha256": input_sha256,
                                      "token_values_sha256": token_values_sha256})
        record["exit_code"] = 0
    except BaseException as error:
        record["error_type"] = type(error).__name__; _private_save(out / "error.private.txt", str(error))
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
    expected_ids = [window["id"] for window in windows]
    for arm in ARMS:
        root = output / arm
        audit = json.loads((root / "runtime-audit.json").read_text())
        final_audit = json.loads((root / "runtime-final-audit.json").read_text())
        execution_path = root / "execution.json"
        execution = json.loads(execution_path.read_text())
        if (execution.get("schema") != "glm53.p8-coupled-cf32-arm-execution.v1"
                or execution.get("arm") != arm or execution.get("exit_code") != 0
                or execution.get("cleanup") != {"ok": True, "errors": []}
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
            score_path = score_root / f"{wid}.score.json"
            score_npz = score_root / f"{wid}.scores.npz"
            retirement_path = score_root / f"{wid}.retirement.json"
            score = json.loads(score_path.read_text())
            retirement = json.loads(retirement_path.read_text())
            metadata_path = Path(retirement.get("capture_metadata_retained", ""))
            raw_path = root / "captures" / f"{wid}.logits.f32"
            if (score.get("schema") != "glm53.p8-coupled-cf32-window-score.v1"
                    or score.get("status") != "complete" or score.get("arm") != arm
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
                    or execution_rows[wid].get("input_sha256") != window["input_sha256"]
                    or execution_rows[wid].get("token_values_sha256") != window["token_values_sha256"]
                    or retirement.get("schema") != "glm53.p8-coupled-cf32-window-retirement.v1"
                    or retirement.get("status") != "complete"
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
            final_manifest, final_windows = authenticate_runtime_manifest(Path(seal["runtime_manifest"]))
            if final_manifest != manifest or final_windows != windows:
                raise ValueError("runtime/model/sidecar/role identities changed across arms")
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
    seal = sub.add_parser("seal"); seal.add_argument("--manifest", type=Path, required=True); seal.add_argument("--seal", type=Path, required=True); seal.add_argument("--global-root", type=Path, required=True); seal.add_argument("--storage-ledger", type=Path, required=True)
    run = sub.add_parser("execute"); run.add_argument("--seal", type=Path, required=True)
    args = parser.parse_args()
    value = make_execution_seal(args.manifest, args.seal, args.global_root, args.storage_ledger) if args.command == "seal" else execute(args.seal)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
