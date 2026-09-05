"""Fail-closed helpers for Tail-V2 P8 versus production-EXL3 validation.

Importing this module does not inspect GPUs, start containers, open teacher
logits, or delete files.  ``score_retire_window`` is called explicitly by the
future sealed GPU lifecycle immediately after one forced-decode capture.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import fields
import json
import os
from pathlib import Path
import re
import stat
from typing import Callable

import numpy as np

from . import p8_decode_analysis as metric
from . import p8_decode_protocol as protocol
from .audit_speed_graphs import verify as verify_graphs


ARMS = ("exl3", "p8")
REPEATS = 5
ROWS = protocol.ROWS
RAW_BYTES = ROWS * protocol.VOCAB_LIMIT * 4
MAX_NEW_BYTES = 30_000_000_000
DOMAINS = {
    "axis1_general", "axis2_legal", "axis3_code_agentic",
    "axis4_reasoning_termination",
}
TOPOLOGY = {
    "p8": {"tp": 4, "ep": False, "dcp": 1},
    "exl3": {"tp": 4, "ep": True, "dcp": 4},
}
PRODUCT = {
    "p8": {
        "quantization": "modelopt_mixed",
        "moe_backend": "native-p8-mxf8f6f4-n64-fused-scratch",
        "activation_precision": "E4M3 UE8M0_K32 at both native P8 MoE MMA boundaries",
        "payload_bpw": 4.25,
    },
    "exl3": {
        "quantization": "exl3",
        "moe_backend": "b12x-exl3-full-expert",
        "activation_precision": "BF16 model and EXL3 kernel boundary",
        "payload_bpw": 4.0,
    },
}
ORDER = [
    {"repeat": repeat, "arm": arm}
    for repeat in range(1, REPEATS + 1)
    for arm in (("exl3", "p8") if repeat % 2 else ("p8", "exl3"))
]
PLAN_SCHEMA = "glm53.tail-v2-product-validation-plan.v1"
_TRANSIENT_SUFFIXES = (
    ".logits.f32", ".logits.f32.partial", ".capture.json.partial",
    ".capture.inprogress.json", ".capture.failed.json",
)


def sha(path: Path) -> str:
    return protocol.sha(path)


def validate_cf32_payload(payload: dict) -> list[dict]:
    """Validate the frozen balanced development role without opening teachers."""
    windows = protocol.validate_role_payload(payload)
    counts = Counter(row["domain"] for row in windows)
    if len(windows) != 32 or counts != Counter({domain: 8 for domain in DOMAINS}):
        raise ValueError("CF32 must contain exactly eight windows per domain")
    return windows


def authenticate_plan(path: Path) -> dict:
    """Authenticate a resolved plan; proposal files cannot authorize execution."""
    path = Path(path)
    seal = path.with_suffix(".sha256")
    if path != path.resolve() or not path.is_file() or not seal.is_file():
        raise ValueError("canonical plan and seal required")
    if sha(path) != seal.read_text().split()[0]:
        raise ValueError("plan seal differs")
    plan = json.loads(path.read_text())
    if (plan.get("schema") != PLAN_SCHEMA or plan.get("status") != "sealed-before-gpu-execution"
            or plan.get("order") != ORDER or plan.get("cold_runs_per_arm") != REPEATS
            or plan.get("max_new_nvme_bytes") != MAX_NEW_BYTES
            or plan.get("raw_bytes_per_window") != RAW_BYTES
            or plan.get("topology") != TOPOLOGY or plan.get("product") != PRODUCT
            or plan.get("opened_roles") != ["conditional-fit"]
            or plan.get("protected_roles_opened") != []):
        raise ValueError("fixed Tail-V2 product protocol differs")
    roles = Path(plan["roles"])
    if sha(roles) != plan["roles_sha256"] or plan["roles_sha256"] != protocol.ROLES_SHA256:
        raise ValueError("CF32 role identity differs")
    windows = validate_cf32_payload(json.loads(roles.read_text()))
    if (len(windows) != len(plan["windows"])
            or any(any(planned.get(key) != value for key, value in raw.items())
                   for raw, planned in zip(windows, plan["windows"], strict=True))):
        raise ValueError("CF32 window inventory differs")
    for relative, expected in plan["source_sha256"].items():
        source = Path(__file__).resolve().parents[1] / relative
        if source != source.resolve() or sha(source) != expected:
            raise ValueError(f"sealed source differs: {relative}")
    for arm, receipt in plan["image_receipts"].items():
        receipt_path = Path(receipt["path"])
        value = json.loads(receipt_path.read_text())
        if (sha(receipt_path) != receipt["sha256"] or value.get("status") != "complete"
                or value.get("image_id") != plan["images"][arm]
                or value.get("patched_kpool_sha256") != plan["patched_kpool_sha256"]
                or value.get("gpu_used") is not False):
            raise ValueError("Tail-V2 image receipt differs")
    activation_receipt = plan["exl3_activation_receipt"]
    activation_path = Path(activation_receipt["path"])
    activation = json.loads(activation_path.read_text())
    if (sha(activation_path) != activation_receipt["sha256"]
            or activation.get("status") != "pass"
            or activation.get("image_id") != plan["images"]["exl3"]
            or activation.get("source_sha256") != "ae92592ea8fcd249978134357ea3cd2510fe2aa9bdb1d1a3ab02afdbaeb39f45"
            or activation.get("gpu_used") is not False):
        raise ValueError("EXL3 activation boundary receipt differs")
    failed_receipt = plan["preserved_failed_exl3_build_receipt"]
    failed_path = Path(failed_receipt["path"])
    if (sha(failed_path) != failed_receipt["sha256"]
            or json.loads(failed_path.read_text()).get("status") != "failed"):
        raise ValueError("preserved failed image-build receipt differs")
    from . import p8_weight_identity as weights
    weight_path = Path(plan["weight_audit"]["path"])
    if sha(weight_path) != plan["weight_audit"]["sha256"]:
        raise ValueError("fresh payload audit receipt differs")
    weights.verify_stats(json.loads(weight_path.read_text()))
    return plan


def storage_inventory(capture_root: Path) -> dict:
    """Account only transient capture bytes; reject aliases and multiple raws."""
    capture_root = Path(capture_root)
    if capture_root != capture_root.resolve() or not capture_root.is_dir():
        raise ValueError("capture root must be a canonical existing directory")
    rows, active = [], set()
    for path in sorted(capture_root.iterdir()):
        if not path.is_file() or not any(path.name.endswith(s) for s in _TRANSIENT_SUFFIXES):
            continue
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or path.is_symlink() or path.resolve() != path:
            raise ValueError("capture budget refuses symlinks or nonregular artifacts")
        match = re.match(r"(conditional-fit-\d{4})\.", path.name)
        if not match:
            raise ValueError("unexpected transient capture filename")
        active.add(match.group(1))
        rows.append({"name": path.name, "bytes": info.st_size})
    total = sum(row["bytes"] for row in rows)
    raw = [row for row in rows if row["name"].endswith((".logits.f32", ".logits.f32.partial"))]
    if total >= MAX_NEW_BYTES or len(raw) > 1 or len(active) > 1:
        raise ValueError("streaming capture exceeded one-window/<30GiB contract")
    if raw and raw[0]["bytes"] > RAW_BYTES:
        raise ValueError("raw capture exceeds fixed causal geometry")
    return {
        "transient_bytes": total,
        "hard_limit_bytes": MAX_NEW_BYTES,
        "active_window_ids": sorted(active),
        "raw_files": len(raw),
        "artifacts": rows,
    }


def _durable_json(path: Path, value: dict) -> None:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _save_scores(path: Path, scores: object) -> None:
    arrays = {field.name: getattr(scores, field.name) for field in fields(scores)}
    with path.open("xb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


def score_retire_window(
    *, arm: str, repeat: int, window: dict, teacher_root: Path,
    capture_root: Path, receipt_root: Path, runtime_audit_sha256: str,
    score_loader: Callable | None = None,
) -> dict:
    """Score one complete raw capture, receipt it, then retire only that raw.

    The optional loader exists solely for CPU unit tests.  Production uses the
    pinned teacher loader and FP64 KLD implementation.
    """
    if arm not in ARMS or repeat not in range(1, REPEATS + 1):
        raise ValueError("undeclared arm/repeat")
    if not re.fullmatch(r"[a-f0-9]{64}", runtime_audit_sha256):
        raise ValueError("sealed runtime-audit SHA-256 required")
    wid = window.get("id", "")
    if not re.fullmatch(r"conditional-fit-\d{4}", wid) or window.get("domain") not in DOMAINS:
        raise ValueError("window is outside balanced CF32")
    capture_root, receipt_root = Path(capture_root), Path(receipt_root)
    if capture_root != capture_root.resolve() or receipt_root != receipt_root.resolve():
        raise ValueError("canonical capture and receipt roots required")
    inventory_before = storage_inventory(capture_root)
    if inventory_before["active_window_ids"] != [wid]:
        raise ValueError("exactly the requested window may occupy scratch")
    token_path = Path(window["token_path"])
    if sha(token_path) != window["input_sha256"]:
        raise ValueError("CF32 token identity changed")
    tokens = np.load(token_path, allow_pickle=False)
    student, metadata = protocol.load_capture(capture_root, wid, tokens)
    raw_path = capture_root / f"{wid}.logits.f32"
    metadata_path = capture_root / f"{wid}.capture.json"
    raw_sha = metadata["raw_sha256"]
    target = receipt_root / f"repeat-{repeat:02d}" / arm
    if target.exists():
        if not target.is_dir():
            raise ValueError("per-repeat receipt path is not a directory")
    else:
        target.mkdir(parents=True, mode=0o700)
    score_path = target / f"{wid}.scores.npz"
    score_json = target / f"{wid}.score.json"
    retire_json = target / f"{wid}.retirement.json"
    if any(path.exists() for path in (score_path, score_json, retire_json)):
        raise ValueError("immutable score receipt already exists")
    if score_loader is None:
        teacher_path = Path(teacher_root) / window["teacher_path"]
        if sha(teacher_path) != window["teacher_sha256"]:
            raise ValueError("teacher identity changed")
        teacher = metric._load_teacher(teacher_path, window)
        scores = protocol.score_aligned(teacher, student, tokens)
    else:
        scores = score_loader(window, student, tokens)
    if not np.isfinite(scores.kld).all() or len(scores.kld) != ROWS:
        raise ValueError("score geometry/finite gate failed")
    _save_scores(score_path, scores)
    reference_sha = None
    deterministic = True
    if repeat > 1:
        reference = receipt_root / "repeat-01" / arm / f"{wid}.score.json"
        prior = json.loads(reference.read_text())
        reference_sha = prior["raw_sha256"]
        deterministic = raw_sha == reference_sha
    score_record = {
        "schema": "glm53.tail-v2-product-window-score.v1",
        "status": "complete",
        "arm": arm,
        "repeat": repeat,
        "window_id": wid,
        "domain": window["domain"],
        "mean_kld": float(np.mean(scores.kld)),
        "true_decode_mean_kld": float(np.mean(scores.kld[1:])),
        "one_token_prefill_kld": float(scores.kld[0]),
        "raw_sha256": raw_sha,
        "reference_repeat_01_raw_sha256": reference_sha,
        "bitwise_equal_to_repeat_01": deterministic,
        "capture_metadata_sha256": sha(metadata_path),
        "scores_sha256": sha(score_path),
        "runtime_audit_sha256": runtime_audit_sha256,
        "raw_bytes": RAW_BYTES,
        "raw_retired": False,
        "storage_before": inventory_before,
        "metric": "KL(teacher || student), nats, pinned FP64 CPU causal alignment",
    }
    _durable_json(score_json, score_record)
    # The durable score receipt binds the raw bytes before the only destructive
    # operation.  Metadata is retained; partial/failure artifacts are never removed.
    del student, scores
    if (raw_path.resolve() != raw_path or raw_path.is_symlink()
            or raw_path.stat().st_size != RAW_BYTES or sha(raw_path) != raw_sha):
        raise ValueError("raw changed before retirement")
    raw_path.unlink()
    descriptor = os.open(capture_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    inventory_after = storage_inventory(capture_root)
    if inventory_after["raw_files"] != 0:
        raise ValueError("raw retirement did not close storage ledger")
    retirement = {
        "schema": "glm53.tail-v2-product-window-retirement.v1",
        "status": "complete",
        "score_receipt_sha256": sha(score_json),
        "retired_raw_sha256": raw_sha,
        "retired_raw_bytes": RAW_BYTES,
        "capture_metadata_retained": str(metadata_path),
        "storage_after": inventory_after,
    }
    _durable_json(retire_json, retirement)
    return {**score_record, "raw_retired": True, "retirement_sha256": sha(retire_json)}


def verify_five_run_determinism(receipt_root: Path, windows: list[dict]) -> dict:
    validate_cf32_payload({
        "schema": "glm53-codec.conditional-fit32-development-role.v1",
        "teacher_repo_id": "brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits",
        "teacher_revision": protocol.FULL_PANEL_SOURCE_REVISION,
        "roles": {"fit": [], "conditional-fit": windows, "selection": [], "confirmation": [], "final": []},
    })
    rows = []
    for arm in ARMS:
        for window in windows:
            values = []
            for repeat in range(1, REPEATS + 1):
                base = Path(receipt_root) / f"repeat-{repeat:02d}" / arm
                score_path = base / f"{window['id']}.score.json"
                retire_path = base / f"{window['id']}.retirement.json"
                score, retired = json.loads(score_path.read_text()), json.loads(retire_path.read_text())
                if (retired["score_receipt_sha256"] != sha(score_path)
                        or retired["retired_raw_sha256"] != score["raw_sha256"]):
                    raise ValueError("score/retirement receipt linkage differs")
                values.append(score)
            hashes = {value["raw_sha256"] for value in values}
            rows.append({"arm": arm, "window_id": window["id"], "hashes": sorted(hashes),
                         "bitwise_deterministic": len(hashes) == 1})
    passed = all(row["bitwise_deterministic"] for row in rows)
    return {"schema": "glm53.tail-v2-product-determinism.v1", "status": "pass" if passed else "fail",
            "cold_runs_per_arm": REPEATS, "windows": len(windows), "rows": rows,
            "kld_experimental_units": "32 windows from repeat 1 only; cold repeats are determinism replications"}


def audit_runtime_log(log: str, arm: str, *, require_dispatch: bool = True) -> dict:
    """Reject any drift in the declared product runtime before measurements count."""
    if arm not in ARMS:
        raise ValueError("unknown arm")
    topology = TOPOLOGY[arm]
    markers = (
        "Using V2 Model Runner", "tensor_parallel_size=4",
        f"decode_context_parallel_size={topology['dcp']}",
        "attention_backend': 'B12X_MLA_SPARSE'", "kv_cache_dtype=nvfp4_ds_mla",
        "speculative_config=None", "dtype=torch.bfloat16", "enforce_eager=False",
    )
    if any(marker not in log for marker in markers):
        raise ValueError("attention/KV/activation/MTP/graph/topology marker missing")
    if ("'enable_expert_parallel': True" in log) != topology["ep"]:
        raise ValueError("expert-parallel regime differs")
    graph = verify_graphs(log)
    if arm == "p8" and require_dispatch:
        expected = {(layer, rank) for layer in range(3, 45) for rank in range(4)}
        found = re.findall(
            r"GLM53_P8_NATIVE_FORWARD layer=(\d+) rank=(\d+)[^\r\n]*"
            r"mma=mxf8f6f4[^\r\n]*alphabet=E4M3[^\r\n]*scale=UE8M0_K32[^\r\n]*"
            r"physical_bpw=4\.25[^\r\n]*small_m_scheduler=true", log,
        )
        if len(found) != 168 or {(int(a), int(b)) for a, b in found} != expected:
            raise ValueError("P8 native/E4M3/4.25bpw dispatch inventory differs")
    elif arm == "exl3":
        for marker in ("quantization=exl3", "moe_backend='b12x'",
                       "EXL3 full-expert EP runtime planned"):
            if marker not in log:
                raise ValueError("EXL3 K4/BF16/B12X dispatch marker missing")
    return {"arm": arm, "topology": topology, **PRODUCT[arm], "graph": graph,
            "attention_backend": "B12X_MLA_SPARSE", "kv_cache_dtype": "nvfp4_ds_mla",
            "mtp": False, "tail_repair": "verified by immutable image receipt, not inferred from log"}


def validate_speed_result(result: dict, arm: str) -> dict:
    """Reuse the established 32K C1/prefill fields without accepting substitutes."""
    from . import p8_fc1_cold_compare as cold
    return cold.summarize(result, arm)
