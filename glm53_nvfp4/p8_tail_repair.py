"""Sealed four-window P8 KPool-tail repair capture and KLD analysis."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from dataclasses import fields
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal

import numpy as np

from . import decode_path_control as control
from . import p8_decode_analysis as metric
from . import p8_decode_launcher as base
from . import p8_decode_protocol as protocol
from . import p8_index_order_full as ordered


REPO = Path(__file__).resolve().parents[1]
ROOT = Path("/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6")
PREFIX = "glm53-p8-tail-repair-v1"
PORT = 8025
ORDER = ({"stage": "tail-repair", "arm": "n64"},)
PARENT_IMAGE = "sha256:0f1eae9329965d68713857e4a5a12e9c5440c866b532e7ba288dc2ae4067fad9"
IMAGE_RECEIPT = ROOT / "p8-tail-repair-image-v1a/receipt.json"
RECIPE = ROOT / "p8-smallm-scheduler-v1/fc1-integrated-v1/container-final.private.json"
RECIPE_SHA256 = "c7f5832af2244926ebf0f2d481efbe7461bee85afb18cc4d85dd3769d44ffdf8"
PREREG = REPO / "experiments/p8-tail-omission-repair-v1-prereg.json"
AMENDMENT = REPO / "experiments/p8-tail-omission-repair-v1b-amendment.json"
ROLES = control.ROLES
TEACHER = control.TEACHER
WINDOW_IDS = control.WINDOW_IDS
BASELINE = control.P8_WINDOW_KLD
BASELINE_MEAN = control.P8_MEAN
RAW_BYTES = 4 * protocol.ROWS * protocol.VOCAB_LIMIT * 4
HEADROOM = 20 * 2**30
KPOOL_SOURCE = "/opt/infernal-invocation/vllm/vllm/models/glm5next/nvidia/ops/kpool_compress.py"
_BASE_CLONE_ARGV = base.clone_argv


def source_files() -> set[str]:
    return ordered.source_files() | {
        "glm53_nvfp4/p8_tail_repair.py",
        "tests/test_p8_tail_repair.py",
        "scripts/build_p8_tail_repair_image.py",
        "runtime_patch/p8_tail_repair/Dockerfile",
        "runtime_patch/p8_tail_repair/kpool-tail-consumed-v1.patch",
        "experiments/p8-tail-omission-repair-v1-prereg.json",
        "experiments/p8-tail-omission-repair-v1a-amendment.json",
        "experiments/p8-tail-omission-repair-v1b-amendment.json",
    }


def sha(path: Path) -> str:
    return protocol.sha(path)


def load_image_receipt() -> dict:
    value = json.loads(IMAGE_RECEIPT.read_text())
    if (
        value.get("schema") != "glm53.p8-tail-repair-image.v1"
        or value.get("status") != "complete"
        or value.get("parent_image_id") != PARENT_IMAGE
        or not re.fullmatch(r"sha256:[a-f0-9]{64}", value.get("image_id", ""))
        or not re.fullmatch(r"[a-f0-9]{64}", value.get("patched_kpool_sha256", ""))
        or value.get("gpu_used") is not False
        or value.get("speed_measurement_valid") is not False
    ):
        raise ValueError("tail-repair image receipt differs")
    expected_sources = {
        name: sha(REPO / "runtime_patch/p8_tail_repair" / name)
        for name in ("Dockerfile", "kpool-tail-consumed-v1.patch")
    }
    if value.get("source_sha256") != expected_sources:
        raise ValueError("tail-repair image sources differ")
    if value.get("builder_sha256") != sha(REPO / "scripts/build_p8_tail_repair_image.py"):
        raise ValueError("tail-repair image builder differs")
    return value


def selected_windows() -> list[dict]:
    return control.selected_windows()


def input_stats(windows: list[dict]) -> dict:
    paths = {ROLES}
    for window in windows:
        paths.add(Path(window["token_path"]))
        paths.add(TEACHER / window["teacher_path"])
    return {str(path): base.fingerprint(path) for path in sorted(paths)}


def load_recipe() -> dict:
    if sha(RECIPE) != RECIPE_SHA256:
        raise ValueError("P8 serving recipe changed")
    value = json.loads(RECIPE.read_text())
    if value.get("Image") != base.cold.IMAGES["p8"]:
        raise ValueError("P8 recipe parent image differs")
    base.cold.validate_recipe(value, "p8")
    return value


def make_plan(path: Path, output: Path) -> dict:
    if (
        path != path.resolve()
        or output != output.resolve()
        or path.exists()
        or path.with_suffix(".sha256").exists()
        or output.exists()
        or output.parent != ROOT
    ):
        raise ValueError("fresh canonical plan, seal, and NVMe output required")
    prereg = json.loads(PREREG.read_text())
    if prereg.get("status") != "predeclared-before-tail-source-implementation-image-build-or-GPU-run":
        raise ValueError("tail-repair preregistration differs")
    image = load_image_receipt()
    windows = selected_windows()
    sources = source_files()
    missing = [name for name in sources if not (REPO / name).is_file()]
    if missing:
        raise ValueError(f"tail-repair source inventory incomplete: {missing}")
    plan = {
        "schema": "glm53.p8-tail-omission-repair-plan.v1",
        "created_at": base.pilot.now(),
        "output": str(output),
        "order": list(ORDER),
        "capture_image": image["image_id"],
        "parent_capture_image": PARENT_IMAGE,
        "image_receipt": str(IMAGE_RECEIPT),
        "image_receipt_sha256": sha(IMAGE_RECEIPT),
        "patched_kpool_sha256": image["patched_kpool_sha256"],
        "recipe": str(RECIPE),
        "recipe_sha256": RECIPE_SHA256,
        "prereg": str(PREREG),
        "prereg_sha256": sha(PREREG),
        "amendment": str(AMENDMENT),
        "amendment_sha256": sha(AMENDMENT),
        "windows": windows,
        "roles": str(ROLES),
        "roles_sha256": sha(ROLES),
        "teacher_root": str(TEACHER),
        "input_stats": input_stats(windows),
        "source_sha256": {name: sha(REPO / name) for name in sorted(sources)},
        "baseline_window_kld": BASELINE,
        "baseline_mean_kld": BASELINE_MEAN,
        "raw_capture_bytes": RAW_BYTES,
        "headroom_bytes": HEADROOM,
        "ready_timeout_seconds": 2400,
        "request_timeout_seconds": 900,
        "capture_artifact_owner": {"uid": os.getuid(), "gid": os.getgid()},
        "runtime": {
            "tp": 4,
            "ep": False,
            "dcp": 1,
            "attention_backend": "B12X_MLA_SPARSE",
            "kv_cache_dtype": "nvfp4_ds_mla",
            "cuda_graphs": True,
            "mtp": False,
            "p8_fc1_tile_n": 64,
            "fixed_index_order": "logical-short-v1",
            "physical_bpw": 4.25,
        },
        "decision": {
            "material_support_relative_improvement": 0.20,
            "material_support_minimum_window_wins": 3,
            "material_reject_maximum_relative_improvement": 0.05,
            "paired_window_interval_is_blocking": False,
        },
        "opened_roles": ["conditional-fit"],
        "protected_roles_opened": [],
        "speed_measurement_valid": False,
        "allocation_restart": False,
        "retry_policy": "no silent retry, reordering, substitution, or threshold change",
    }
    base.pilot.save(path, plan)
    base.pilot.save(path.with_suffix(".sha256"), sha(path) + "  " + path.name + "\n")
    return plan


def verify_inputs(plan: dict) -> None:
    for name, expected in plan["input_stats"].items():
        if base.fingerprint(Path(name)) != expected:
            raise ValueError("approved role/token/teacher input changed")


def authenticate(path: Path) -> dict:
    if path != path.resolve() or not path.is_file() or not path.with_suffix(".sha256").is_file():
        raise ValueError("canonical sealed tail-repair plan required")
    if sha(path) != path.with_suffix(".sha256").read_text().split()[0]:
        raise ValueError("tail-repair plan seal differs")
    plan = json.loads(path.read_text())
    if (
        plan.get("schema") != "glm53.p8-tail-omission-repair-plan.v1"
        or plan.get("order") != list(ORDER)
        or plan.get("baseline_window_kld") != BASELINE
        or plan.get("baseline_mean_kld") != BASELINE_MEAN
        or plan.get("raw_capture_bytes") != RAW_BYTES
        or plan.get("headroom_bytes") != HEADROOM
        or plan.get("ready_timeout_seconds") != 2400
        or plan.get("request_timeout_seconds") != 900
        or plan.get("opened_roles") != ["conditional-fit"]
        or plan.get("protected_roles_opened") != []
        or plan.get("runtime", {}).get("physical_bpw") != 4.25
        or plan.get("runtime", {}).get("mtp") is not False
    ):
        raise ValueError("tail-repair fixed contract differs")
    if set(plan.get("source_sha256", {})) != source_files():
        raise ValueError("tail-repair source inventory differs")
    for name, expected in plan["source_sha256"].items():
        if sha(REPO / name) != expected:
            raise ValueError(f"tail-repair source changed: {name}")
    image = load_image_receipt()
    if (
        sha(IMAGE_RECEIPT) != plan["image_receipt_sha256"]
        or image["image_id"] != plan["capture_image"]
        or image["patched_kpool_sha256"] != plan["patched_kpool_sha256"]
        or sha(PREREG) != plan["prereg_sha256"]
        or sha(AMENDMENT) != plan["amendment_sha256"]
        or sha(RECIPE) != plan["recipe_sha256"]
        or sha(ROLES) != plan["roles_sha256"]
    ):
        raise ValueError("tail-repair image, preregistration, recipe, or role changed")
    windows = selected_windows()
    if windows != plan["windows"] or input_stats(windows) != plan["input_stats"]:
        raise ValueError("fixed tail-repair windows changed")
    verify_inputs(plan)
    load_recipe()
    return plan


def model_name(arm: str) -> str:
    if arm != "n64":
        raise ValueError("tail repair has one N64 arm")
    return PREFIX


def slot_name(entry: dict) -> str:
    if entry not in ORDER:
        raise ValueError("undeclared tail-repair stage")
    return "candidate"


def stage_windows(plan: dict, stage: str) -> list[dict]:
    if stage != "tail-repair":
        raise ValueError("undeclared tail-repair stage")
    return plan["windows"]


def clone_argv(container: dict, image: str, entry: dict, out: Path, windows: list[dict], owner: str) -> list[str]:
    args = _BASE_CLONE_ARGV(container, image, entry, out, windows, owner)
    return [
        *args[:-3],
        "--env", "GLM53_P8_INDEX_ORDER=logical-short-v1",
        "--env", "GLM53_P8_INDEX_ORDER_RECEIPT=1",
        "--env", "GLM53_P8_INDEX_TRACE=",
        *args[-3:],
    ]


def runtime_audit(log: str, arm: str, completed_windows=None) -> dict:
    return {
        **ordered.runtime_audit(log, arm, completed_windows),
        "tail_repair": "tail-consumed-v1",
    }


def verify_stage_identities(plan: dict) -> None:
    path = Path(plan["plan_path"])
    candidate = authenticate(path)
    if candidate["capture_image"] != plan["capture_image"]:
        raise ValueError("stage image differs from sealed plan")


@contextmanager
def adapter(plan: dict):
    names = (
        "PREFIX", "PORT", "stage_windows", "verify_stage_identities",
        "clone_argv", "runtime_audit", "model_name", "slot_name",
    )
    previous = {name: getattr(base, name) for name in names}
    try:
        base.PREFIX, base.PORT = PREFIX, PORT
        base.stage_windows = stage_windows
        base.verify_stage_identities = verify_stage_identities
        base.clone_argv = clone_argv
        base.runtime_audit = runtime_audit
        base.model_name = model_name
        base.slot_name = slot_name
        yield
    finally:
        for name, value in previous.items():
            setattr(base, name, value)


def restoration_safety(plan: dict, plan_sha: str) -> dict:
    result = {"ok": False, "errors": [], "containers": []}
    try:
        ids = base.pilot.command([
            "docker", "ps", "-a", "--no-trunc", "--filter",
            f"name=^/{PREFIX}-", "--format", "{{.ID}}",
        ]).stdout.splitlines()
        for cid in ids:
            if not re.fullmatch(r"[a-f0-9]{64}", cid):
                raise ValueError("unresolved tail-repair container")
            container = json.loads(base.pilot.command(["docker", "inspect", cid]).stdout)[0]
            if (
                container["Id"] != cid
                or container["Name"] != f"/{PREFIX}-candidate"
                or container["Image"] != plan["capture_image"]
                or container["Config"].get("Labels", {}).get(base.cold.LABEL)
                != f"{plan_sha}:candidate"
            ):
                raise ValueError("tail-repair container ownership uncertain")
            result["containers"].append({"id": cid, "name": container["Name"], "running": container["State"]["Running"]})
            if container["State"]["Running"]:
                raise RuntimeError("tail-repair container remains live")
        result["ok"] = True
    except BaseException as error:
        result["errors"].append(type(error).__name__)
    return result


def run(path: Path) -> dict:
    plan = authenticate(path)
    output = Path(plan["output"])
    if output.exists() or output.parent != ROOT or shutil.disk_usage(ROOT).free < RAW_BYTES + HEADROOM:
        raise ValueError("fresh NVMe output and 5.07GB capture plus20GiB headroom required")
    if base.pilot.command(["git", "-C", str(REPO), "status", "--porcelain"]).stdout.strip():
        raise ValueError("clean sealed checkout required")
    image = plan["capture_image"]
    if base.pilot.command(["docker", "image", "inspect", image, "--format", "{{.Id}}"] ).stdout.strip() != image:
        raise ValueError("immutable tail-repair image missing")
    installed = base.pilot.command([
        "docker", "run", "--rm", "--network=none", "--runtime", "runc",
        "-e", "NVIDIA_VISIBLE_DEVICES=void", "--entrypoint", "sha256sum", image,
        KPOOL_SOURCE,
    ]).stdout.split()[0]
    if installed != plan["patched_kpool_sha256"]:
        raise ValueError("installed tail-repair source differs")
    recipe, digest = load_recipe(), sha(path)
    with open("/run/lock/klc/model-stack.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(mode=0o700)
        prior = {
            "backend": base.pilot.active("klc-backend.service", True),
            "timer": base.pilot.active("klc-model-stack.timer"),
        }
        record = {
            "schema": "glm53.p8-tail-omission-repair-execution.v1",
            "plan_sha256": digest,
            "started_at": base.pilot.now(),
            "prior": prior,
            "exit_code": 1,
            "stages": [],
            "opened_roles": ["conditional-fit"],
            "protected_roles_opened": [],
            "speed_measurement_valid": False,
            "allocation_restart": False,
            "mtp": False,
        }
        handlers = {}
        try:
            def interrupted(signum, frame):
                raise RuntimeError(f"tail-repair capture interrupted by signal {signum}")
            for sig in base.INTERRUPT_SIGNALS:
                handlers[sig] = signal.signal(sig, interrupted)
            if prior["timer"]:
                base.pilot.command(["sudo", "-n", "systemctl", "stop", "klc-model-stack.timer"])
            if prior["backend"]:
                base.pilot.command(["systemctl", "--user", "stop", "klc-backend.service"])
            hardware = base.cold.inventory()
            base.pilot.save(output / "hardware-inventory.json", hardware)
            entry = dict(ORDER[0])
            stage_plan = {**copy.deepcopy(plan), "plan_path": str(path)}
            with adapter(stage_plan):
                stage = base.capture_stage(stage_plan, entry, recipe, output / "candidate", digest, hardware)
            record["stages"].append({"slot": "candidate", "execution_sha256": sha(output / "candidate/execution.json")})
            if stage["exit_code"] != 0:
                raise RuntimeError("tail-repair capture failed; no automatic retry")
            record["capture_protocol_complete"] = True
            record["exit_code"] = 0
        except BaseException as error:
            record["error_type"] = type(error).__name__
            base.pilot.private_save(output / "error.private.txt", str(error))
        finally:
            for sig in handlers:
                signal.signal(sig, signal.SIG_IGN)
            try:
                authenticate(path)
                record["final_identity_audit"] = {"ok": True}
            except BaseException as error:
                record["final_identity_audit"] = {"ok": False, "error_type": type(error).__name__}
                record["exit_code"] = 1
            safety = restoration_safety(plan, digest)
            record["restoration_safety"] = safety
            record["restoration"] = base.cold.restore_if_safe(prior, safety)
            if record["restoration"].get("errors") or any(record["restoration"].get(k) != v for k, v in prior.items()):
                record["exit_code"] = 1
            record["finished_at"] = base.pilot.now()
            base.pilot.save(output / "execution.json", record)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    if record["exit_code"]:
        raise RuntimeError("tail-repair capture failed; evidence preserved")
    return record


def analyze(path: Path, output: Path) -> dict:
    import torch

    torch.set_num_threads(metric.CPU_THREADS)
    plan = authenticate(path)
    root = Path(plan["output"])
    execution = json.loads((root / "execution.json").read_text())
    stage = json.loads((root / "candidate/execution.json").read_text())
    if (
        execution.get("exit_code") != 0
        or execution.get("capture_protocol_complete") is not True
        or stage.get("exit_code") != 0
        or [row.get("id") for row in stage.get("windows", [])] != list(WINDOW_IDS)
    ):
        raise ValueError("successful four-window tail-repair capture required")
    if output != output.resolve() or output.exists():
        raise ValueError("fresh canonical tail-repair analysis output required")
    output.mkdir(parents=True, mode=0o700)
    records = []
    for window in plan["windows"]:
        wid = window["id"]
        tokens = np.load(window["token_path"], allow_pickle=False)
        student, metadata = protocol.load_capture(root / "candidate/captures", wid, tokens)
        teacher_path = Path(plan["teacher_root"]) / window["teacher_path"]
        if sha(teacher_path) != window["teacher_sha256"]:
            raise ValueError("teacher changed")
        teacher = metric._load_teacher(teacher_path, window)
        scores = protocol.score_aligned(teacher, student, tokens)
        with (output / f"{wid}.npz").open("xb") as stream:
            np.savez(stream, **{field.name: getattr(scores, field.name) for field in fields(scores)})
        candidate = float(scores.kld.mean())
        row = {
            "window_id": wid,
            "domain": window["domain"],
            "candidate_mean_kld": candidate,
            "candidate_true_decode_mean_kld": float(scores.kld[1:].mean()),
            "candidate_one_token_prefill_kld": float(scores.kld[0]),
            "baseline_mean_kld": BASELINE[wid],
            "candidate_minus_baseline_kld": candidate - BASELINE[wid],
            "raw_sha256": metadata["raw_sha256"],
        }
        records.append(row)
        base.pilot.save(output / f"{wid}.json", row)
        print(json.dumps(row, sort_keys=True), flush=True)
        del teacher, student, scores
    candidate_mean = float(np.mean([row["candidate_mean_kld"] for row in records]))
    deltas = [row["candidate_minus_baseline_kld"] for row in records]
    relative = (BASELINE_MEAN - candidate_mean) / BASELINE_MEAN
    wins = sum(delta < 0 for delta in deltas)
    if relative >= 0.20 and wins >= 3:
        verdict = "material-tail-cause-supported"
    elif relative <= 0.05:
        verdict = "material-tail-cause-rejected"
    else:
        verdict = "directional-inconclusive"
    result = {
        "schema": "glm53.p8-tail-omission-repair-analysis.v1",
        "status": "complete",
        "verdict": verdict,
        "plan_sha256": sha(path),
        "metric": "KL(teacher || student), nats, FP64 CPU, all154880 entries",
        "windows": 4,
        "causal_rows": 4 * protocol.ROWS,
        "baseline_mean_kld": BASELINE_MEAN,
        "candidate_mean_kld": candidate_mean,
        "mean_delta_kld": candidate_mean - BASELINE_MEAN,
        "relative_improvement": relative,
        "window_wins": wins,
        "paired_window_delta_ci95_percentile": control.bootstrap(deltas, 20260905),
        "records": records,
        "opened_roles": ["conditional-fit"],
        "protected_roles_opened": [],
        "speed_measurement_valid": False,
        "mtp": False,
        "physical_bpw": 4.25,
        "claim_boundary": "Targeted four-window serving-path diagnostic; not full32 product qualification.",
    }
    result["files"] = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    base.pilot.save(output / "analysis.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    planning = subs.add_parser("plan")
    planning.add_argument("--plan", type=Path, required=True)
    planning.add_argument("--output", type=Path, required=True)
    running = subs.add_parser("run")
    running.add_argument("--plan", type=Path, required=True)
    scoring = subs.add_parser("analyze")
    scoring.add_argument("--plan", type=Path, required=True)
    scoring.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = (
        make_plan(args.plan, args.output)
        if args.command == "plan"
        else run(args.plan)
        if args.command == "run"
        else analyze(args.plan, args.output)
    )
    print(json.dumps(result, sort_keys=True))
