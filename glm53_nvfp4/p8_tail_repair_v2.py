"""Sealed V2 P8 KPool tail repair with mandatory device row closure."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re

from . import p8_tail_repair as prior
from . import p8_decode_launcher as base
from . import p8_decode_protocol as protocol


REPO = Path(__file__).resolve().parents[1]
ROOT = prior.ROOT
PREFIX = "glm53-p8-tail-repair-v2"
PORT = 8028
PARENT_IMAGE = prior.PARENT_IMAGE
IMAGE_RECEIPT = ROOT / "p8-tail-repair-image-v2a/receipt.json"
DEVICE_RECEIPT = ROOT / "p8-tail-repair-image-v2a/device-row-closure.json"
PREREG = REPO / "experiments/p8-tail-omission-repair-v2-prereg.json"
AMENDMENT = REPO / "experiments/p8-tail-omission-repair-v2a-build-amendment.json"
PATCH = REPO / "runtime_patch/p8_tail_repair_v2/kpool-tail-after-valid-history-v2.patch"
DOCKERFILE = REPO / "runtime_patch/p8_tail_repair_v2/Dockerfile"
BUILDER = REPO / "scripts/build_p8_tail_repair_v2_image.py"
DEVICE_RUNNER = REPO / "scripts/run_p8_tail_repair_v2_device_closure.py"
DEVICE_VERIFIER = REPO / "scripts/verify_p8_tail_repair_v2_device.py"
ORDER = prior.ORDER
WINDOW_IDS = prior.WINDOW_IDS
BASELINE = prior.BASELINE
BASELINE_MEAN = prior.BASELINE_MEAN
RAW_BYTES = prior.RAW_BYTES
HEADROOM = prior.HEADROOM
_V1_SOURCE_FILES = prior.source_files


def sha(path: Path) -> str:
    return protocol.sha(path)


def source_files() -> set[str]:
    return _V1_SOURCE_FILES() | {
        "glm53_nvfp4/p8_tail_repair_v2.py",
        "tests/test_p8_tail_repair_v2.py",
        "scripts/build_p8_tail_repair_v2_image.py",
        "scripts/run_p8_tail_repair_v2_device_closure.py",
        "scripts/verify_p8_tail_repair_v2_device.py",
        "runtime_patch/p8_tail_repair_v2/Dockerfile",
        "runtime_patch/p8_tail_repair_v2/kpool-tail-after-valid-history-v2.patch",
        "experiments/p8-tail-omission-repair-v2-prereg.json",
        "experiments/p8-tail-omission-repair-v2a-build-amendment.json",
    }


def load_image_receipt() -> dict:
    value = json.loads(IMAGE_RECEIPT.read_text())
    expected_sources = {"Dockerfile": sha(DOCKERFILE), "kpool-tail-after-valid-history-v2.patch": sha(PATCH)}
    if (
        value.get("schema") != "glm53.p8-tail-repair-image.v2"
        or value.get("status") != "complete"
        or value.get("parent_image_id") != PARENT_IMAGE
        or not re.fullmatch(r"sha256:[a-f0-9]{64}", value.get("image_id", ""))
        or not re.fullmatch(r"[a-f0-9]{64}", value.get("patched_kpool_sha256", ""))
        or value.get("source_sha256") != expected_sources
        or value.get("builder_sha256") != sha(BUILDER)
        or value.get("gpu_used") is not False
    ):
        raise ValueError("tail-repair V2 image receipt differs")
    return value


def load_device_receipt(image: dict | None = None) -> dict:
    image = image or load_image_receipt()
    value = json.loads(DEVICE_RECEIPT.read_text())
    if (
        value.get("schema") != "glm53.p8-tail-repair-device-row-closure.v2"
        or value.get("status") != "pass"
        or value.get("image_id") != image["image_id"]
        or value.get("image_receipt_sha256") != sha(IMAGE_RECEIPT)
        or value.get("verifier_sha256") != sha(DEVICE_VERIFIER)
        or value.get("runner_sha256") != sha(DEVICE_RUNNER)
        or value.get("sequence_lengths") != [1, 2047]
        or value.get("differing_rows") != 1536
        or value.get("unchanged_rows") != 511
        or value.get("diff_count_by_L_mod_4") != {"0": 0, "1": 512, "2": 512, "3": 512}
        or value.get("gpu_used") is not True
        or value.get("kld_run") is not False
    ):
        raise ValueError("mandatory V2 device row closure differs")
    expected_rows = [row for row in range(2047) if (row + 1) % 4 != 0]
    if value.get("differing_rows_zero_based") != expected_rows:
        raise ValueError("device per-row diff inventory differs")
    return value


def make_plan(path: Path, output: Path) -> dict:
    if path != path.resolve() or output != output.resolve() or path.exists() or path.with_suffix(".sha256").exists() or output.exists() or output.parent != ROOT:
        raise ValueError("fresh canonical V2 plan, seal, and NVMe output required")
    if json.loads(PREREG.read_text()).get("status") != "predeclared-before-v2-tail-source-implementation-image-build-device-closure-or-kld-run":
        raise ValueError("tail-repair V2 preregistration differs")
    image = load_image_receipt()
    device = load_device_receipt(image)
    windows = prior.selected_windows()
    sources = source_files()
    plan = {
        "schema": "glm53.p8-tail-omission-repair-plan.v2", "created_at": base.pilot.now(),
        "output": str(output), "order": list(ORDER), "capture_image": image["image_id"],
        "parent_capture_image": PARENT_IMAGE, "image_receipt": str(IMAGE_RECEIPT),
        "image_receipt_sha256": sha(IMAGE_RECEIPT), "patched_kpool_sha256": image["patched_kpool_sha256"],
        "device_row_closure": str(DEVICE_RECEIPT), "device_row_closure_sha256": sha(DEVICE_RECEIPT),
        "recipe": str(prior.RECIPE), "recipe_sha256": prior.RECIPE_SHA256,
        "prereg": str(PREREG), "prereg_sha256": sha(PREREG), "windows": windows,
        "amendment": str(AMENDMENT), "amendment_sha256": sha(AMENDMENT),
        "roles": str(prior.ROLES), "roles_sha256": sha(prior.ROLES), "teacher_root": str(prior.TEACHER),
        "input_stats": prior.input_stats(windows), "source_sha256": {name: sha(REPO / name) for name in sorted(sources)},
        "baseline_window_kld": BASELINE, "baseline_mean_kld": BASELINE_MEAN,
        "raw_capture_bytes": RAW_BYTES, "headroom_bytes": HEADROOM,
        "ready_timeout_seconds": 2400, "request_timeout_seconds": 900,
        "capture_artifact_owner": {"uid": os.getuid(), "gid": os.getgid()},
        "runtime": {"tp": 4, "ep": False, "dcp": 1, "attention_backend": "B12X_MLA_SPARSE",
                    "kv_cache_dtype": "nvfp4_ds_mla", "cuda_graphs": True, "mtp": False,
                    "p8_fc1_tile_n": 64, "fixed_index_order": "logical-short-v1", "physical_bpw": 4.25,
                    "tail_repair": "tail-after-valid-history-v2"},
        "intervention_integrity": {"status": "pass", "device_receipt_sha256": sha(DEVICE_RECEIPT),
                                   "differing_rows": device["differing_rows"], "unchanged_rows": device["unchanged_rows"],
                                   "diff_count_by_L_mod_4": device["diff_count_by_L_mod_4"]},
        "decision": {"material_support_relative_improvement": 0.20, "material_support_minimum_window_wins": 3,
                     "material_reject_maximum_relative_improvement": 0.05, "paired_window_interval_is_blocking": False},
        "opened_roles": ["conditional-fit"], "protected_roles_opened": [], "speed_measurement_valid": False,
        "allocation_restart": False, "retry_policy": "one sealed V2 execution; no silent retry or threshold change",
    }
    base.pilot.save(path, plan)
    base.pilot.save(path.with_suffix(".sha256"), sha(path) + "  " + path.name + "\n")
    return plan


def authenticate(path: Path) -> dict:
    if path != path.resolve() or not path.is_file() or not path.with_suffix(".sha256").is_file() or sha(path) != path.with_suffix(".sha256").read_text().split()[0]:
        raise ValueError("canonical sealed tail-repair V2 plan required")
    plan = json.loads(path.read_text())
    if (plan.get("schema") != "glm53.p8-tail-omission-repair-plan.v2" or plan.get("order") != list(ORDER)
            or plan.get("baseline_window_kld") != BASELINE or plan.get("baseline_mean_kld") != BASELINE_MEAN
            or plan.get("raw_capture_bytes") != RAW_BYTES or plan.get("headroom_bytes") != HEADROOM
            or plan.get("protected_roles_opened") != [] or plan.get("opened_roles") != ["conditional-fit"]
            or plan.get("runtime", {}).get("tail_repair") != "tail-after-valid-history-v2"
            or plan.get("runtime", {}).get("physical_bpw") != 4.25 or plan.get("runtime", {}).get("mtp") is not False):
        raise ValueError("tail-repair V2 fixed contract differs")
    if set(plan.get("source_sha256", {})) != source_files():
        raise ValueError("tail-repair V2 source inventory differs")
    for name, expected in plan["source_sha256"].items():
        if sha(REPO / name) != expected:
            raise ValueError(f"tail-repair V2 source changed: {name}")
    image, device = load_image_receipt(), load_device_receipt()
    if (plan["capture_image"] != image["image_id"] or plan["patched_kpool_sha256"] != image["patched_kpool_sha256"]
            or sha(IMAGE_RECEIPT) != plan["image_receipt_sha256"] or sha(DEVICE_RECEIPT) != plan["device_row_closure_sha256"]
            or sha(PREREG) != plan["prereg_sha256"] or sha(prior.RECIPE) != plan["recipe_sha256"]
            or sha(AMENDMENT) != plan["amendment_sha256"]
            or sha(prior.ROLES) != plan["roles_sha256"] or plan.get("intervention_integrity", {}).get("status") != "pass"):
        raise ValueError("tail-repair V2 image, device closure, preregistration, recipe, or role changed")
    windows = prior.selected_windows()
    if windows != plan["windows"] or prior.input_stats(windows) != plan["input_stats"]:
        raise ValueError("tail-repair V2 inputs changed")
    prior.verify_inputs(plan)
    prior.load_recipe()
    return plan


def runtime_audit(log: str, arm: str, completed_windows=None) -> dict:
    return {**prior.ordered.runtime_audit(log, arm, completed_windows), "tail_repair": "tail-after-valid-history-v2"}


@contextmanager
def patched_prior():
    names = ("PREFIX", "PORT", "PARENT_IMAGE", "IMAGE_RECEIPT", "PREREG", "source_files", "load_image_receipt", "authenticate", "runtime_audit")
    saved = {name: getattr(prior, name) for name in names}
    try:
        prior.PREFIX, prior.PORT = PREFIX, PORT
        prior.PARENT_IMAGE, prior.IMAGE_RECEIPT, prior.PREREG = PARENT_IMAGE, IMAGE_RECEIPT, PREREG
        prior.source_files, prior.load_image_receipt = source_files, load_image_receipt
        prior.authenticate, prior.runtime_audit = authenticate, runtime_audit
        yield
    finally:
        for name, value in saved.items():
            setattr(prior, name, value)


def run(path: Path) -> dict:
    with patched_prior():
        return prior.run(path)


def analyze(path: Path, output: Path) -> dict:
    authenticate(path)
    with patched_prior():
        return prior.analyze(path, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    planning = subs.add_parser("plan"); planning.add_argument("--plan", type=Path, required=True); planning.add_argument("--output", type=Path, required=True)
    running = subs.add_parser("run"); running.add_argument("--plan", type=Path, required=True)
    scoring = subs.add_parser("analyze"); scoring.add_argument("--plan", type=Path, required=True); scoring.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = make_plan(args.plan, args.output) if args.command == "plan" else run(args.plan) if args.command == "run" else analyze(args.plan, args.output)
    print(json.dumps(result, sort_keys=True))
