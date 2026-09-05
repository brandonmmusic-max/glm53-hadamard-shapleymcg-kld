"""P8 and production-EXL3 forced-decode control using FP8 MLA KV."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from dataclasses import fields
import json
import os
from pathlib import Path
import re
import shlex

import numpy as np

from . import decode_path_control as prior
from . import p8_decode_analysis as metric
from . import p8_decode_launcher as base
from . import p8_decode_protocol as protocol


REPO = Path(__file__).resolve().parents[1]
ROOT = Path("/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6")
PREFIX = "glm53-p8-exl3-fp8-control-v1"
PORT = 8027
ORDER = ({"stage": "control", "arm": "stock"}, {"stage": "control", "arm": "exl3"})
WINDOW_IDS = prior.WINDOW_IDS
ROLES, TEACHER = prior.ROLES, prior.TEACHER
PREREG = REPO / "experiments/p8-exl3-fp8-ds-mla-forced-decode-v1-prereg.json"
AMENDMENT = REPO / "experiments/p8-exl3-fp8-ds-mla-forced-decode-v2-amendment.json"
ATTENTION_BACKEND = "FLASHINFER_MLA_SPARSE_SM120"
P8_RECEIPT = ROOT / "p8-smallm-scheduler-v1/decode-capture-image-v2/receipt.json"
EXL3_RECEIPT = ROOT / "decode-path-control-images-v1/exl3/receipt.json"
P8_RECIPE = ROOT / "p8-smallm-scheduler-v1/fc1-integrated-v1/container-final.private.json"
EXL3_RECIPE = ROOT / "uniform-p8-all42-v1/speed-v2a3/round-01-exl3/container.json"
P8_RECIPE_SHA = "c7f5832af2244926ebf0f2d481efbe7461bee85afb18cc4d85dd3769d44ffdf8"
EXL3_RECIPE_SHA = "5c0e4de4b4494be7a4a65ad349c62ca54c97e4d65cef08311ec6821cc3d634d9"
RAW_BYTES = 2 * 4 * protocol.ROWS * protocol.VOCAB_LIMIT * 4
HEADROOM = 20 * 2**30
BASELINES = {
    "stock": dict(prior.P8_WINDOW_KLD),
    "exl3": {
        "conditional-fit-0032": 0.12777844900697938,
        "conditional-fit-0021": 0.17043228583393247,
        "conditional-fit-0090": 0.07371076785152708,
        "conditional-fit-0003": 0.07947765461782906,
    },
}
BASELINE_MEANS = {arm: float(np.mean(list(rows.values()))) for arm, rows in BASELINES.items()}
PRODUCT = {"stock": "p8", "exl3": "exl3"}
_P8_CLONE = base.clone_argv
_EXL3_CLONE = prior.clone_argv
_PRIOR_LOAD_IMAGES = prior.load_image_receipts


def sha(path: Path) -> str:
    return protocol.sha(path)


def source_files() -> set[str]:
    return {
        "glm53_nvfp4/p8_exl3_fp8_decode_control.py",
        "tests/test_p8_exl3_fp8_decode_control.py",
        "glm53_nvfp4/decode_path_control.py",
        "glm53_nvfp4/p8_decode_launcher.py",
        "glm53_nvfp4/p8_decode_protocol.py",
        "glm53_nvfp4/p8_decode_analysis.py",
        "experiments/p8-exl3-fp8-ds-mla-forced-decode-v1-prereg.json",
        "experiments/p8-exl3-fp8-ds-mla-forced-decode-v2-amendment.json",
    }


def selected_windows() -> list[dict]:
    return prior.selected_windows()


def image_receipts() -> dict[str, dict]:
    p8_raw = json.loads(P8_RECEIPT.read_text())
    p8 = base.verify_image_receipt(P8_RECEIPT, p8_raw["image_id"])
    exl3 = _PRIOR_LOAD_IMAGES()["exl3"]
    return {"stock": p8, "exl3": exl3}


def recipes() -> dict[str, dict]:
    if sha(P8_RECIPE) != P8_RECIPE_SHA or sha(EXL3_RECIPE) != EXL3_RECIPE_SHA:
        raise ValueError("production recipe identity differs")
    p8 = json.loads(P8_RECIPE.read_text())
    exl3_rows = json.loads(EXL3_RECIPE.read_text())
    if len(exl3_rows) != 1:
        raise ValueError("EXL3 recipe inventory differs")
    base.cold.validate_recipe(p8, "p8")
    return {"stock": p8, "exl3": exl3_rows[0]}


def input_stats(windows: list[dict]) -> dict:
    paths = {ROLES}
    for window in windows:
        paths.add(Path(window["token_path"]))
        paths.add(TEACHER / window["teacher_path"])
    return {str(path): base.fingerprint(path) for path in sorted(paths)}


def make_plan(path: Path, output: Path) -> dict:
    if path != path.resolve() or output != output.resolve() or path.exists() or path.with_suffix(".sha256").exists() or output.exists() or output.parent != ROOT:
        raise ValueError("fresh canonical plan, seal, and NVMe output required")
    if json.loads(PREREG.read_text()).get("status") != "predeclared-before-plan-seal-or-gpu-run":
        raise ValueError("FP8 control preregistration differs")
    receipts, windows = image_receipts(), selected_windows()
    recipes()
    plan = {
        "schema": "glm53.p8-exl3-fp8-ds-mla-control-plan.v2",
        "created_at": base.pilot.now(), "output": str(output), "order": list(ORDER), "port": PORT,
        "windows": windows, "roles": str(ROLES), "roles_sha256": sha(ROLES),
        "teacher_root": str(TEACHER), "input_stats": input_stats(windows),
        "images": {arm: row["image_id"] for arm, row in receipts.items()},
        "image_receipts": {
            "stock": {"path": str(P8_RECEIPT), "sha256": sha(P8_RECEIPT)},
            "exl3": {"path": str(EXL3_RECEIPT), "sha256": sha(EXL3_RECEIPT)},
        },
        "recipes": {
            "stock": {"path": str(P8_RECIPE), "sha256": P8_RECIPE_SHA},
            "exl3": {"path": str(EXL3_RECIPE), "sha256": EXL3_RECIPE_SHA},
        },
        "source_sha256": {name: sha(REPO / name) for name in sorted(source_files())},
        "prereg": str(PREREG), "prereg_sha256": sha(PREREG),
        "amendment": str(AMENDMENT), "amendment_sha256": sha(AMENDMENT),
        "product_labels": PRODUCT, "baseline_window_kld": BASELINES,
        "baseline_mean_kld": BASELINE_MEANS, "raw_capture_bytes": RAW_BYTES,
        "headroom_bytes": HEADROOM, "ready_timeout_seconds": 2400,
        "request_timeout_seconds": 900, "capture_artifact_owner": {"uid": os.getuid(), "gid": os.getgid()},
        "runtime": {
            "kv_cache_dtype": "fp8_ds_mla", "attention_backend": ATTENTION_BACKEND,
            "cuda_graphs": True, "mtp": False, "max_num_seqs": 1,
            "stock": {"product": "p8", "tp": 4, "ep": False, "dcp": 1, "tile_n": 64, "fused_scratch": True},
            "exl3": {"product": "exl3", "tp": 4, "ep": True, "dcp": 4, "moe_backend": "b12x"},
        },
        "decision": {
            "nvfp4_production_mla_stack_implicated": "both relative improvements >=0.20 and both FP8 means <0.09",
            "fp8_path_does_not_exonerate_stack": "both relative improvements <0.20 and both FP8 means >=0.09",
            "otherwise": "arm-specific-or-inconclusive",
        },
        "opened_roles": ["conditional-fit"], "protected_roles_opened": [],
        "speed_measurement_valid": False, "retry_policy": "one amended v2 execution; no silent retry or substitution",
        "causal_boundary": "FP8-compatible attention-plus-KV path versus the B12X/NVFP4 production baseline; not KV dtype alone",
    }
    base.pilot.save(path, plan)
    base.pilot.save(path.with_suffix(".sha256"), sha(path) + "  " + path.name + "\n")
    return plan


def authenticate(path: Path) -> dict:
    if path != path.resolve() or not path.is_file() or not path.with_suffix(".sha256").is_file() or sha(path) != path.with_suffix(".sha256").read_text().split()[0]:
        raise ValueError("canonical sealed FP8 control plan required")
    plan = json.loads(path.read_text())
    if (
        plan.get("schema") != "glm53.p8-exl3-fp8-ds-mla-control-plan.v2"
        or plan.get("order") != list(ORDER) or plan.get("product_labels") != PRODUCT
        or plan.get("baseline_window_kld") != BASELINES or plan.get("baseline_mean_kld") != BASELINE_MEANS
        or plan.get("raw_capture_bytes") != RAW_BYTES or plan.get("headroom_bytes") != HEADROOM
        or plan.get("opened_roles") != ["conditional-fit"] or plan.get("protected_roles_opened") != []
        or plan.get("runtime", {}).get("kv_cache_dtype") != "fp8_ds_mla"
        or plan.get("runtime", {}).get("attention_backend") != ATTENTION_BACKEND
        or plan.get("runtime", {}).get("mtp") is not False
    ):
        raise ValueError("FP8 control fixed contract differs")
    if set(plan.get("source_sha256", {})) != source_files():
        raise ValueError("FP8 control source inventory differs")
    for name, expected in plan["source_sha256"].items():
        if sha(REPO / name) != expected:
            raise ValueError(f"FP8 control source changed: {name}")
    receipts = image_receipts()
    if plan["images"] != {arm: row["image_id"] for arm, row in receipts.items()}:
        raise ValueError("capture image identity differs")
    for arm, item in plan["image_receipts"].items():
        if sha(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"{arm} capture receipt changed")
    recipes()
    if (sha(PREREG) != plan["prereg_sha256"] or sha(AMENDMENT) != plan["amendment_sha256"]
            or sha(ROLES) != plan["roles_sha256"]):
        raise ValueError("preregistration, amendment, or role identity changed")
    windows = selected_windows()
    if windows != plan["windows"] or input_stats(windows) != plan["input_stats"]:
        raise ValueError("fixed window inputs changed")
    return plan


def model_name(arm: str) -> str:
    if arm == "n64":
        arm = "stock"
    if arm not in PRODUCT:
        raise ValueError("unknown FP8 control arm")
    return f"{PREFIX}-{PRODUCT[arm]}"


def slot_name(entry: dict) -> str:
    arm = "stock" if entry["arm"] == "n64" else entry["arm"]
    if arm not in PRODUCT:
        raise ValueError("unknown FP8 control slot")
    return arm


def replace_option(command: str, option: str, value: str) -> str:
    tokens = shlex.split(command)
    if tokens.count(option) != 1:
        raise ValueError(f"production recipe option differs: {option}")
    tokens[tokens.index(option) + 1] = value
    return shlex.join(tokens)


def clone_argv(_recipe: dict, image: str, entry: dict, out: Path, windows: list[dict], owner: str) -> list[str]:
    runtime_recipes = recipes()
    if entry["arm"] == "stock":
        argv = _P8_CLONE(runtime_recipes["stock"], image, {"stage": "control", "arm": "n64"}, out, windows, owner)
    elif entry["arm"] == "exl3":
        argv = _EXL3_CLONE(runtime_recipes["exl3"], image, entry, out, windows, owner)
    else:
        raise ValueError("unknown FP8 control arm")
    argv[-1] = replace_option(argv[-1], "--kv-cache-dtype", "fp8_ds_mla")
    argv[-1] = replace_option(argv[-1], "--attention-backend", ATTENTION_BACKEND)
    return argv


def runtime_audit(log: str, arm: str, completed_windows=None) -> dict:
    for marker in ("Using V2 Model Runner", "tensor_parallel_size=4", "speculative_config=None",
                   "kv_cache_dtype=fp8_ds_mla", ATTENTION_BACKEND):
        if marker not in log:
            raise ValueError(f"missing runtime marker: {marker}")
    if arm == "stock":
        proof = base.pilot.verify_runtime(log, 64, True)
        if "decode_context_parallel_size=1" not in log or "'enable_expert_parallel': True" in log:
            raise ValueError("P8 DCP1/no-EP runtime differs")
    elif arm == "exl3":
        proof = {}
        if "decode_context_parallel_size=4" not in log or "'enable_expert_parallel': True" not in log:
            raise ValueError("EXL3 DCP4/EP4 runtime differs")
        if "quantization=exl3" not in log and "'quantization': 'exl3'" not in log:
            raise ValueError("EXL3 quantization marker missing")
        if "moe_backend='b12x'" not in log and "'moe_backend': 'b12x'" not in log:
            raise ValueError("EXL3 B12X marker missing")
    else:
        raise ValueError("unknown FP8 control arm")
    ready = re.findall(r"GLM53_P8_DECODE_CAPTURE_V2_READY tp_rank=(\d+) max_num_reqs=(\d+) real_vocab=(\d+) expected_outputs=(\d+)", log)
    if len(ready) != 4 or set(ready) != {(str(rank), "1", str(protocol.VOCAB_LIMIT), str(protocol.ROWS)) for rank in range(4)}:
        raise ValueError("four-rank capture readiness differs")
    closed = re.findall(r"GLM53_P8_DECODE_CAPTURE_V2_WARMUP_SCOPE_CLOSED tp_rank=(\d+) registrations=(\d+) samples=(\d+)", log)
    if len(closed) != 4 or set(closed) != {(str(rank), "1", "2") for rank in range(4)}:
        raise ValueError("capture warmup closure differs")
    result = {**proof, "arm": arm, "product": PRODUCT[arm], "tp": 4,
              "kv_cache_dtype": "fp8_ds_mla", "attention_backend": ATTENTION_BACKEND,
              "mtp": False, "cuda_graphs": True}
    if completed_windows is not None:
        completed = re.findall(r"GLM53_P8_DECODE_CAPTURE_V2_COMPLETE window=(conditional-fit-\d{4}) rows=(\d+) tp_rank=(\d+)", log)
        expected = {(row["id"], str(protocol.ROWS), "0") for row in completed_windows}
        if len(completed) != len(expected) or set(completed) != expected:
            raise ValueError("completed capture inventory differs")
        result["complete_window_ids"] = [row["id"] for row in completed_windows]
    return result


@contextmanager
def patched_prior():
    names = ("PREFIX", "PORT", "ORDER", "P8_WINDOW_KLD", "P8_MEAN", "RAW_BYTES", "HEADROOM", "PREREG",
             "load_image_receipts", "load_recipe", "source_files", "authenticate", "clone_argv", "runtime_audit",
             "model_name", "slot_name")
    saved = {name: getattr(prior, name) for name in names}
    try:
        prior.PREFIX, prior.PORT, prior.ORDER = PREFIX, PORT, ORDER
        prior.P8_WINDOW_KLD, prior.P8_MEAN = BASELINES["stock"], BASELINE_MEANS["stock"]
        prior.RAW_BYTES, prior.HEADROOM, prior.PREREG = RAW_BYTES, HEADROOM, PREREG
        prior.load_image_receipts = image_receipts
        prior.load_recipe = lambda: {}
        prior.source_files = source_files
        prior.authenticate = authenticate
        prior.clone_argv, prior.runtime_audit = clone_argv, runtime_audit
        prior.model_name, prior.slot_name = model_name, slot_name
        yield
    finally:
        for name, value in saved.items():
            setattr(prior, name, value)


def run(path: Path) -> dict:
    with patched_prior():
        return prior.run(path)


def analyze(path: Path, output: Path) -> dict:
    import torch
    torch.set_num_threads(metric.CPU_THREADS)
    plan = authenticate(path)
    root = Path(plan["output"])
    execution = json.loads((root / "execution.json").read_text())
    if execution.get("exit_code") != 0 or execution.get("capture_protocol_complete") is not True or output.exists() or output != output.resolve():
        raise ValueError("successful fresh two-arm capture and analysis destination required")
    output.mkdir(parents=True, mode=0o700)
    records = []
    for arm in ("stock", "exl3"):
        stage = json.loads((root / arm / "execution.json").read_text())
        if stage.get("exit_code") != 0 or [row["id"] for row in stage["windows"]] != list(WINDOW_IDS):
            raise ValueError(f"{arm} FP8 stage is incomplete")
        for window in plan["windows"]:
            wid = window["id"]
            tokens = np.load(window["token_path"], allow_pickle=False)
            student, metadata = protocol.load_capture(root / arm / "captures", wid, tokens)
            teacher_path = Path(plan["teacher_root"]) / window["teacher_path"]
            if sha(teacher_path) != window["teacher_sha256"]:
                raise ValueError("teacher changed")
            teacher = metric._load_teacher(teacher_path, window)
            scores = protocol.score_aligned(teacher, student, tokens)
            with (output / f"{wid}.{PRODUCT[arm]}.npz").open("xb") as stream:
                np.savez(stream, **{field.name: getattr(scores, field.name) for field in fields(scores)})
            mean = float(scores.kld.mean())
            row = {"arm": PRODUCT[arm], "window_id": wid, "domain": window["domain"], "mean_kld": mean,
                   "true_decode_mean_kld": float(scores.kld[1:].mean()), "one_token_prefill_kld": float(scores.kld[0]),
                   "nvfp4_ds_mla_baseline_kld": BASELINES[arm][wid],
                   "fp8_minus_nvfp4_kld": mean - BASELINES[arm][wid], "raw_sha256": metadata["raw_sha256"]}
            records.append(row)
            base.pilot.save(output / f"{wid}.{PRODUCT[arm]}.json", row)
            print(json.dumps(row, sort_keys=True), flush=True)
            del teacher, student, scores
    arms = {}
    for internal, product in PRODUCT.items():
        rows = [row for row in records if row["arm"] == product]
        mean = float(np.mean([row["mean_kld"] for row in rows]))
        deltas = [row["fp8_minus_nvfp4_kld"] for row in rows]
        arms[product] = {"fp8_mean_kld": mean, "nvfp4_mean_kld": BASELINE_MEANS[internal],
                         "mean_delta_kld": float(np.mean(deltas)),
                         "relative_improvement": (BASELINE_MEANS[internal] - mean) / BASELINE_MEANS[internal],
                         "window_wins": sum(value < 0 for value in deltas),
                         "paired_delta_ci95_percentile": prior.bootstrap(deltas, 20260905),
                         "true_decode_mean_kld": float(np.mean([row["true_decode_mean_kld"] for row in rows]))}
    implicated = all(row["relative_improvement"] >= 0.20 and row["fp8_mean_kld"] < 0.09 for row in arms.values())
    exonerated = all(row["relative_improvement"] < 0.20 and row["fp8_mean_kld"] >= 0.09 for row in arms.values())
    verdict = ("nvfp4-production-mla-stack-implicated" if implicated else
               "fp8-path-does-not-exonerate-stack" if exonerated else "arm-specific-or-inconclusive")
    result = {"schema": "glm53.p8-exl3-fp8-ds-mla-control-analysis.v1", "status": "complete", "verdict": verdict,
              "plan_sha256": sha(path), "metric": "KL(teacher || student), nats, FP64 CPU, all154880 entries",
              "windows": 4, "arms": arms, "records": records, "opened_roles": ["conditional-fit"],
              "protected_roles_opened": [], "speed_measurement_valid": False, "mtp": False,
              "claim_boundary": "FP8-compatible attention-plus-KV path versus B12X/NVFP4 within product; not KV dtype alone, full32 quality, or speed qualification."}
    result["files"] = {item.name: sha(item) for item in output.iterdir() if item.is_file()}
    base.pilot.save(output / "analysis.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    planning = subs.add_parser("plan"); planning.add_argument("--plan", type=Path, required=True); planning.add_argument("--output", type=Path, required=True)
    running = subs.add_parser("run"); running.add_argument("--plan", type=Path, required=True)
    scoring = subs.add_parser("analyze"); scoring.add_argument("--plan", type=Path, required=True); scoring.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = make_plan(args.plan, args.output) if args.command == "plan" else run(args.plan) if args.command == "run" else analyze(args.plan, args.output)
    print(json.dumps(result, sort_keys=True))
