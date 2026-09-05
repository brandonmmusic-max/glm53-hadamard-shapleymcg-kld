"""Four-window stock-NVFP4/production-EXL3 forced-decode control."""
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
import shlex
import shutil
import signal
import time

import numpy as np

from . import p8_decode_analysis as metric
from . import p8_decode_launcher as base
from . import p8_decode_protocol as protocol


REPO = Path(__file__).resolve().parents[1]
ROOT = Path("/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6")
PREFIX = "glm53-decode-path-control-v1"
PORT = 8024
PREREG = REPO / "experiments/decode-path-matched-control-v1-prereg.json"
AMENDMENT = REPO / "experiments/decode-path-matched-control-v2-amendment.json"
WINDOW_IDS = ("conditional-fit-0032", "conditional-fit-0021", "conditional-fit-0090", "conditional-fit-0003")
P8_WINDOW_KLD = {
    "conditional-fit-0032": 0.097427959573676,
    "conditional-fit-0021": 0.20159250798813763,
    "conditional-fit-0090": 0.09817271669377985,
    "conditional-fit-0003": 0.07270314982124808,
}
P8_MEAN = 0.11747408351921039
NEAR = (0.09, 0.15)
RAW_BYTES = 2 * 4 * protocol.ROWS * protocol.VOCAB_LIMIT * 4
HEADROOM = 20 * 2**30
ORDER = ({"stage": "control", "arm": "stock"}, {"stage": "control", "arm": "exl3"})
MODEL_ROOTS = {
    "stock": Path("/home/brandonmusic/models/GLM-5.3-Flash-NVFP4"),
    "exl3": Path("/home/brandonmusic/models/GLM-5.3-Flash-EXL3-4bpw"),
}
CACHE_ROOTS = {
    "stock": Path("/home/brandonmusic/KLC_SANDBOXES/glm53-exl3-k4-sm120/cache-dflash2-nvfp4-v77"),
    "exl3": Path("/home/brandonmusic/KLC_SANDBOXES/glm53-exl3-k4-sm120/cache-mtp5-tp4-v84"),
}
PARENTS = {
    "stock": "sha256:ed027a3a2ff93b9cf60c95f7adfaf676cabc8e040a28cffa7486a262c82fdfbe",
    "exl3": "sha256:d1b6c021df11056cebde469cadc05f55cb21ec4ffc8b54ae6f08161df0c493bf",
}
IMAGE_RECEIPTS = {
    arm: ROOT / f"decode-path-control-images-v1/{arm}/receipt.json" for arm in PARENTS
}
RECIPE = ROOT / "uniform-p8-all42-v1/speed-v2a3/round-01-exl3/container.json"
RECIPE_SHA256 = "5c0e4de4b4494be7a4a65ad349c62ca54c97e4d65cef08311ec6821cc3d634d9"
ROLES = Path("/media/brandonmusic/klcstore/bmxfp4-glm53/roles/roles-codec-conditional-fit32-v1.json")
TEACHER = Path("/media/brandonmusic/klcstore/bmxfp4-glm53/teacher")


def source_files() -> set[str]:
    return {
        "glm53_nvfp4/decode_path_control.py",
        "tests/test_decode_path_control.py",
        "glm53_nvfp4/p8_decode_launcher.py",
        "glm53_nvfp4/p8_decode_protocol.py",
        "glm53_nvfp4/p8_decode_analysis.py",
        "scripts/build_decode_path_control_image.py",
        "runtime_patch/p8_decode_capture/Dockerfile.stock-control",
        "runtime_patch/p8_decode_capture/Dockerfile.exl3-control",
        "runtime_patch/p8_decode_capture/__init__.py",
        "runtime_patch/p8_decode_capture/processor.py",
        "runtime_patch/p8_decode_capture/v2_hook.py",
        "runtime_patch/p8_decode_capture/sampler-v2-hook.patch",
        "runtime_patch/p8_decode_capture/warmup-v2-hook.patch",
        "experiments/decode-path-matched-control-v1-prereg.json",
        "experiments/decode-path-matched-control-v2-amendment.json",
    }


def sha(path: Path) -> str:
    return protocol.sha(path)


def load_image_receipts() -> dict:
    rows = {}
    for arm, path in IMAGE_RECEIPTS.items():
        row = json.loads(path.read_text())
        if (row.get("schema") != "glm53.decode-path-control-image.v1"
                or row.get("status") != "complete" or row.get("arm") != arm
                or row.get("parent_image_id") != PARENTS[arm]
                or not re.fullmatch(r"sha256:[a-f0-9]{64}", row.get("image_id", ""))
                or row.get("gpu_used") is not False or row.get("speed_measurement_valid") is not False):
            raise ValueError(f"invalid {arm} control image receipt")
        rows[arm] = row
    return rows


def load_recipe() -> dict:
    if sha(RECIPE) != RECIPE_SHA256:
        raise ValueError("production EXL3 recipe changed")
    rows = json.loads(RECIPE.read_text())
    if len(rows) != 1 or rows[0].get("Image") != PARENTS["exl3"]:
        raise ValueError("production EXL3 recipe identity differs")
    return rows[0]


def selected_windows() -> list[dict]:
    windows = protocol.load_role_inputs(ROLES, TEACHER, verify_teacher_bytes=True)
    by_id = {row["id"]: row for row in windows}
    result = [by_id[wid] for wid in WINDOW_IDS]
    if [row["domain"] for row in result] != [
        "axis1_general", "axis2_legal", "axis3_code_agentic", "axis4_reasoning_termination"
    ]:
        raise ValueError("fixed control panel is not one-per-domain")
    return result


def make_plan(path: Path, output: Path) -> dict:
    if path != path.resolve() or output != output.resolve() or path.exists() or path.with_suffix(".sha256").exists() or output.exists():
        raise ValueError("fresh canonical plan, seal, and output required")
    if output.parent != ROOT:
        raise ValueError("control output must use the budgeted NVMe root")
    prereg = json.loads(PREREG.read_text())
    if prereg.get("status") != "predeclared-before-control-image-build-or-GPU-run":
        raise ValueError("pre-registration identity differs")
    receipts, windows = load_image_receipts(), selected_windows()
    recipe = load_recipe()
    plan = {
        "schema": "glm53.decode-path-matched-control-plan.v2",
        "created_at": base.pilot.now(), "output": str(output), "order": list(ORDER),
        "port": PORT, "windows": windows, "roles": str(ROLES), "roles_sha256": sha(ROLES),
        "teacher_root": str(TEACHER), "raw_capture_bytes": RAW_BYTES,
        "headroom_bytes": HEADROOM, "destination_filesystem": str(ROOT),
        "images": {arm: row["image_id"] for arm, row in receipts.items()},
        "image_receipts": {arm: {"path": str(IMAGE_RECEIPTS[arm]), "sha256": sha(IMAGE_RECEIPTS[arm])} for arm in receipts},
        "parent_images": PARENTS, "recipe": str(RECIPE), "recipe_sha256": RECIPE_SHA256,
        "model_roots": {arm: str(path) for arm, path in MODEL_ROOTS.items()},
        "model_index_sha256": {arm: sha(root / "model.safetensors.index.json") for arm, root in MODEL_ROOTS.items()},
        "cache_roots": {arm: str(path) for arm, path in CACHE_ROOTS.items()},
        "source_sha256": {name: sha(REPO / name) for name in sorted(source_files())},
        "prereg": str(PREREG), "prereg_sha256": sha(PREREG),
        "amendment": str(AMENDMENT), "amendment_sha256": sha(AMENDMENT),
        "p8_reference_window_kld": P8_WINDOW_KLD, "p8_reference_mean_kld": P8_MEAN,
        "near_interval": list(NEAR), "ready_timeout_seconds": 2400, "request_timeout_seconds": 900,
        "capture_artifact_owner": {"uid": os.getuid(), "gid": os.getgid()},
        "runtime": {"tp": 4, "ep": True, "dcp": 4, "attention_backend": "B12X_MLA_SPARSE",
                    "kv_cache_dtype": "nvfp4_ds_mla", "cuda_graphs": True, "mtp": False, "max_num_seqs": 1,
                    "arm_moe_backends": {"stock": "humming", "exl3": "b12x"}},
        "decision": "both all-row means in [0.09,0.15]; shared serving bug additionally requires each control to improve less than20% versus paired P8",
        "opened_roles": ["conditional-fit"], "protected_roles_opened": [],
        "speed_measurement_valid": False, "retry_policy": "no silent retry or substitution",
    }
    base.pilot.save(path, plan)
    base.pilot.save(path.with_suffix(".sha256"), sha(path) + "  " + path.name + "\n")
    return plan


def authenticate(path: Path) -> dict:
    if path != path.resolve() or not path.is_file() or not path.with_suffix(".sha256").is_file():
        raise ValueError("canonical sealed plan required")
    if sha(path) != path.with_suffix(".sha256").read_text().split()[0]:
        raise ValueError("control plan seal differs")
    plan = json.loads(path.read_text())
    if (plan.get("schema") != "glm53.decode-path-matched-control-plan.v2"
            or plan.get("order") != list(ORDER) or plan.get("raw_capture_bytes") != RAW_BYTES
            or plan.get("headroom_bytes") != HEADROOM or tuple(plan.get("near_interval", [])) != NEAR
            or plan.get("p8_reference_window_kld") != P8_WINDOW_KLD
            or plan.get("p8_reference_mean_kld") != P8_MEAN
            or plan.get("protected_roles_opened") != [] or plan.get("opened_roles") != ["conditional-fit"]):
        raise ValueError("fixed control contract differs")
    if plan.get("source_sha256", {}).keys() != source_files():
        raise ValueError("control source inventory differs")
    for name, expected in plan["source_sha256"].items():
        if sha(REPO / name) != expected:
            raise ValueError(f"control source changed: {name}")
    if (sha(PREREG) != plan["prereg_sha256"] or sha(AMENDMENT) != plan["amendment_sha256"]
            or sha(RECIPE) != plan["recipe_sha256"]):
        raise ValueError("preregistration, amendment, or recipe changed")
    receipts = load_image_receipts()
    if plan["images"] != {arm: row["image_id"] for arm, row in receipts.items()}:
        raise ValueError("control image identity differs")
    for arm in receipts:
        if sha(IMAGE_RECEIPTS[arm]) != plan["image_receipts"][arm]["sha256"]:
            raise ValueError("control image receipt changed")
        root = MODEL_ROOTS[arm]
        if sha(root / "model.safetensors.index.json") != plan["model_index_sha256"][arm]:
            raise ValueError("model index changed")
    windows = selected_windows()
    if windows != plan["windows"] or sha(ROLES) != plan["roles_sha256"]:
        raise ValueError("fixed window inputs changed")
    load_recipe()
    return plan


def model_name(arm: str) -> str:
    return f"{PREFIX}-{arm}"


def slot_name(entry: dict) -> str:
    if entry not in ORDER:
        raise ValueError("undeclared control arm")
    return entry["arm"]


def clone_argv(recipe: dict, image: str, entry: dict, out: Path, windows: list[dict], owner: str) -> list[str]:
    arm = entry["arm"]
    if image != authenticate_image_id(arm):
        raise ValueError("arm/image mismatch")
    config, host = recipe["Config"], recipe["HostConfig"]
    tokens = shlex.split(config["Cmd"][1])
    replacements = {
        "--port": str(PORT), "--served-model-name": model_name(arm), "--tensor-parallel-size": "4",
        "--decode-context-parallel-size": "4", "--attention-backend": "B12X_MLA_SPARSE",
        "--kv-cache-dtype": "nvfp4_ds_mla", "--max-num-seqs": "1",
        "--quantization": "modelopt" if arm == "stock" else "exl3", "--load-format": "safetensors",
        "--moe-backend": "humming" if arm == "stock" else "b12x",
    }
    for option, value in replacements.items():
        if tokens.count(option) != 1:
            raise ValueError(f"production recipe option differs: {option}")
        tokens[tokens.index(option) + 1] = value
    if "--enable-expert-parallel" not in tokens or any("speculat" in token or token == "--enforce-eager" for token in tokens):
        raise ValueError("control must retain EP4 graphs without speculation")
    overrides = {
        "VLLM_USE_V2_MODEL_RUNNER": "1", "GLM53_P8_DECODE_CAPTURE_V2": "1",
        "GLM53_P8_DECODE_CAPTURE_ROOT": "/p8-captures",
        "GLM53_P8_DECODE_CAPTURE_ALLOWED_WINDOW_IDS": ",".join(row["id"] for row in windows),
        "GLM53_P8_DECODE_CAPTURE_EXPECTED_OUTPUT_TOKENS": str(protocol.ROWS),
    }
    env = [value for value in config["Env"] if value.split("=", 1)[0] not in overrides]
    env.extend(f"{key}={value}" for key, value in overrides.items())
    name = f"{PREFIX}-{slot_name(entry)}"
    argv = ["docker", "create", "--name", name, "--cidfile", str(out / "container.cid"),
            "--label", f"{base.cold.LABEL}={owner}", "--network", "host", "--ipc", "host",
            "--shm-size", str(host["ShmSize"]), "--gpus", "all", "--runtime", host["Runtime"],
            "--restart", "no", "--security-opt", "label=disable", "--workdir", "/",
            "--entrypoint", "/bin/bash"]
    for value in env:
        argv += ["--env", value]
    argv += ["--volume", f"{MODEL_ROOTS[arm]}:/model:ro", "--volume", f"{CACHE_ROOTS[arm]}:/cache:rw",
             "--volume", f"{out / 'captures'}:/p8-captures:rw"]
    return [*argv, image, "-lc", shlex.join(tokens)]


def authenticate_image_id(arm: str) -> str:
    return load_image_receipts()[arm]["image_id"]


def runtime_audit(log: str, arm: str, completed_windows=None) -> dict:
    for marker in ("Using V2 Model Runner", "tensor_parallel_size=4", "decode_context_parallel_size=4",
                   "speculative_config=None", "kv_cache_dtype=nvfp4_ds_mla"):
        if marker not in log:
            raise ValueError(f"missing runtime marker: {marker}")
    if "'enable_expert_parallel': True" not in log:
        raise ValueError("EP4 runtime marker missing")
    quant = "modelopt" if arm == "stock" else "exl3"
    if f"quantization={quant}" not in log and f"'quantization': '{quant}'" not in log:
        raise ValueError("quantization runtime marker missing")
    backend = "humming" if arm == "stock" else "b12x"
    if f"moe_backend='{backend}'" not in log and f"'moe_backend': '{backend}'" not in log:
        raise ValueError("MoE backend runtime marker missing")
    ready = re.findall(r"GLM53_P8_DECODE_CAPTURE_V2_READY tp_rank=(\d+) max_num_reqs=(\d+) real_vocab=(\d+) expected_outputs=(\d+)", log)
    expected_ready = {(str(rank), "1", str(protocol.VOCAB_LIMIT), str(protocol.ROWS)) for rank in range(4)}
    if len(ready) != 4 or set(ready) != expected_ready:
        raise ValueError("four-rank capture readiness proof differs")
    closed = re.findall(r"GLM53_P8_DECODE_CAPTURE_V2_WARMUP_SCOPE_CLOSED tp_rank=(\d+) registrations=(\d+) samples=(\d+)", log)
    if len(closed) != 4 or set(closed) != {(str(rank), "1", "2") for rank in range(4)}:
        raise ValueError("capture warmup closure differs")
    result = {"arm": arm, "tp": 4, "ep": True, "dcp": 4, "quantization": quant,
              "capture_ready_ranks": list(range(4)), "mtp": False, "cuda_graphs": True,
              "moe_backend": backend}
    if completed_windows is not None:
        completed = re.findall(r"GLM53_P8_DECODE_CAPTURE_V2_COMPLETE window=(conditional-fit-\d{4}) rows=(\d+) tp_rank=(\d+)", log)
        expected = {(row["id"], str(protocol.ROWS), "0") for row in completed_windows}
        if len(completed) != len(expected) or set(completed) != expected:
            raise ValueError("completed capture inventory differs")
        result["complete_window_ids"] = [row["id"] for row in completed_windows]
    return result


def restoration_safety(plan: dict, digest: str) -> dict:
    """Prove no owned control container remains live before restoring services."""
    result = {"ok": False, "errors": [], "containers": []}
    try:
        expected = {f"/{PREFIX}-{arm}": arm for arm in ("stock", "exl3")}
        ids = base.pilot.command([
            "docker", "ps", "-a", "--no-trunc", "--filter", f"name=^/{PREFIX}-",
            "--format", "{{.ID}}",
        ]).stdout.splitlines()
        for cid in ids:
            if not re.fullmatch(r"[a-f0-9]{64}", cid):
                raise ValueError("unresolved control container identity")
            container = json.loads(base.pilot.command(["docker", "inspect", cid]).stdout)[0]
            arm = expected.get(container["Name"])
            if (arm is None or container["Id"] != cid or container["Image"] != plan["images"][arm]
                    or container["Config"].get("Labels", {}).get(base.cold.LABEL) != f"{digest}:{arm}"):
                raise ValueError("control container ownership uncertain")
            result["containers"].append({"id": cid, "name": container["Name"], "running": container["State"]["Running"]})
            if container["State"]["Running"]:
                raise RuntimeError("control container remains live")
        result["ok"] = True
    except BaseException as error:
        result["errors"].append(type(error).__name__)
    return result


@contextmanager
def adapter(plan: dict, arm: str):
    names = ("PREFIX", "PORT", "stage_windows", "verify_stage_identities", "clone_argv", "runtime_audit", "model_name", "slot_name")
    prior = {name: getattr(base, name) for name in names}
    try:
        base.PREFIX, base.PORT = PREFIX, PORT
        base.stage_windows = lambda value, stage: value["windows"] if stage == "control" else (_ for _ in ()).throw(ValueError("undeclared stage"))
        base.verify_stage_identities = lambda value: authenticate(Path(value["plan_path"]))
        base.clone_argv = clone_argv
        base.runtime_audit = runtime_audit
        base.model_name = model_name
        base.slot_name = slot_name
        yield
    finally:
        for name, value in prior.items():
            setattr(base, name, value)


def run(path: Path) -> dict:
    plan = authenticate(path)
    output = Path(plan["output"])
    if output.exists() or output.parent != ROOT or shutil.disk_usage(ROOT).free < RAW_BYTES + HEADROOM:
        raise ValueError("fresh output and raw-capture budget plus20GiB headroom required")
    if base.pilot.command(["git", "-C", str(REPO), "status", "--porcelain"]).stdout.strip():
        raise ValueError("clean sealed checkout required")
    for arm, image in plan["images"].items():
        if base.pilot.command(["docker", "image", "inspect", image, "--format", "{{.Id}}"]).stdout.strip() != image:
            raise ValueError(f"missing immutable {arm} capture image")
    recipe, digest = load_recipe(), sha(path)
    with open("/run/lock/klc/model-stack.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(mode=0o700)
        prior = {"backend": base.pilot.active("klc-backend.service", True),
                 "timer": base.pilot.active("klc-model-stack.timer")}
        record = {"schema": "glm53.decode-path-matched-control-execution.v1", "plan_sha256": digest,
                  "started_at": base.pilot.now(), "prior": prior, "exit_code": 1, "stages": [],
                  "opened_roles": ["conditional-fit"], "protected_roles_opened": [],
                  "speed_measurement_valid": False, "mtp": False}
        handlers = {}
        try:
            def interrupted(signum, frame):
                raise RuntimeError(f"control interrupted by signal {signum}")
            for sig in base.INTERRUPT_SIGNALS:
                handlers[sig] = signal.signal(sig, interrupted)
            if prior["timer"]:
                base.pilot.command(["sudo", "-n", "systemctl", "stop", "klc-model-stack.timer"])
            if prior["backend"]:
                base.pilot.command(["systemctl", "--user", "stop", "klc-backend.service"])
            hardware = base.cold.inventory()
            base.pilot.save(output / "hardware-inventory.json", hardware)
            for entry in ORDER:
                arm = entry["arm"]
                arm_plan = {**copy.deepcopy(plan), "capture_image": plan["images"][arm], "plan_path": str(path)}
                with adapter(arm_plan, arm):
                    stage = base.capture_stage(arm_plan, entry, recipe, output / arm, digest, hardware)
                record["stages"].append({"arm": arm, "execution_sha256": sha(output / arm / "execution.json")})
                if stage["exit_code"] != 0:
                    raise RuntimeError(f"{arm} control failed; no automatic retry")
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
        raise RuntimeError("matched decode-path control failed; evidence preserved")
    return record


def bootstrap(values: list[float], seed: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(20000, len(values)))].mean(axis=1)
    return [float(x) for x in np.quantile(draws, (0.025, 0.975))]


def analyze(path: Path, output: Path) -> dict:
    import torch
    torch.set_num_threads(metric.CPU_THREADS)
    plan = authenticate(path)
    root = Path(plan["output"])
    execution = json.loads((root / "execution.json").read_text())
    if execution.get("exit_code") != 0 or execution.get("capture_protocol_complete") is not True:
        raise ValueError("successful two-arm capture required")
    if output.exists() or output != output.resolve():
        raise ValueError("fresh canonical analysis output required")
    output.mkdir(parents=True, mode=0o700)
    records = []
    for arm in ("stock", "exl3"):
        stage = json.loads((root / arm / "execution.json").read_text())
        if stage.get("exit_code") != 0 or [row["id"] for row in stage["windows"]] != list(WINDOW_IDS):
            raise ValueError("control stage is incomplete")
        for window in plan["windows"]:
            wid = window["id"]
            tokens = np.load(window["token_path"], allow_pickle=False)
            student, metadata = protocol.load_capture(root / arm / "captures", wid, tokens)
            teacher_path = Path(plan["teacher_root"]) / window["teacher_path"]
            if sha(teacher_path) != window["teacher_sha256"]:
                raise ValueError("teacher changed")
            teacher = metric._load_teacher(teacher_path, window)
            scores = protocol.score_aligned(teacher, student, tokens)
            with (output / f"{wid}.{arm}.npz").open("xb") as handle:
                np.savez(handle, **{field.name: getattr(scores, field.name) for field in fields(scores)})
            row = {"arm": arm, "window_id": wid, "domain": window["domain"],
                   "mean_kld": float(scores.kld.mean()), "true_decode_mean_kld": float(scores.kld[1:].mean()),
                   "one_token_prefill_kld": float(scores.kld[0]), "raw_sha256": metadata["raw_sha256"],
                   "p8_reference_mean_kld": P8_WINDOW_KLD[wid]}
            row["candidate_minus_p8_kld"] = row["mean_kld"] - row["p8_reference_mean_kld"]
            records.append(row)
            base.pilot.save(output / f"{wid}.{arm}.json", row)
            print(json.dumps(row, sort_keys=True), flush=True)
            del teacher, student, scores
    arms = {}
    for arm in ("stock", "exl3"):
        rows = [row for row in records if row["arm"] == arm]
        means = [row["mean_kld"] for row in rows]
        decode = [row["true_decode_mean_kld"] for row in rows]
        deltas = [row["candidate_minus_p8_kld"] for row in rows]
        mean = float(np.mean(means))
        arms[arm] = {"mean_kld": mean, "ci95_window_percentile": bootstrap(means, 20260905),
                     "true_decode_mean_kld": float(np.mean(decode)), "mean_delta_vs_p8": float(np.mean(deltas)),
                     "relative_improvement_vs_p8": (P8_MEAN - mean) / P8_MEAN,
                     "near_0_11": NEAR[0] <= mean <= NEAR[1]}
    exonerated = all(row["near_0_11"] for row in arms.values())
    serving_bug = exonerated and all(row["relative_improvement_vs_p8"] < 0.20 for row in arms.values())
    result = {"schema": "glm53.decode-path-matched-control-analysis.v1", "status": "complete",
              "plan_sha256": sha(path), "metric": "KL(teacher || student), nats, FP64 CPU, all154880 entries",
              "windows": 4, "causal_rows_per_arm": 4 * protocol.ROWS, "p8_selected4_mean_kld": P8_MEAN,
              "near_interval": list(NEAR), "arms": arms, "codec_exonerated_for_0_11_endpoint": exonerated,
              "shared_production_serving_bug_supported": serving_bug, "records": records,
              "opened_roles": ["conditional-fit"], "protected_roles_opened": [],
              "speed_measurement_valid": False, "mtp": False,
              "claim_boundary": "Targeted four-window serving-path diagnostic; not a replacement for full32 quality inference."}
    result["files"] = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    base.pilot.save(output / "analysis.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("plan"); p.add_argument("--plan", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    r = subs.add_parser("run"); r.add_argument("--plan", type=Path, required=True)
    a = subs.add_parser("analyze"); a.add_argument("--plan", type=Path, required=True); a.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = make_plan(args.plan, args.output) if args.command == "plan" else run(args.plan) if args.command == "run" else analyze(args.plan, args.output)
    print(json.dumps(result, sort_keys=True))
