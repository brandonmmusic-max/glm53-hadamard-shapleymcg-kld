"""Sealed Tail-V2 CF32 quality/determinism and separate product-speed runner.

No work occurs on import.  ``quality`` and ``speed`` are intentionally
separate commands and independently restore the pre-run production state.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import queue
import re
import shlex
import shutil
import signal
import socket
import statistics
import threading
import time
from urllib.request import Request, urlopen

import numpy as np

from . import decode_path_control as exl3_control
from . import p8_decode_launcher as capture
from . import p8_fc1_cold_compare as cold
from . import p8_tail_repair as p8_recipe
from . import p8_weight_identity as weights
from . import tail_v2_product_validation as validation
from .paired_role_analysis import BOOTSTRAP_B, BOOTSTRAP_SEED, bca_mean_interval


REPO = Path(__file__).resolve().parents[1]
ROOT = Path("/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6")
QUALITY_PREFIX = "glm53-tail-v2-cf32-product"
SPEED_PREFIX = "glm53-tail-v2-product-speed"
QUALITY_PORT = 8030
SPEED_PORT = 8031
P8_RUNTIME_MOUNT = "/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-smallm-v1/runtime_patch:/runtime-patch:ro"
SOURCE_IMAGES = dict(cold.IMAGES)
SPEED_ORDER = [{"round": row["repeat"], "arm": row["arm"]} for row in validation.ORDER]


def model_name(prefix: str, arm: str) -> str:
    return f"{prefix}-{arm}"


def slot_name(entry: dict) -> str:
    number = entry.get("repeat", entry.get("round"))
    return f"round-{number:02d}-{entry['arm']}"


def source_recipes() -> dict:
    recipes = {"p8": p8_recipe.load_recipe(), "exl3": exl3_control.load_recipe()}
    for arm, recipe in recipes.items():
        _validate_source_recipe(recipe, arm)
    return recipes


def _validate_source_recipe(container: dict, arm: str) -> None:
    host, config = container["HostConfig"], container["Config"]
    if (container["Image"] != SOURCE_IMAGES[arm] or config["Entrypoint"] != ["/bin/bash"]
            or config["WorkingDir"] != "/" or config["Cmd"][0] != "-lc"
            or host["NetworkMode"] != "host" or host["IpcMode"] != "host"
            or host["Privileged"] or host["SecurityOpt"] != ["label=disable"]):
        raise ValueError("source product recipe identity/host regime differs")
    tokens = shlex.split(config["Cmd"][1])
    required = {
        "--tensor-parallel-size": "4",
        "--decode-context-parallel-size": str(validation.TOPOLOGY[arm]["dcp"]),
        "--attention-backend": "B12X_MLA_SPARSE",
        "--kv-cache-dtype": "nvfp4_ds_mla",
        "--max-num-seqs": "1",
        "--quantization": "modelopt" if arm == "p8" else "exl3",
    }
    for option, expected in required.items():
        if tokens.count(option) != 1 or tokens[tokens.index(option) + 1] != expected:
            raise ValueError(f"source product recipe option differs: {option}")
    if (("--enable-expert-parallel" in tokens) != validation.TOPOLOGY[arm]["ep"]
            or any("speculat" in token or token == "--enforce-eager" for token in tokens)):
        raise ValueError("source recipe EP/MTP/graph regime differs")


def _replace(tokens: list[str], option: str, value: str) -> None:
    if tokens.count(option) != 1:
        raise ValueError(f"missing or duplicate serving option: {option}")
    tokens[tokens.index(option) + 1] = value


def clone_argv(
    plan: dict, recipe: dict, arm: str, entry: dict, out: Path,
    owner: str, *, capture_enabled: bool, prefix: str, port: int,
) -> list[str]:
    _validate_source_recipe(recipe, arm)
    tokens = shlex.split(recipe["Config"]["Cmd"][1])
    _replace(tokens, "--port", str(port))
    _replace(tokens, "--served-model-name", model_name(prefix, arm))
    if "--logits-processors" in tokens:
        raise ValueError("V1/custom logits processor is forbidden")
    overrides = {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "GLM53_P8_DECODE_CAPTURE_V2": "1" if capture_enabled else "",
        "GLM53_P8_DECODE_CAPTURE_ROOT": "/p8-captures" if capture_enabled else "",
        "GLM53_P8_DECODE_CAPTURE_ALLOWED_WINDOW_IDS": (
            ",".join(window["id"] for window in plan["windows"]) if capture_enabled else ""
        ),
        "GLM53_P8_DECODE_CAPTURE_EXPECTED_OUTPUT_TOKENS": str(validation.ROWS) if capture_enabled else "",
    }
    if arm == "p8":
        overrides.update({"GLM53_P8_INDEX_ORDER": "logical-short-v1",
                          "GLM53_P8_INDEX_ORDER_RECEIPT": "1", "GLM53_P8_INDEX_TRACE": ""})
    env = [value for value in recipe["Config"]["Env"] if value.split("=", 1)[0] not in overrides]
    env.extend(f"{key}={value}" for key, value in overrides.items())
    name = f"{prefix}-{slot_name(entry)}"
    host = recipe["HostConfig"]
    argv = ["docker", "create", "--name", name, "--cidfile", str(out / "container.cid"),
            "--label", f"{cold.LABEL}={owner}", "--network", "host", "--ipc", "host",
            "--shm-size", str(host["ShmSize"]), "--gpus", "all", "--runtime", host["Runtime"],
            "--restart", "no", "--security-opt", "label=disable", "--workdir", "/",
            "--entrypoint", "/bin/bash"]
    for value in env:
        argv += ["--env", value]
    for bind in host["Binds"]:
        destination = bind.split(":")[1]
        if destination == "/p8-captures":
            raise ValueError("source recipe contains capture mount")
        if arm == "p8" and bind == P8_RUNTIME_MOUNT:
            bind = f"{REPO / 'runtime_patch'}:/runtime-patch:ro"
        argv += ["--volume", bind]
    if capture_enabled:
        argv += ["--volume", f"{out / 'captures'}:/p8-captures:rw"]
    return [*argv, plan["images"][arm], "-lc", shlex.join(tokens)]


def _request(request: dict, port: int, tick, timeout: int) -> dict:
    replies: queue.Queue = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            payload = json.dumps(request).encode()
            query = Request(f"http://127.0.0.1:{port}/v1/completions", payload,
                            {"Content-Type": "application/json"})
            with urlopen(query, timeout=timeout) as response:
                replies.put((True, json.load(response)))
        except BaseException as error:
            replies.put((False, error))

    threading.Thread(target=worker, daemon=True).start()
    deadline = time.monotonic() + timeout
    while True:
        tick()
        try:
            ok, value = replies.get(timeout=min(2, max(0.01, deadline - time.monotonic())))
        except queue.Empty:
            if time.monotonic() >= deadline:
                raise RuntimeError("forced-decode request timeout")
            continue
        if not ok:
            raise value
        return value


def _cool_and_inventory(out: Path, expected: list[str]) -> None:
    with (out / "cooldown.jsonl").open("x") as stream:
        deadline = time.monotonic() + 1800
        while True:
            values = cold.pilot.temperatures()
            stream.write(json.dumps({"at": cold.pilot.now(), "temperatures": values}) + "\n")
            stream.flush()
            if max(values) <= 75:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("startup cooling timeout")
            time.sleep(10)
    if cold.inventory() != expected:
        raise RuntimeError("GPU identity or power limits changed")
    if cold.pilot.command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]).stdout.strip():
        raise RuntimeError("another GPU compute process is active")


def _campaign_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            total += path.stat().st_size
    if total >= validation.MAX_NEW_BYTES:
        raise RuntimeError("campaign reached the hard 30,000,000,000-byte ceiling")
    return total


def run_quality_slot(plan: dict, recipe: dict, entry: dict, root: Path,
                     plan_sha: str, hardware: list[str]) -> dict:
    arm, repeat = entry["arm"], entry["repeat"]
    out = root / slot_name(entry)
    out.mkdir(mode=0o700)
    (out / "captures").mkdir(mode=0o700)
    (out / "requests").mkdir(mode=0o700)
    name = f"{QUALITY_PREFIX}-{slot_name(entry)}"
    owner = f"{plan_sha}:quality:{slot_name(entry)}"
    record = {**entry, "schema": "glm53.tail-v2-product-quality-slot.v1",
              "started_at": cold.pilot.now(), "plan_sha256": plan_sha,
              "image": plan["images"][arm], "exit_code": 1, "windows": [],
              "protected_roles_opened": []}
    cid = None
    try:
        validation.authenticate_plan(Path(plan["plan_path"]))
        if cold.pilot.command(["docker", "ps", "-a", "--filter", f"name=^/{name}$",
                               "--format", "{{.ID}}"]).stdout.strip():
            raise ValueError("quality container name already exists")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", QUALITY_PORT))
        _cool_and_inventory(out, hardware)
        cold.pilot.save(out / "nvidia-before.xml", cold.pilot.command(["nvidia-smi", "-q", "-x"]).stdout)
        argv = clone_argv(plan, recipe, arm, entry, out, owner, capture_enabled=True,
                          prefix=QUALITY_PREFIX, port=QUALITY_PORT)
        cold.pilot.private_save(out / "launch-spec.private.json", {"argv": argv})
        cid = cold.pilot.command(argv).stdout.strip()
        if not re.fullmatch(r"[a-f0-9]{64}", cid):
            raise RuntimeError("invalid quality container ID")
        record["container_id"] = cid
        cold.pilot.command(["docker", "start", cid])
        cold.pilot.private_save(out / "container.private.json", cold.pilot.command(["docker", "inspect", cid]).stdout)
        with (out / "thermal.jsonl").open("x") as thermal:
            def tick() -> None:
                values = cold.pilot.temperatures()
                thermal.write(json.dumps({"at": cold.pilot.now(), "temperatures": values}) + "\n")
                thermal.flush()
                if max(values) >= 90:
                    raise RuntimeError("quality thermal abort at or above 90C")
                state = json.loads(cold.pilot.command(
                    ["docker", "inspect", "-f", "{{json .State}}", cid]).stdout)
                if not state["Running"]:
                    raise RuntimeError("quality server exited")
            deadline = time.monotonic() + 2400
            while time.monotonic() < deadline:
                tick()
                try:
                    with urlopen(f"http://127.0.0.1:{QUALITY_PORT}/v1/models", timeout=5) as response:
                        models = json.load(response)
                    if [row["id"] for row in models["data"]] == [model_name(QUALITY_PREFIX, arm)]:
                        cold.pilot.save(out / "models.json", models)
                        break
                except OSError:
                    pass
                time.sleep(2)
            else:
                raise RuntimeError("quality readiness timeout")
            ready = cold.pilot.command(["docker", "logs", cid])
            ready_log = ready.stdout + ready.stderr
            cold.pilot.private_save(out / "server-ready.private.log", ready_log)
            validation.audit_runtime_log(ready_log, arm, require_dispatch=False)
            runtime_path = out / "runtime-audit.json"
            for index, window in enumerate(plan["windows"]):
                _campaign_bytes(root)
                if validation.storage_inventory(out / "captures")["raw_files"] != 0:
                    raise RuntimeError("prior raw was not retired before next request")
                token_path = Path(window["token_path"])
                tokens = np.load(token_path, allow_pickle=False)
                request = capture.protocol.completion_request(tokens, window["id"], model_name(QUALITY_PREFIX, arm))
                cold.pilot.private_save(out / "requests" / f"{window['id']}.request.json", request)
                response = _request(request, QUALITY_PORT, tick, 900)
                cold.pilot.private_save(out / "requests" / f"{window['id']}.response.json", response)
                response_audit = capture.protocol.verify_response(response, request)
                capture.normalize_capture_ownership(out / "captures", [window], cid,
                                                    plan["images"][arm], name, owner,
                                                    {"uid": os.getuid(), "gid": os.getgid()})
                if index == 0:
                    logs = cold.pilot.command(["docker", "logs", cid])
                    proof_log = logs.stdout + logs.stderr
                    cold.pilot.private_save(out / "runtime-proof.private.log", proof_log)
                    proof = validation.audit_runtime_log(proof_log, arm)
                    cold.pilot.save(runtime_path, proof)
                score = validation.score_retire_window(
                    arm=arm, repeat=repeat, window=window,
                    teacher_root=Path(plan["teacher_root"]), capture_root=out / "captures",
                    receipt_root=root / "scores", runtime_audit_sha256=validation.sha(runtime_path),
                )
                record["windows"].append({"id": window["id"], "domain": window["domain"],
                                          "response_audit": response_audit,
                                          "raw_sha256": score["raw_sha256"],
                                          "score_receipt": str(root / "scores" / f"repeat-{repeat:02d}" / arm / f"{window['id']}.score.json")})
                _campaign_bytes(root)
                tick()
        record["exit_code"] = 0
    except BaseException as error:
        record["error_type"] = type(error).__name__
        cold.pilot.private_save(out / "error.private.txt", str(error))
    finally:
        ignored = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGTERM, signal.SIGHUP)}
        try:
            ok, errors = cold.cleanup_owned(out, name, plan["images"][arm], owner)
            record["cleanup"] = {"ok": ok, "errors": errors}
            if not ok:
                record["exit_code"] = 1
        except BaseException as error:
            record["cleanup"] = {"ok": False, "error_type": type(error).__name__}
            record["exit_code"] = 1
        try:
            cold.pilot.save(out / "nvidia-after.xml", cold.pilot.command(["nvidia-smi", "-q", "-x"]).stdout)
            if cold.inventory() != hardware:
                raise RuntimeError("GPU identity/power changed")
            if record["exit_code"] == 0:
                final_log = (out / "server-final.private.log").read_text(errors="replace")
                final = validation.audit_runtime_log(final_log, arm)
                completed = re.findall(r"GLM53_P8_DECODE_CAPTURE_V2_COMPLETE window=(conditional-fit-\d{4}) rows=2047 tp_rank=0", final_log)
                if completed != [window["id"] for window in plan["windows"]]:
                    raise ValueError("ordered CF32 capture completion inventory differs")
                cold.pilot.save(out / "runtime-final-audit.json", final)
        except BaseException as error:
            record["final_audit_error_type"] = type(error).__name__
            record["exit_code"] = 1
        record["finished_at"] = cold.pilot.now()
        cold.pilot.save(out / "execution.json", record)
        for sig, handler in ignored.items():
            signal.signal(sig, handler)
    return record


def _restoration_safety(plan: dict, prefix: str, plan_sha: str) -> dict:
    result = {"ok": False, "containers": [], "errors": []}
    try:
        ids = cold.pilot.command(["docker", "ps", "-a", "--no-trunc", "--filter",
                                  f"name=^/{prefix}-round-", "--format", "{{.ID}}"]).stdout.splitlines()
        for cid in ids:
            container = json.loads(cold.pilot.command(["docker", "inspect", cid]).stdout)[0]
            match = re.fullmatch(rf"/{re.escape(prefix)}-round-\d{{2}}-(p8|exl3)", container["Name"])
            arm = match.group(1) if match else None
            label = container["Config"].get("Labels", {}).get(cold.LABEL, "")
            if (not re.fullmatch(r"[a-f0-9]{64}", cid) or arm is None or container["Id"] != cid
                    or container["Image"] != plan["images"][arm] or not label.startswith(plan_sha + ":")
                    or container["State"]["Running"]):
                raise ValueError("campaign container identity/state uncertain")
            result["containers"].append({"id": cid, "name": container["Name"], "running": False})
        result["ok"] = True
    except BaseException as error:
        result["errors"].append(type(error).__name__)
    return result


@contextmanager
def _speed_adapter(plan: dict):
    saved = {name: getattr(cold, name) for name in
             ("IMAGES", "TOPOLOGY", "ORDER", "PREFIX", "PORT", "model_name", "slot_name",
              "clone_argv", "runtime_audit")}
    recipes = source_recipes()
    try:
        cold.IMAGES = dict(plan["images"])
        cold.TOPOLOGY = copy.deepcopy(validation.TOPOLOGY)
        cold.ORDER = copy.deepcopy(SPEED_ORDER)
        cold.PREFIX, cold.PORT = SPEED_PREFIX, SPEED_PORT
        cold.model_name = lambda arm: model_name(SPEED_PREFIX, arm)
        cold.slot_name = slot_name
        cold.clone_argv = lambda container, arm, entry, out, owner: clone_argv(
            plan, container, arm, entry, out, owner, capture_enabled=False,
            prefix=SPEED_PREFIX, port=SPEED_PORT)
        cold.runtime_audit = lambda log, arm: validation.audit_runtime_log(log, arm)
        yield recipes
    finally:
        for name, value in saved.items():
            setattr(cold, name, value)


def analyze_quality(plan: dict, root: Path) -> dict:
    determinism = validation.verify_five_run_determinism(root / "scores", plan["windows"])
    records = []
    for window in plan["windows"]:
        for arm in validation.ARMS:
            path = root / "scores/repeat-01" / arm / f"{window['id']}.score.json"
            row = json.loads(path.read_text())
            records.append({"arm": arm, "window_id": window["id"], "domain": window["domain"],
                            "mean_kld": row["mean_kld"], "score_receipt_sha256": validation.sha(path)})
    by_arm = {arm: np.array([row["mean_kld"] for row in records if row["arm"] == arm], dtype=np.float64)
              for arm in validation.ARMS}
    delta = by_arm["p8"] - by_arm["exl3"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(delta), size=(BOOTSTRAP_B, len(delta)))
    bootstrap = delta[indices].mean(axis=1)
    result = {
        "schema": "glm53.tail-v2-product-quality-analysis.v1",
        "status": "pass" if determinism["status"] == "pass" else "fail",
        "determinism": determinism,
        "windows": 32,
        "arms": {arm: {"mean_kld": float(values.mean())} for arm, values in by_arm.items()},
        "p8_minus_exl3_mean_kld": float(delta.mean()),
        "p8_minus_exl3_ci95_bca": list(bca_mean_interval(delta, bootstrap)),
        "p8_minus_exl3_ci95_percentile": [float(x) for x in np.quantile(bootstrap, (0.025, 0.975))],
        "bootstrap": {"replicates": BOOTSTRAP_B, "seed": BOOTSTRAP_SEED, "unit": "window"},
        "records": records,
        "claim_boundary": plan["claim_boundary"],
        "protected_roles_opened": [],
    }
    return result


def _campaign(plan_path: Path, mode: str) -> dict:
    plan = validation.authenticate_plan(plan_path)
    plan = {**plan, "plan_path": str(plan_path)}
    destination = Path(plan["output"]) / mode
    if destination.exists() or destination.parent != Path(plan["output"]):
        raise ValueError("fresh campaign destination required")
    if cold.pilot.command(["git", "-C", str(REPO), "status", "--porcelain"]).stdout.strip():
        raise ValueError("clean sealed checkout required")
    if shutil.disk_usage(ROOT).free < validation.MAX_NEW_BYTES:
        raise ValueError("30GiB storage safety reserve unavailable")
    for image in plan["images"].values():
        if cold.pilot.command(["docker", "image", "inspect", image, "--format", "{{.Id}}"]).stdout.strip() != image:
            raise ValueError("immutable Tail-V2 image unavailable")
    weight_receipt = json.loads(Path(plan["weight_audit"]["path"]).read_text())
    weights.verify_stats(weight_receipt)
    prefix = QUALITY_PREFIX if mode == "quality" else SPEED_PREFIX
    with open("/run/lock/klc/model-stack.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        destination.mkdir(parents=True, mode=0o700)
        prior = {"backend": cold.pilot.active("klc-backend.service", True),
                 "timer": cold.pilot.active("klc-model-stack.timer")}
        record = {"schema": f"glm53.tail-v2-product-{mode}-execution.v1",
                  "plan_sha256": validation.sha(plan_path), "started_at": cold.pilot.now(),
                  "prior": prior, "exit_code": 1, "completed_slots": 0, "runs": [],
                  "protected_roles_opened": []}
        old_handlers = {}
        try:
            def interrupted(signum, frame):
                raise RuntimeError(f"{mode} interrupted by signal {signum}")
            for sig in (signal.SIGTERM, signal.SIGHUP):
                old_handlers[sig] = signal.signal(sig, interrupted)
            if prior["timer"]:
                cold.pilot.command(["sudo", "-n", "systemctl", "stop", "klc-model-stack.timer"])
            if prior["backend"]:
                cold.pilot.command(["systemctl", "--user", "stop", "klc-backend.service"])
            hardware = cold.inventory()
            cold.pilot.save(destination / "hardware-inventory.json", hardware)
            recipes = source_recipes()
            if mode == "quality":
                for entry in validation.ORDER:
                    weights.verify_stats(weight_receipt)
                    result = run_quality_slot(plan, recipes[entry["arm"]], entry, destination,
                                              validation.sha(plan_path), hardware)
                    record["runs"].append({"slot": slot_name(entry),
                                           "execution_sha256": validation.sha(destination / slot_name(entry) / "execution.json")})
                    if result["exit_code"] != 0:
                        raise RuntimeError("quality slot failed; no retry/advance")
                    record["completed_slots"] += 1
            else:
                with _speed_adapter(plan) as recipes:
                    for entry in SPEED_ORDER:
                        weights.verify_stats(weight_receipt)
                        result = cold.run_slot(entry, recipes[entry["arm"]], destination / slot_name(entry),
                                               validation.sha(plan_path), hardware)
                        record["runs"].append({"slot": slot_name(entry),
                                               "execution_sha256": validation.sha(destination / slot_name(entry) / "execution.json")})
                        if result["exit_code"] != 0:
                            raise RuntimeError("speed slot failed; no retry/advance")
                        record["completed_slots"] += 1
            record["exit_code"] = 0
        except BaseException as error:
            record["error_type"] = type(error).__name__
            cold.pilot.private_save(destination / "error.private.txt", str(error))
        finally:
            for sig in old_handlers:
                signal.signal(sig, signal.SIG_IGN)
            safety = _restoration_safety(plan, prefix, validation.sha(plan_path))
            record["restoration_safety"] = safety
            record["restoration"] = cold.restore_if_safe(prior, safety)
            if record["restoration"].get("errors") or any(record["restoration"].get(k) != v for k, v in prior.items()):
                record["exit_code"] = 1
            record["finished_at"] = cold.pilot.now()
            cold.pilot.save(destination / "execution.json", record)
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
    if record["exit_code"]:
        raise RuntimeError(f"{mode} campaign failed; partial evidence preserved")
    if mode == "quality":
        analysis = analyze_quality(plan, destination)
    else:
        rows = []
        for entry in SPEED_ORDER:
            slot = destination / slot_name(entry)
            summary = json.loads((slot / "summary.json").read_text())
            rows.append({**entry, **summary})
        medians = {arm: {key: statistics.median(row[key] for row in rows if row["arm"] == arm)
                         for key in ("decode_tokens_per_second", "prefill_server_tokens_per_second")}
                   for arm in validation.ARMS}
        wins = {key: medians["p8"][key] > medians["exl3"][key] for key in medians["p8"]}
        analysis = {"schema": "glm53.tail-v2-product-speed-analysis.v1", "rows": rows,
                    "medians": medians, "primary_wins": wins, "speed_gate_pass": all(wins.values()),
                    "cold_runs_per_arm": 5, "capture_enabled": False,
                    "claim_boundary": plan["claim_boundary"], "protected_roles_opened": []}
    cold.pilot.save(destination / "analysis.json", analysis)
    return analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("quality", "speed", "analyze-quality"))
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "analyze-quality":
        plan = validation.authenticate_plan(args.plan)
        result = analyze_quality(plan, Path(plan["output"]) / "quality")
    else:
        result = _campaign(args.plan, args.command)
    print(json.dumps(result, sort_keys=True))
