"""One sealed TP4 FC1 integration diagnostic; importing never launches work."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

from .analyze_p8_fc1_tiles import analyze as analyze_device
from .audit_speed_graphs import verify as verify_graphs
from .p8_smallm_profile import (
    SOURCE, SOURCE_SHA, build_argv as validate_source_recipe,
    command, active, temperatures, now, save, sha,
)

REPO = Path(__file__).resolve().parents[1]
ROOT = SOURCE.parent
NAME = "glm53-p8-fc1-integrated-v1"
PORT = 8021
BENCH = Path("/home/brandonmusic/KLC_SANDBOXES/glm53-exl3-k4-r10-rebase/tooling/llm-inference-bench/llm_decode_bench.py")
OWNER_LABEL = "org.klc.p8-fc1-integration-plan"
REQUIRED_SOURCES = {
    "runtime_patch/sitecustomize.py", "runtime_patch/p8_native_kernel.py",
    "runtime_patch/p8_smallm_schedule.py", "glm53_nvfp4/p8_fc1_integration.py",
    "glm53_nvfp4/p8_smallm_profile.py", "glm53_nvfp4/p8_profile_launch.py",
    "glm53_nvfp4/audit_speed_graphs.py", "glm53_nvfp4/analyze_p8_fc1_tiles.py",
    "glm53_nvfp4/probe_p8_fc1_tiles.py",
}


def private_save(path, value):
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(value if isinstance(value, str)
                     else json.dumps(value, indent=2, sort_keys=True) + "\n")


def benchmark_argv(out):
    return [sys.executable, str(BENCH), "--host", "127.0.0.1", "--port", str(PORT),
            "--model", NAME, "--concurrency", "1", "--contexts", "32k",
            "--duration", "20", "--max-tokens", "4096", "--decode-warmup-seconds", "3",
            "--standalone-prefill", "--prefill-contexts", "32k",
            "--prefill-duration", "20", "--prefill-metric", "auto",
            "--token-targeting", "estimate", "--display-mode", "plain",
            "--no-resume", "--output", str(out / "result.json")]


def clone_argv(container, out, image, tile, fused, owner):
    # This pure validator authenticates the old executable, options, entrypoint
    # and host regime. Its Nsight argv is deliberately never executed.
    validate_source_recipe(container, out)
    if container["Config"]["WorkingDir"] != "/":
        raise ValueError("source working directory differs")
    if tile not in (64, 32) or not isinstance(fused, bool):
        raise ValueError("invalid integration candidate")
    tokens = shlex.split(container["Config"]["Cmd"][1])
    tokens[tokens.index("--port") + 1] = str(PORT)
    tokens[tokens.index("--served-model-name") + 1] = NAME
    env = list(container["Config"]["Env"])
    overrides = {
        "GLM53_P8_FC1_TILE_N": str(tile),
        "GLM53_P8_FUSED_SCRATCH": "1" if fused else "",
    }
    keys = [value.split("=", 1)[0] for value in env]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate source environment key")
    env = [value for value in env if value.split("=", 1)[0] not in overrides]
    env.extend(f"{key}={value}" for key, value in overrides.items())
    host = container["HostConfig"]
    argv = [
        "docker", "create", "--name", NAME, "--cidfile", str(out / "container.cid"),
        "--label", f"{OWNER_LABEL}={owner}", "--network", "host", "--ipc", "host",
        "--shm-size", str(host["ShmSize"]), "--gpus", "all",
        "--runtime", host["Runtime"], "--restart", "no",
        "--security-opt", "label=disable", "--workdir", "/", "--entrypoint", "/bin/bash",
    ]
    for value in env:
        argv.extend(["--env", value])
    for value in host["Binds"]:
        argv.extend(["--volume", value])
    return [*argv, image, "-lc", shlex.join(tokens)]


def authenticate(plan_path):
    plan = json.loads(plan_path.read_text())
    if sha(plan_path) != plan_path.with_suffix(".sha256").read_text().split()[0]:
        raise ValueError("integration seal differs")
    fixed = {
        "schema": "glm53-p8-fc1-integration-plan.v1", "runs": 1,
        "port": PORT, "thermal_start_max_c": 75, "thermal_abort_c": 90,
        "source_container_sha256": SOURCE_SHA,
        "five_cold_run_product_comparison": False, "allocation_restart": False,
    }
    if any(plan.get(key) != value for key, value in fixed.items()):
        raise ValueError("integration plan differs from fixed diagnostic")
    if plan["tile_n"] not in (64, 32) or not isinstance(plan["fused_scratch_zero"], bool):
        raise ValueError("invalid candidate configuration")
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", plan["candidate_image"]):
        raise ValueError("immutable candidate image required")
    if not REQUIRED_SOURCES <= plan["source_sha256"].keys():
        raise ValueError("required source identities missing")
    for relative, expected in plan["source_sha256"].items():
        path = REPO / relative
        if path.resolve() != path or not path.is_relative_to(REPO):
            raise ValueError("source path must be canonical and inside checkout")
        if sha(path) != expected:
            raise ValueError(f"source identity differs: {relative}")
    for name, expected in (
        ("container.json", SOURCE_SHA),
        ("execution.json", plan["source_execution_sha256"]),
        ("analysis.json", plan["source_analysis_sha256"]),
    ):
        if sha(SOURCE / name) != expected:
            raise ValueError("authenticated integrated source differs")
    old_execution = json.loads((SOURCE / "execution.json").read_text())
    if (old_execution["exit_code"] != 0
            or old_execution["files"]["container.json"]["sha256"] != SOURCE_SHA
            or json.loads((SOURCE / "analysis.json").read_text())["integration_status"] != "pass"):
        raise ValueError("prior integration is not qualified")
    if sha(BENCH) != plan["benchmark_sha256"]:
        raise ValueError("benchmark identity differs")
    device_root, device_plan = Path(plan["device_root"]), Path(plan["device_plan"])
    if (not device_root.is_absolute() or device_root != device_root.resolve()
            or not device_plan.is_absolute() or device_plan != device_plan.resolve()):
        raise ValueError("canonical device prerequisite paths required")
    for path, expected in (
        (device_plan, plan["device_plan_sha256"]),
        (device_root / "execution.json", plan["device_execution_sha256"]),
        (device_root / "analysis.json", plan["device_analysis_sha256"]),
    ):
        if sha(path) != expected:
            raise ValueError("device prerequisite changed")
    device_declared = json.loads(device_plan.read_text())
    if (device_declared["physical_gpu_order"] != [0, 1, 2, 3]
            or device_declared["minimum_reduction"] != .35
            or device_declared["candidate_image"] != plan["candidate_image"]
            or device_declared.get("fused_scratch_candidates", False) != plan["fused_scratch_zero"]):
        raise ValueError("device prerequisite regime differs")
    for relative in ("runtime_patch/p8_native_kernel.py", "runtime_patch/p8_smallm_schedule.py"):
        if device_declared["source_sha256"][relative] != plan["source_sha256"][relative]:
            raise ValueError("runtime differs from the tested device candidate")
    if set(plan["device_result_sha256"]) != {"0", "1", "2", "3"}:
        raise ValueError("exactly four device results required")
    for gpu, expected in plan["device_result_sha256"].items():
        if sha(device_root / f"gpu{gpu}" / "result.json") != expected:
            raise ValueError("device result identity differs")
    analysis = analyze_device(device_root, device_plan)
    if analysis != json.loads((device_root / "analysis.json").read_text()):
        raise ValueError("device analysis does not replay")
    if (analysis["decision"] != "advance_to_integrated_tp4"
            or str(plan["tile_n"]) not in analysis["eligible_tiles"]
            or {row["physical_gpu"] for row in analysis["devices"]} != {0, 1, 2, 3}):
        raise ValueError("candidate did not pass the all-four-device gate")
    return plan, analysis


def verify_runtime(log, tile, fused):
    expected = {(layer, rank) for layer in range(3, 45) for rank in range(4)}
    forward = re.findall(
        r"GLM53_P8_NATIVE_FORWARD layer=(\d+) rank=(\d+)[^\r\n]*small_m_scheduler=true", log)
    m1 = re.findall(
        r"GLM53_P8_M1_DISPATCH layer=(\d+) rank=(\d+) fc1_tile_n=(\d+) fused_scratch_zero=(true|false)", log)
    if len(forward) != 168 or {(int(a), int(b)) for a, b in forward} != expected:
        raise ValueError("native forward receipts differ from exactly 168")
    if (len(m1) != 168 or {(int(a), int(b)) for a, b, _, _ in m1} != expected
            or any(int(n) != tile or enabled != str(fused).lower() for _, _, n, enabled in m1)):
        raise ValueError("actual M1 dispatch receipts differ from candidate")
    return {"forward_pairs": 168, "m1_pairs": 168, "graph": verify_graphs(log)}


def summarize(result):
    metadata = result.get("metadata", {})
    required_metadata = {
        "engine": "vllm", "model": NAME, "server": f"127.0.0.1:{PORT}",
        "concurrency_levels": [1], "context_lengths": [32768],
        "duration_per_test": 20, "max_tokens": 4096, "decode_warmup_seconds": 3,
        "decode_mode": "duration", "primary_decode_layer": "sustained_decode",
        "prefill_mode": "standalone_cold", "standalone_prefill": True,
        "prefill_only": False, "skip_prefill": False, "run_burst": False,
    }
    if any(metadata.get(key) != value for key, value in required_metadata.items()):
        raise ValueError("benchmark protocol metadata differs")
    rows = result.get("results", [])
    prefill = result.get("prefill", {}).get("32768")
    if (len(rows) != 1 or rows[0].get("concurrency") != 1
            or rows[0].get("context_tokens") != 32768
            or set(result.get("prefill", {})) != {"32768"} or not prefill):
        raise ValueError("benchmark did not cover the required 32K regimes")
    row = rows[0]
    if (row.get("aggregate_source") != "openai_continuous_usage"
            or row.get("benchmark_mode") != "duration" or row.get("num_errors") != 0
            or row.get("underfilled") is not False or row.get("effective_concurrency") != 1
            or row.get("warmup_timed_out") is not False or row.get("capacity_limited") is not False
            or row.get("timeout_reason") != ""):
        raise ValueError("decode measurement integrity failed")
    # The fixed historical protocol uses token-targeting=estimate; its actual
    # prompt was 32,316 tokens, not exactly 32,768. Predeclare a +/-5% window.
    prompt_tokens, decode_tokens = prefill.get("prompt_tokens", 0), row.get("input_seq_len_avg", 0)
    samples = prefill.get("samples")
    if (prefill.get("method") != "client" or type(samples) is not int or samples <= 0
            or not all(32768 * .95 <= value <= 32768 * 1.05
                       for value in (prompt_tokens, decode_tokens))):
        raise ValueError("prefill samples or actual 32K prompt regime differs")
    elapsed = row.get("measurement_seconds", 0)
    if not math.isfinite(elapsed) or not 19 <= elapsed <= 21:
        raise ValueError("decode measurement window differs")
    for hardware in (result.get("hardware_run_summary", {}), row.get("hardware_summary", {}),
                     prefill.get("hardware_summary", {})):
        temperature = hardware.get("temp_max_c", math.inf)
        if (hardware.get("gpu_count") != 4 or not hardware.get("samples", 0) > 0
                or not math.isfinite(temperature) or temperature >= 90):
            raise ValueError("benchmark hardware or temperature gate failed")
    decode, prompt = float(rows[0]["aggregate_tps"]), float(prefill["client_tok_per_sec"])
    if not all(math.isfinite(value) and value > 0 for value in (decode, prompt)):
        raise ValueError("invalid measured throughput")
    return {"decode_c1_tokens_per_second": decode, "prefill_32k_client_tokens_per_second": prompt,
            "single_run_diagnostic": True, "five_cold_run_product_comparison": False,
            "allocation_restart": False}


def cleanup_owned(out, image, owner, run=command):
    errors = []
    # The name is unique and preflighted. A label authenticates ownership even
    # when docker create times out after creating a container but before reply.
    result = run(["docker", "inspect", NAME], check=False)
    if result.returncode:
        # Confirm absence; an observation/daemon error must not count as gone.
        listing = run(["docker", "ps", "-a", "--filter", f"name=^/{NAME}$", "--format", "{{.ID}}"])
        if listing.stdout.strip():
            return False, ["owned container could not be inspected"]
        return True, errors
    container = json.loads(result.stdout)[0]
    cid = container["Id"]
    if (not re.fullmatch("[a-f0-9]{64}", cid) or container["Image"] != image
            or container["Name"] != "/" + NAME
            or container["Config"].get("Labels", {}).get(OWNER_LABEL) != owner):
        return False, ["cleanup target ownership did not authenticate"]
    for action in (["stop", "--time", "30"], ["kill"]):
        if not container["State"]["Running"]:
            break
        try:
            run(["docker", *action, cid], check=False, timeout=45)
        except Exception as error:
            errors.append(type(error).__name__)
        try:
            container = json.loads(run(["docker", "inspect", cid]).stdout)[0]
        except Exception as error:
            # Preserve the last known running state so kill is still attempted.
            errors.append("inspect:" + type(error).__name__)
    for filename, read in (
        ("server-final.log", lambda: run(["docker", "logs", cid], check=False)),
        ("container-final.private.json", lambda: container),
    ):
        try:
            value = read()
            private_save(out / filename, value.stdout + value.stderr
                         if filename.endswith(".log") else value)
        except Exception as error:
            errors.append(filename + ":" + type(error).__name__)
    if container["State"]["Running"]:
        return False, [*errors, "owned container remains live"]
    run(["docker", "rm", cid])
    return not errors, errors


def restore_services(prior, run=command, read_active=active):
    record = {"errors": []}
    for key, unit, user in (
        ("backend", "klc-backend.service", True), ("timer", "klc-model-stack.timer", False),
    ):
        if prior[key]:
            argv = (["systemctl", "--user", "start", unit] if user
                    else ["sudo", "-n", "systemctl", "start", unit])
            try:
                run(argv)
            except Exception as error:
                record["errors"].append({"unit": unit, "operation": "start",
                                         "error_type": type(error).__name__})
        try:
            record[key] = read_active(unit, user)
        except Exception as error:
            record[key] = None
            record["errors"].append({"unit": unit, "operation": "verify",
                                     "error_type": type(error).__name__})
    return record


def run(plan_path, out):
    plan, device_analysis = authenticate(plan_path)
    if out.parent != ROOT or out != out.resolve() or out.exists():
        raise ValueError("fresh canonical output directly under campaign root required")
    if command(["git", "-C", str(REPO), "status", "--porcelain"]).stdout.strip():
        raise ValueError("source checkout must be clean")
    image, owner = plan["candidate_image"], sha(plan_path)
    if command(["docker", "image", "inspect", image, "--format", "{{.Id}}"]).stdout.strip() != image:
        raise ValueError("candidate image not installed")
    if shutil.disk_usage(ROOT).free < 5 * 2**30 or max(temperatures()) > 75:
        raise ValueError("disk or startup temperature preflight failed")
    existing = command(["docker", "ps", "-a", "--filter", f"name=^/{NAME}$", "--format", "{{.ID}}"])
    if existing.stdout.strip():
        raise ValueError("integration container name already exists")
    with socket.socket() as port:
        port.bind(("127.0.0.1", PORT))
    with open("/run/lock/klc/model-stack.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out.mkdir(mode=0o700)
        prior = {"backend": active("klc-backend.service", True),
                 "timer": active("klc-model-stack.timer")}
        record = {"schema": "glm53-p8-fc1-integration-execution.v1", "started_at": now(),
                  "plan_sha256": owner, "image": image, "tile_n": plan["tile_n"],
                  "fused_scratch_zero": plan["fused_scratch_zero"], "exit_code": 1,
                  "prior": prior, "protected_roles_opened": [], "allocation_restart": False,
                  "single_run_diagnostic": True, "five_cold_run_product_comparison": False,
                  "source_commit": command(["git", "-C", str(REPO), "rev-parse", "HEAD"]).stdout.strip()}
        process = None
        previous_signals = {}
        try:
            def interrupt(signum, frame):
                raise RuntimeError(f"integration interrupted by signal {signum}")

            for signum in (signal.SIGTERM, signal.SIGHUP):
                previous_signals[signum] = signal.signal(signum, interrupt)
            container = json.loads((SOURCE / "container.json").read_text())[0]
            launch = clone_argv(container, out, image, plan["tile_n"],
                                plan["fused_scratch_zero"], owner)
            private_save(out / "launch-spec.private.json", {"argv": launch, "plan_sha256": owner})
            save(out / "device-prerequisite.json", device_analysis)
            if prior["timer"]:
                command(["sudo", "-n", "systemctl", "stop", "klc-model-stack.timer"])
            if prior["backend"]:
                command(["systemctl", "--user", "stop", "klc-backend.service"])
            if command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]).stdout.strip():
                raise RuntimeError("another GPU compute process is active")
            if max(temperatures()) > 75:
                raise RuntimeError("startup temperature changed after lock")
            save(out / "nvidia-before.xml", command(["nvidia-smi", "-q", "-x"]).stdout)
            cid = command(launch).stdout.strip()
            if not re.fullmatch("[a-f0-9]{64}", cid):
                raise RuntimeError("invalid owned container id")
            command(["docker", "start", cid])
            private_save(out / "container.private.json", command(["docker", "inspect", cid]).stdout)
            with (out / "thermal.jsonl").open("x") as thermal:
                def tick():
                    values = temperatures()
                    thermal.write(json.dumps({"at": now(), "temperatures": values}) + "\n")
                    thermal.flush()
                    if max(values) >= 90:
                        raise RuntimeError("thermal abort at or above 90 C")
                    state = json.loads(command(["docker", "inspect", "-f", "{{json .State}}", cid]).stdout)
                    if not state["Running"]:
                        raise RuntimeError("integration server exited")

                deadline = time.monotonic() + 2400
                while time.monotonic() < deadline:
                    tick()
                    try:
                        with urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5) as response:
                            models = json.load(response)
                        if [entry["id"] for entry in models["data"]] == [NAME]:
                            save(out / "models.json", models)
                            break
                    except OSError:
                        pass
                    time.sleep(2)
                else:
                    raise RuntimeError("server readiness timeout")
                logs = command(["docker", "logs", cid])
                log = logs.stdout + logs.stderr
                private_save(out / "server-ready.log", log)
                save(out / "runtime-audit.json", verify_runtime(
                    log, plan["tile_n"], plan["fused_scratch_zero"]))
                with (out / "benchmark.log").open("x") as benchlog:
                    process = subprocess.Popen(benchmark_argv(out), stdout=benchlog, stderr=subprocess.STDOUT)
                    deadline = time.monotonic() + 900
                    while process.poll() is None:
                        tick()
                        if time.monotonic() >= deadline:
                            raise RuntimeError("benchmark timeout")
                        time.sleep(2)
                    if process.returncode:
                        raise RuntimeError("benchmark process failed")
                tick()
            save(out / "summary.json", summarize(json.loads((out / "result.json").read_text())))
            record["exit_code"] = 0
        except BaseException as error:
            record["error_type"] = type(error).__name__
            # Subprocess exceptions may contain cloned env/launch arguments.
            private_save(out / "error.private.txt", str(error))
        finally:
            # Do not let repeated termination requests interrupt restoration.
            for signum in previous_signals:
                signal.signal(signum, signal.SIG_IGN)
            try:
                if process is not None and process.poll() is None:
                    try:
                        process.terminate()
                        process.wait(timeout=10)
                    except Exception:
                        process.kill()
                        process.wait(timeout=10)
            except BaseException as error:
                record["benchmark_cleanup_error_type"] = type(error).__name__
                record["exit_code"] = 1
            try:
                cleanup_ok, errors = cleanup_owned(out, image, owner)
                record["cleanup"] = {"ok": cleanup_ok, "errors": errors}
            except BaseException as error:
                record["cleanup"] = {"ok": False, "error_type": type(error).__name__}
            # Each service is attempted/verified independently, including after
            # a cleanup, launch, benchmark, or other service restoration error.
            record["restoration"] = restore_services(prior)
            if (not record["cleanup"]["ok"] or record["restoration"]["errors"]
                    or any(record["restoration"][key] != prior[key] for key in prior)):
                record["exit_code"] = 1
            try:
                save(out / "nvidia-after.xml", command(["nvidia-smi", "-q", "-x"]).stdout)
            except Exception as error:
                record["nvidia_after_error_type"] = type(error).__name__
                record["exit_code"] = 1
            record["finished_at"] = now()
            record["files"] = {}
            try:
                for path in out.iterdir():
                    if path.is_file():
                        record["files"][path.name] = {"sha256": sha(path), "bytes": path.stat().st_size}
            except Exception as error:
                record["file_inventory_error_type"] = type(error).__name__
                record["exit_code"] = 1
            save(out / "execution.json", record)
            for signum, handler in previous_signals.items():
                signal.signal(signum, handler)
    if record["exit_code"]:
        raise RuntimeError("integration diagnostic failed; inspect preserved receipt")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.plan, args.output), sort_keys=True))
