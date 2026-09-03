"""Capture immutable local state before downloads or GPU sessions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def run(*command: str) -> dict:
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return {"command": list(command), "exit_code": completed.returncode, "output": completed.stdout.strip()}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    args = parser.parse_args()
    store = Path("/media/brandonmusic/klcstore")
    carrier = Path("/home/brandonmusic/models/GLM-5.3-Flash-NVFP4")
    usage = shutil.disk_usage(store)
    paths = {
        "roles": args.roles,
        "carrier_config": carrier / "config.json",
        "carrier_index": carrier / "model.safetensors.index.json",
        "calibration_jsonl": Path("/home/brandonmusic/klc-linux/reap_recall_build/work/calibration/reap_recall_calib.jsonl"),
        "tokenizer_json": carrier / "tokenizer.json",
        "plan_v2": Path(__file__).resolve().parents[1] / "PLAN_GLM53_FLASH_NVFP4_V2.md",
        "prompt_v2": Path(__file__).resolve().parents[1] / "SOL_PROMPT_GLM53_NVFP4_V2.md",
        "authorization": Path(__file__).resolve().parents[1] / "AUTHORIZATION.md",
    }
    record = {
        "schema": "glm53-nvfp4-v2.preflight.v1",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "uid": os.getuid(),
        "storage": {"path": str(store), "total": usage.total, "used": usage.used, "free": usage.free},
        "files": {name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)} for name, path in paths.items()},
        "commands": {
            "gpu": run("nvidia-smi", "--query-gpu=index,name,uuid,memory.total,compute_cap", "--format=csv,noheader"),
            "service": run("systemctl", "--user", "is-active", "glm53-r10-tp2-mtp3.service"),
            "backend_service": run("systemctl", "--user", "is-active", "klc-backend.service"),
            "model_stack_timer": run("systemctl", "is-active", "klc-model-stack.timer"),
            "production_container": run("docker", "inspect", "-f", "{{.State.Running}}", "glm53-flash-exl3-k4-tp4-vision-mtp3"),
            "port_8000": run("ss", "-ltn", "sport = :8000"),
            "docker_image": run("docker", "image", "inspect", "klc/glm53-flash-nvfp4:r19-sm120-tp4-ep4-dcp4-v79-dflash2-packed-aux-candidate", "--format", "{{.Id}}"),
            "hf_version": run("hf", "version"),
        },
        "pinned": {
            "bf16_repo": "zai-org/GLM-5.3-Flash-BF16",
            "bf16_revision": "a6c167b62691b2bac901344b65cb651a70f53e43",
            "bf16_repository_bytes": 642672288322,
            "teacher_repo": "brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits",
            "teacher_revision": "95f4fdd94bf29989db2e0d1054e4931f55edb6aa",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "free_bytes": usage.free}, sort_keys=True))


if __name__ == "__main__":
    main()
