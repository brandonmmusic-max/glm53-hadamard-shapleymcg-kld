#!/usr/bin/env python3
"""Prepare or explicitly build the immutable coupled P8 research image."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Sequence


REPO = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO / "runtime_patch/p8_coupled_image/Dockerfile"
MANIFEST = REPO / "runtime_patch/p8_coupled_image/image_manifest.json"
VERIFY_IN_IMAGE = "/opt/p8-coupled-runtime/verify_image.py"
PARENT = "sha256:0336113e0fff876cccf9e6ac5347528ae59f4ad894a0ce7cb4c4e90b4651a745"
TAG = "klc/glm53-p8-coupled:v1"
INTEGRATION_BASE = "eded4c349f0bec82634ccf0fa0749be833665670"
TAIL_V2_SHA256 = "494192195da43c46d99a684555fc10fd13a19e89288cb9f51da2536ccdf1f251"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output(command: Sequence[str]) -> str:
    return subprocess.check_output(command, text=True).strip()


def image_id(name: str) -> str:
    return output(["docker", "image", "inspect", name, "--format", "{{.Id}}"])


def image_exists(name: str) -> bool:
    return subprocess.run(
        ["docker", "image", "inspect", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def git(*args: str) -> str:
    return output(["git", "-C", str(REPO), *args])


def load_manifest() -> dict[str, object]:
    manifest = json.loads(MANIFEST.read_text())
    if manifest["parent_image_id"] != PARENT:
        raise ValueError("manifest parent image differs from the pinned parent")
    if manifest["tail_v2"]["sha256"] != TAIL_V2_SHA256:
        raise ValueError("manifest tail-V2 identity differs")
    for relative, expected in manifest["source_sha256"].items():
        path = REPO / relative
        if not path.is_file():
            raise ValueError(f"missing image source: {relative}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(
                f"image source hash mismatch: {relative}: expected {expected}, got {actual}"
            )
    return manifest


def build_command(
    *, context: Path, source_commit: str, source_tree: str, tag: str
) -> list[str]:
    return [
        "docker",
        "build",
        "--pull=false",
        "--network=none",
        "--file",
        str(context / "runtime_patch/p8_coupled_image/Dockerfile"),
        "--build-arg",
        f"KLC_SOURCE_COMMIT={source_commit}",
        "--build-arg",
        f"KLC_SOURCE_TREE={source_tree}",
        "--tag",
        tag,
        str(context),
    ]


def stage_context(
    *, output_dir: Path, manifest: dict[str, object]
) -> tuple[Path, dict[str, str]]:
    """Create a minimal context so no model, role, or unrelated repo data is sent."""
    context = output_dir / "context"
    staged: dict[str, str] = {}
    relative_paths = [
        *manifest["source_sha256"],
        "runtime_patch/p8_coupled_image/image_manifest.json",
    ]
    for relative in relative_paths:
        source = REPO / relative
        destination = context / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        staged[relative] = sha256(destination)
    return context, staged


def preflight(*, output_dir: Path, source_commit: str, tag: str) -> dict[str, object]:
    if output_dir != output_dir.resolve() or output_dir.exists():
        raise ValueError("fresh canonical output directory required")
    if tag != TAG:
        raise ValueError(f"exact dedicated tag required: {TAG}")
    if image_exists(tag):
        raise ValueError(f"refusing to overwrite existing image tag: {tag}")
    if image_id(PARENT) != PARENT:
        raise ValueError("immutable tail-V2 P8 parent image is unavailable")

    head = git("rev-parse", "HEAD")
    if source_commit != head:
        raise ValueError(f"--source-commit must equal clean worktree HEAD {head}")
    if git("status", "--porcelain"):
        raise ValueError("refusing a dirty or untracked source worktree")
    ancestor = subprocess.run(
        ["git", "-C", str(REPO), "merge-base", "--is-ancestor", INTEGRATION_BASE, head],
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("required audited coupled runtime integration is not an ancestor")
    source_tree = git("rev-parse", "HEAD^{tree}")
    manifest = load_manifest()
    source_hashes = dict(manifest["source_sha256"])
    source_hashes["runtime_patch/p8_coupled_image/image_manifest.json"] = sha256(MANIFEST)
    source_hashes["scripts/build_p8_coupled_image.py"] = sha256(Path(__file__))

    output_dir.mkdir(parents=True)
    context, staged_hashes = stage_context(output_dir=output_dir, manifest=manifest)
    expected_staged = {
        key: value
        for key, value in source_hashes.items()
        if key != "scripts/build_p8_coupled_image.py"
    }
    if staged_hashes != expected_staged:
        raise RuntimeError("minimal Docker context differs from pinned source hashes")
    return {
        "schema": "glm53.p8-coupled-image-build.v1",
        "status": "prepared",
        "prepared_unix_ns": time.time_ns(),
        "parent_image_id": PARENT,
        "tag": tag,
        "integration_base_commit": INTEGRATION_BASE,
        "source_commit": head,
        "source_tree": source_tree,
        "source_sha256": source_hashes,
        "staged_context_sha256": staged_hashes,
        "context": str(context),
        "tail_v2_sha256": TAIL_V2_SHA256,
        "command": build_command(
            context=context,
            source_commit=head,
            source_tree=source_tree,
            tag=tag,
        ),
        "execute_requested": False,
        "gpu_used": False,
        "device_closure_tested": False,
        "throughput_tested": False,
        "kld_tested": False,
    }


def execute(plan: dict[str, object], output_dir: Path) -> None:
    command = plan["command"]
    plan["execute_requested"] = True
    plan["status"] = "building"
    with (output_dir / "build.log").open("x") as log:
        subprocess.run(
            command,
            env={**os.environ, "DOCKER_BUILDKIT": "0"},
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    built = image_id(str(plan["tag"]))
    if image_id(PARENT) != PARENT:
        raise RuntimeError("parent image identity changed during build")

    verify_text = output(
        [
            "docker",
            "run",
            "--rm",
            "--network=none",
            "--runtime=runc",
            "-e",
            "NVIDIA_VISIBLE_DEVICES=void",
            "--entrypoint",
            "/opt/venv/bin/python",
            built,
            VERIFY_IN_IMAGE,
        ]
    )
    verification = json.loads(verify_text.splitlines()[-1])
    if verification.get("status") != "pass":
        raise RuntimeError("installed image verification did not pass")

    labels = json.loads(
        output(
            [
                "docker",
                "image",
                "inspect",
                built,
                "--format",
                "{{json .Config.Labels}}",
            ]
        )
    )
    required_labels = {
        "org.klc.parent.digest": PARENT,
        "org.klc.tail-v2.sha256": TAIL_V2_SHA256,
        "org.klc.source.commit": plan["source_commit"],
        "org.klc.source.tree": plan["source_tree"],
        "org.klc.qualification": "research-only-not-device-qualified",
    }
    mismatches = {
        key: {"expected": expected, "actual": labels.get(key)}
        for key, expected in required_labels.items()
        if labels.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"image label mismatch: {mismatches}")

    plan.update(
        status="complete",
        completed_unix_ns=time.time_ns(),
        image_id=built,
        verification=verification,
        labels=required_labels,
        build_log_sha256=sha256(output_dir / "build.log"),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-commit",
        required=True,
        help="full clean-worktree HEAD to embed and verify",
    )
    parser.add_argument("--tag", default=TAG)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="perform the build; without this flag only a preflight plan is written",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    plan = preflight(
        output_dir=args.output,
        source_commit=args.source_commit,
        tag=args.tag,
    )
    plan_path = args.output / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    if not args.execute:
        print(json.dumps({"status": "prepared", "plan": str(plan_path)}))
        return

    try:
        execute(plan, args.output)
    except BaseException as error:
        plan.update(
            status="failed",
            completed_unix_ns=time.time_ns(),
            error=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        (args.output / "receipt.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n"
        )
    print(
        json.dumps(
            {
                "status": plan["status"],
                "image_id": plan["image_id"],
                "receipt": str(args.output / "receipt.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
