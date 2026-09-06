"""Fail-closed runtime manifest helpers for the full-model (layers 3..44) CF32 arms.

Two arms share one immutable v10 image and one serving recipe:

``coupled_full``
    all 42 routed layers native P8 with the coupled H512/H128 + suh/svh boundary;
    sidecars come from the full coupled build manifest and may carry two encoder
    design hashes (the 42-layer preparation and the pilot V3 design for the reused
    layers 3, 20 and 22).
``identity_full``
    all 42 routed layers native P8 with the identity boundary from the pinned
    uniform checkpoint, re-measured on the same image so the pair is matched.

This module performs no work on import and never touches a GPU, a container,
or production.
"""
from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
from pathlib import Path

from .full_coupled_build_support import MANIFEST_SCHEMA as FULL_COUPLED_MANIFEST_SCHEMA
from .full_coupled_build_support import safetensors_header

LAYERS = tuple(range(3, 45))
RANKS = (0, 1, 2, 3)
ARMS = ("coupled_full", "identity_full")
PAIRS = len(LAYERS) * len(RANKS)
IDENTITY_MANIFEST_SCHEMA = "glm53-p8-uniform-all42-tp4-checkpoint.v1"
IDENTITY_DESIGN_SHA256 = "74637836d2680d3040ef000167a052c68519f89cdbc5fdeba03231c180a4d781"
TRANSFORM_SHA256 = "093d219b18ba32471adcee746442b1481c7ba5659bbea62b94f1a665d4343a12"
ROLE_SHA256 = "b5d7e4524eb98ddfbd230a5d9a44de0dc5dbeb796c03e859898b5838e4463d14"
V10_PYTHONPATH = "/opt/p8-coupled-runtime:/opt/exllamav3:/opt/infernal-invocation/vllm:/opt/infernal-invocation/b12x"
IMAGE_BUILD_SCHEMA = "glm53.p8-full-coupled-image-build.v10"
IMAGE_EXPERIMENT_LABEL = "glm53-p8-full-coupled-h512-h128-suh-svh-v10"
IMAGE_PARENT = "sha256:0336113e0fff876cccf9e6ac5347528ae59f4ad894a0ce7cb4c4e90b4651a745"
BOUNDARIES = {"coupled_full": "coupled-h512-h128-suh-svh-v1", "identity_full": "identity"}
FULL_COUPLED_FLAG = {"coupled_full": "true", "identity_full": "false"}
LAYER_SPEC = ",".join(str(layer) for layer in LAYERS)
SERVED_NAME = "glm53-p8-full-cf32-{arm}"
PORT = 8032


def sha(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def arm_environment(arm: str, window_ids: list[str], *, design_count: int) -> dict[str, str]:
    """Container environment for one arm; every P8 flag is set explicitly."""
    if arm not in ARMS:
        raise ValueError("undeclared arm")
    if design_count < 1:
        raise ValueError("at least one design file is required")
    designs = ":".join(f"/p8-design/design-{index}.json" for index in range(design_count))
    return {
        "PYTHONPATH": V10_PYTHONPATH,
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "GLM53_P8_DECODE_CAPTURE_V2": "1",
        "GLM53_P8_DECODE_CAPTURE_ROOT": "/p8-captures",
        "GLM53_P8_DECODE_CAPTURE_ALLOWED_WINDOW_IDS": ",".join(window_ids),
        "GLM53_P8_DECODE_CAPTURE_EXPECTED_OUTPUT_TOKENS": "2047",
        "GLM53_P8_NATIVE": "1",
        "GLM53_P8_SMALL_M": "1",
        "GLM53_P8_FC1_TILE_N": "128",
        "GLM53_P8_FUSED_SCRATCH": "",
        "GLM53_P8_NATIVE_LAYERS": LAYER_SPEC,
        "GLM53_P8_NATIVE_SIDECAR_DIR": "/p8-sidecars",
        "GLM53_P8_NATIVE_DESIGN": designs,
        "GLM53_P8_NATIVE_TRANSFORM": "/p8-design/transform.json" if arm == "coupled_full" else "",
    }


def verify_runtime_log(text: str, arm: str, *, design_by_layer: dict[int, str]) -> dict:
    """Require every layer/rank pair to report the intended boundary and design."""
    if arm not in ARMS:
        raise ValueError("undeclared arm")
    if set(design_by_layer) != set(LAYERS):
        raise ValueError("design_by_layer must cover exactly layers 3..44")
    expected_boundary = BOUNDARIES[arm]
    expected_full = FULL_COUPLED_FLAG[arm]
    ready = re.findall(
        r"GLM53_P8_NATIVE_WEIGHTS_READY layer=(\d+) rank=(\d+) sidecar=\S+ "
        r"design_sha256=([0-9a-f]{64})[^\n]*?boundary=(\S+) full_coupled=(true|false)",
        text,
    )
    forwards = re.findall(
        r"GLM53_P8_NATIVE_FORWARD layer=(\d+) rank=(\d+)[^\n]*?boundary=(\S+)", text
    )
    if len(ready) != PAIRS:
        raise ValueError(f"weights-ready inventory has {len(ready)} lines, expected {PAIRS}")
    seen = set()
    for layer_text, rank_text, design, boundary, full in ready:
        layer, rank = int(layer_text), int(rank_text)
        if layer not in design_by_layer or rank not in RANKS:
            raise ValueError(f"unexpected weights-ready pair layer={layer} rank={rank}")
        if design != design_by_layer[layer]:
            raise ValueError(f"layer {layer} rank {rank} loaded design {design}, expected {design_by_layer[layer]}")
        if boundary != expected_boundary or full != expected_full:
            raise ValueError(f"layer {layer} rank {rank} boundary/full_coupled differs from {arm}")
        seen.add((layer, rank))
    if seen != {(layer, rank) for layer in LAYERS for rank in RANKS}:
        raise ValueError("weights-ready inventory does not cover every layer/rank exactly once")
    if len(forwards) != PAIRS or {(int(a), int(b), c) for a, b, c in forwards} != {
        (layer, rank, expected_boundary) for layer in LAYERS for rank in RANKS
    }:
        raise ValueError("native forward inventory indicates missing rank/layer or fallback")
    if "GLM53_P8_NATIVE_PATCH_ACTIVE layers=" + LAYER_SPEC + " " not in text:
        raise ValueError("native patch activation line for layers 3..44 is missing")
    return {
        "arm": arm,
        "pairs": PAIRS,
        "boundary": expected_boundary,
        "full_coupled": expected_full == "true",
        "designs": sorted(set(design_by_layer.values())),
    }


def validate_full_coupled_manifest(path: Path, *, sidecar_dir: Path, transform: Path) -> dict:
    """Authenticate the full coupled checkpoint manifest and return its design map."""
    path, sidecar_dir, transform = Path(path), Path(sidecar_dir), Path(transform)
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != FULL_COUPLED_MANIFEST_SCHEMA or manifest.get("layer_range") != [3, 44]:
        raise ValueError("full coupled manifest schema/layer range differs")
    if sha(transform) != TRANSFORM_SHA256 or manifest.get("transform", {}).get("sha256") != TRANSFORM_SHA256:
        raise ValueError("coupled transform identity differs")
    payload = manifest.get("payload", {})
    if payload.get("weight_payload_bpw") != 4.25:
        raise ValueError("full coupled manifest is not exactly 4.25 weight-payload bpw")
    layers = manifest.get("layers", [])
    if [row.get("layer") for row in layers] != list(LAYERS):
        raise ValueError("full coupled manifest does not list layers 3..44 in order")
    design_by_layer: dict[int, str] = {}
    files = []
    for row in layers:
        if row.get("status") != "pass" or [r.get("rank") for r in row.get("ranks", [])] != list(RANKS):
            raise ValueError(f"layer {row.get('layer')} lacks a passing four-rank receipt")
        design_by_layer[int(row["layer"])] = row["source_design_sha256"]
        for rank in row["ranks"]:
            expected = sidecar_dir / f"p8-layer-{row['layer']:03d}-tp4-rank-{rank['rank']}.safetensors"
            if Path(rank["path"]) != expected.resolve() or not expected.is_file():
                raise ValueError(f"sidecar path differs for layer {row['layer']} rank {rank['rank']}")
            if expected.stat().st_size != rank["bytes"]:
                raise ValueError(f"sidecar bytes differ for layer {row['layer']} rank {rank['rank']}")
            metadata, _, _ = safetensors_header(expected)
            if (metadata.get("source_design_sha256") != rank["source_design_sha256"]
                    or metadata.get("boundary") != BOUNDARIES["coupled_full"]
                    or metadata.get("full_coupled") != "true"
                    or metadata.get("encoder_transform_sha256") != TRANSFORM_SHA256):
                raise ValueError(f"sidecar metadata differs for layer {row['layer']} rank {rank['rank']}")
            files.append({"layer": row["layer"], "rank": rank["rank"], "path": str(expected),
                          "bytes": rank["bytes"], "sha256": rank["sha256"]})
    return {"manifest_sha256": sha(path), "design_by_layer": design_by_layer,
            "designs": manifest.get("designs"), "payload": payload, "files": files}


def validate_identity_manifest(path: Path, *, sidecar_dir: Path, design: Path) -> dict:
    """Authenticate the pinned uniform identity checkpoint for the matched control."""
    path, sidecar_dir, design = Path(path), Path(sidecar_dir), Path(design)
    manifest = json.loads(path.read_text())
    if (manifest.get("schema") != IDENTITY_MANIFEST_SCHEMA or manifest.get("layer_range") != [3, 44]
            or manifest.get("codec", {}).get("bits") != 4 or manifest.get("codec", {}).get("ldlq") is not False
            or manifest.get("payload", {}).get("bpw") != 4.25):
        raise ValueError("identity manifest schema/codec/rate differs")
    design_sha = sha(design)
    if design_sha != IDENTITY_DESIGN_SHA256 or manifest.get("design", {}).get("sha256") != design_sha:
        raise ValueError("identity design identity differs")
    layers = manifest.get("layers", [])
    if [row.get("layer") for row in layers] != list(LAYERS):
        raise ValueError("identity manifest does not list layers 3..44 in order")
    files = []
    for row in layers:
        if [r.get("rank") for r in row.get("ranks", [])] != list(RANKS):
            raise ValueError(f"identity layer {row.get('layer')} lacks four ranks")
        for rank in row["ranks"]:
            expected = sidecar_dir / f"p8-layer-{row['layer']:03d}-tp4-rank-{rank['rank']}.safetensors"
            if Path(rank["path"]) != expected.resolve() or not expected.is_file():
                raise ValueError(f"identity sidecar path differs for layer {row['layer']} rank {rank['rank']}")
            if expected.stat().st_size != rank["bytes"] or rank.get("payload_bpw") != 4.25:
                raise ValueError(f"identity sidecar bytes/rate differ for layer {row['layer']} rank {rank['rank']}")
            metadata, _, _ = safetensors_header(expected)
            if metadata.get("source_design_sha256") != design_sha or metadata.get("boundary") != "identity":
                raise ValueError(f"identity sidecar metadata differs for layer {row['layer']} rank {rank['rank']}")
            files.append({"layer": row["layer"], "rank": rank["rank"], "path": str(expected),
                          "bytes": rank["bytes"], "sha256": rank["sha256"]})
    return {"manifest_sha256": sha(path), "design_sha256": design_sha,
            "design_by_layer": {layer: design_sha for layer in LAYERS}, "files": files}


def validate_image_receipt(path: Path, *, docker_inspect=None) -> dict:
    """Authenticate the v10 image build receipt and confirm the image is present."""
    path = Path(path)
    value = json.loads(path.read_text())
    image_id = value.get("image_id", "")
    if (value.get("schema") != IMAGE_BUILD_SCHEMA or value.get("status") != "complete"
            or value.get("parent_image_id") != IMAGE_PARENT
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id)
            or value.get("verification", {}).get("status") != "pass"
            or value.get("labels", {}).get("org.klc.experiment") != IMAGE_EXPERIMENT_LABEL
            or value.get("labels", {}).get("org.klc.parent.digest") != IMAGE_PARENT
            or value.get("gpu_used") is not False):
        raise ValueError("v10 image build receipt differs")
    inspect = docker_inspect or (lambda name: subprocess.run(
        ["docker", "image", "inspect", name, "--format", "{{.Id}}"],
        text=True, capture_output=True, check=True).stdout.strip())
    if inspect(image_id) != image_id:
        raise ValueError("v10 image is not present under its recorded id")
    return {"receipt": {"path": str(path.resolve()), "sha256": sha(path)}, "image_id": image_id,
            "source_commit": value.get("source_commit"), "source_tree": value.get("source_tree"),
            "labels": value.get("labels")}


def launch_argv(recipe: dict, image: str, arm: str, env: dict[str, str], output: Path,
                model_root: Path, sidecar_dir: Path, designs: list[Path],
                transform: Path | None, *, port: int = PORT) -> list[str]:
    """Derive one arm's ``docker create`` argv from the authenticated serving recipe.

    The recipe is the ``docker inspect`` record of the reference P8 server.  Only
    the served model name, port, model mount, sidecar mount, design mounts and the
    explicit P8 environment differ between arms; topology flags are re-checked.
    """
    if arm not in ARMS:
        raise ValueError("undeclared arm")
    config, host = recipe["Config"], recipe["HostConfig"]
    raw_command = config.get("Cmd", [])
    if len(raw_command) != 2 or raw_command[0] != "-lc":
        raise ValueError("source recipe shell command shape differs")
    tokens = shlex.split(raw_command[1])
    if tokens[:1] == ["exec"]:
        tokens = tokens[1:]
    if "exec" in tokens or tokens[:4] != ["/opt/venv/bin/python", "-m", "vllm.entrypoints.cli.main", "serve"]:
        raise ValueError("source recipe serving prefix differs")
    required = {"--tensor-parallel-size": "4", "--decode-context-parallel-size": "1",
                "--attention-backend": "B12X_MLA_SPARSE", "--kv-cache-dtype": "nvfp4_ds_mla",
                "--max-num-seqs": "1", "--quantization": "modelopt"}
    for option, value in required.items():
        if tokens.count(option) != 1 or tokens[tokens.index(option) + 1] != value:
            raise ValueError(f"source recipe topology differs: {option}")
    if "--enable-expert-parallel" in tokens or "--enforce-eager" in tokens or any("speculat" in t for t in tokens):
        raise ValueError("source recipe must be TP4/DCP1/noEP/noMTP/graphs")

    def replace(option: str, value: str) -> None:
        if tokens.count(option) != 1:
            raise ValueError(f"missing or duplicate source serving option: {option}")
        tokens[tokens.index(option) + 1] = value

    replace("--port", str(port))
    replace("--served-model-name", SERVED_NAME.format(arm=arm))
    tokens[tokens.index("serve") + 1] = "/model"
    name = SERVED_NAME.format(arm=arm)
    argv = ["docker", "create", "--name", name, "--network", "host", "--ipc", "host",
            "--shm-size", str(host["ShmSize"]), "--gpus", "all", "--runtime", host["Runtime"],
            "--restart", "no", "--security-opt", "label=disable", "--workdir", "/",
            "--entrypoint", "/bin/bash"]
    overrides = set(env)
    for value in config.get("Env", []):
        if value.split("=", 1)[0] not in overrides:
            argv += ["--env", value]
    for key, value in env.items():
        argv += ["--env", f"{key}={value}"]
    for bind in host.get("Binds", []):
        destination = bind.split(":")[1]
        if destination in {"/runtime-patch", "/model", "/p8-sidecars", "/p8-design", "/p8-captures"} or destination.startswith("/p8-design/"):
            continue
        argv += ["--volume", bind]
    argv += ["--volume", f"{model_root}:/model:ro", "--volume", f"{output / 'captures'}:/p8-captures:rw",
             "--volume", f"{sidecar_dir}:/p8-sidecars:ro"]
    expected_designs = env["GLM53_P8_NATIVE_DESIGN"].split(":")
    if len(designs) != len(expected_designs):
        raise ValueError("design mounts differ from the arm environment")
    for index, design in enumerate(designs):
        argv += ["--volume", f"{design}:/p8-design/design-{index}.json:ro"]
    if arm == "coupled_full":
        if transform is None:
            raise ValueError("coupled_full requires the encoder transform")
        argv += ["--volume", f"{transform}:/p8-design/transform.json:ro"]
    elif transform is not None:
        raise ValueError("identity_full must not mount a transform")
    return [*argv, image, "-lc", "exec " + shlex.join(tokens)]
