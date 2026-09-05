#!/usr/bin/env python3
"""Seal the runtime/capture launch manifest for the fixed three-arm CF32 pilot."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess

from glm53_nvfp4 import p8_coupled_three_layer_runtime as runtime
from glm53_nvfp4 import p8_decode_protocol as protocol


def production_off(run=subprocess.run) -> dict:
    states = {}
    for unit, scope in (("klc-backend.service", "user"), ("klc-model-stack.timer", "system")):
        command = ["systemctl", "--user", "is-active", unit] if scope == "user" else ["systemctl", "is-active", unit]
        result = run(command, text=True, capture_output=True)
        state = result.stdout.strip()
        if state != "inactive":
            raise ValueError(f"{scope} production unit must remain inactive: {unit}={state!r}")
        states[unit] = {"scope": scope, "state": state}
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", 8000))
    return {"units": states, "port_8000": "unbound",
            "restore_policy": "never restore or start production from this campaign"}


def _replace(tokens: list[str], option: str, value: str) -> None:
    if tokens.count(option) != 1:
        raise ValueError(f"missing or duplicate source serving option: {option}")
    tokens[tokens.index(option) + 1] = value


def launch_argv(recipe: dict, image: str, arm: str, env: dict[str, str], output: Path,
                model_root: Path, identity_root: Path, coupled_root: Path,
                identity_design: Path, coupled_design: Path, transform: Path) -> list[str]:
    config, host = recipe["Config"], recipe["HostConfig"]
    tokens = shlex.split(config["Cmd"][1])
    required = {"--tensor-parallel-size": "4", "--decode-context-parallel-size": "1",
                "--attention-backend": "B12X_MLA_SPARSE", "--kv-cache-dtype": "nvfp4_ds_mla",
                "--max-num-seqs": "1", "--quantization": "modelopt"}
    for option, value in required.items():
        if tokens.count(option) != 1 or tokens[tokens.index(option) + 1] != value:
            raise ValueError(f"source recipe topology differs: {option}")
    if "--enable-expert-parallel" in tokens or "--enforce-eager" in tokens or any("speculat" in t for t in tokens):
        raise ValueError("source recipe must be TP4/DCP1/noEP/noMTP/graphs")
    _replace(tokens, "--port", "8032")
    _replace(tokens, "--served-model-name", f"glm53-p8-three-layer-cf32-{arm}")
    tokens[tokens.index("serve") + 1] = "/model"

    name = f"glm53-p8-three-layer-cf32-{arm}"
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
        if destination in {"/runtime-patch", "/model", "/p8-sidecars", "/p8-design", "/p8-captures"}:
            continue
        argv += ["--volume", bind]
    argv += ["--volume", f"{model_root}:/model:ro", "--volume", f"{output / 'captures'}:/p8-captures:rw"]
    prelude = ""
    if arm == "identity_p8":
        argv += ["--volume", f"{identity_root}:/p8-sidecars:ro",
                 "--volume", f"{identity_design}:/p8-design/design.json:ro"]
    elif arm == "coupled_p8":
        argv += ["--volume", f"{coupled_root / 'sidecars'}:/p8-coupled-sidecars:ro",
                 "--volume", f"{coupled_design}:/p8-design/design.json:ro",
                 "--volume", f"{transform}:/p8-design/transform.json:ro"]
        links = [f"ln -s /p8-coupled-sidecars/layer-{layer:03d}/p8-layer-{layer:03d}-tp4-rank-{rank}.safetensors /tmp/p8-three-layer-sidecars/"
                 for layer in runtime.LAYERS for rank in runtime.RANKS]
        prelude = "set -e; mkdir /tmp/p8-three-layer-sidecars; " + "; ".join(links) + "; "
    return [*argv, image, "-lc", prelude + "exec " + shlex.join(tokens)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "capture-image-receipt", "source-recipe", "model-root", "identity-root",
                 "identity-manifest", "identity-design", "coupled-root", "coupled-design", "transform",
                 "roles", "teacher-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--coupled-postwrite", action="append", required=True,
                        help="explicit LAYER=/absolute/postwrite-receipt.json; exactly layers 3,20,22")
    parser.add_argument("--coupled-loader", action="append", required=True,
                        help="explicit LAYER=/absolute/real-loader-result.json; exactly layers 3,20,22")
    args = parser.parse_args()
    paths = {key: value for key, value in vars(args).items() if isinstance(value, Path)}
    if any(value != value.resolve() for value in paths.values()):
        raise ValueError("all preparation paths must be absolute and canonical")
    if args.output.exists() or args.output.with_suffix(".sha256").exists():
        raise ValueError("output and seal must be fresh")
    production = production_off()
    image = runtime.validate_capture_image_receipt(args.capture_image_receipt)
    identity = runtime.validate_identity_sidecars(args.identity_root, args.identity_manifest, args.identity_design)
    def explicit_map(values: list[str], label: str) -> dict[int, Path]:
        result = {}
        for value in values:
            layer_text, separator, path_text = value.partition("=")
            if not separator or not layer_text.isdigit():
                raise ValueError(f"invalid explicit {label} mapping")
            layer, path = int(layer_text), Path(path_text)
            if layer in result or path != path.resolve():
                raise ValueError(f"duplicate/noncanonical {label} mapping")
            result[layer] = path
        if set(result) != set(runtime.LAYERS):
            raise ValueError(f"{label} mapping must name exactly layers 3,20,22")
        return result
    postwrites = explicit_map(args.coupled_postwrite, "postwrite")
    loaders = explicit_map(args.coupled_loader, "loader")
    coupled = runtime.validate_coupled_sidecars(
        args.coupled_root, args.coupled_design, args.transform, postwrites, loaders
    )
    if runtime.sha(args.roles) != runtime.ROLE_SHA256:
        raise ValueError("CF32 role identity differs")
    windows = protocol.load_role_inputs(args.roles, args.teacher_root, verify_teacher_bytes=True)
    recipe = json.loads(args.source_recipe.read_text())
    if (not args.model_root.is_dir()
            or runtime.sha(args.model_root / "config.json") != "676382abd1e90a6c85f0c8f33d45441ecd45fd514fd7b63ce5610e732d8e4996"
            or runtime.sha(args.model_root / "model.safetensors.index.json") != "0d1d9e6b226e76520e182de10d4e7194cc885c5cb1bf885bb90de1916ce312cb"):
        raise ValueError("stock NVFP4 carrier config/index differs")
    output_root = args.output.parent / (args.output.stem + "-captures")
    if output_root.exists():
        raise ValueError("fresh capture root required")
    ids = [row["id"] for row in windows]
    arms = {}
    for arm in ("stock", "identity_p8", "coupled_p8"):
        arm_out = output_root / arm
        env = runtime.arm_environment(arm, ids)
        arms[arm] = {"environment": env, "capture_root": str(arm_out),
            "launch_argv": launch_argv(recipe, image["image_id"], arm, env, arm_out, args.model_root,
                                       args.identity_root, args.coupled_root, args.identity_design,
                                       args.coupled_design, args.transform),
            "runtime_log_gate": "verify_runtime_log; exactly layers 3,20,22 x ranks 0..3; no fallback"}
    manifest = {"schema": "glm53.p8-coupled-three-layer-cf32-runtime.v1",
        "status": "sealed-before-execution", "execution_authority": False,
        "production": production, "image": image,
        "capture_attestation": {"path": str(args.capture_image_receipt),
                                "sha256": runtime.sha(args.capture_image_receipt)},
        "source_recipe": {"path": str(args.source_recipe), "sha256": runtime.sha(args.source_recipe)},
        "stock_carrier": {"path": str(args.model_root), "config_sha256": runtime.sha(args.model_root / "config.json"),
                          "index_sha256": runtime.sha(args.model_root / "model.safetensors.index.json")},
        "roles": {"path": str(args.roles), "sha256": runtime.ROLE_SHA256, "window_ids": ids,
                  "teacher_root": str(args.teacher_root), "teacher_bytes_verified": True},
        "identity_inputs": {"root": str(args.identity_root),
                            "manifest": {"path": str(args.identity_manifest), "sha256": runtime.sha(args.identity_manifest)},
                            "design": {"path": str(args.identity_design), "sha256": runtime.sha(args.identity_design)}},
        "coupled_inputs": {"root": str(args.coupled_root),
                           "design": {"path": str(args.coupled_design), "sha256": runtime.sha(args.coupled_design)},
                           "transform": {"path": str(args.transform), "sha256": runtime.sha(args.transform)}},
        "identity_sidecars": identity, "coupled_sidecars": coupled,
        "coupled_evidence": {str(layer): {
            "postwrite": {"path": str(postwrites[layer]), "sha256": runtime.sha(postwrites[layer])},
            "real_loader": {"path": str(loaders[layer]), "sha256": runtime.sha(loaders[layer])},
        } for layer in runtime.LAYERS},
        "arms_in_order": list(arms), "arms": arms,
        "runtime": {"tp": 4, "dcp": 1, "ep": False, "mtp": False, "graphs": True,
                    "attention_backend": "B12X_MLA_SPARSE", "kv_dtype": "nvfp4_ds_mla",
                    "forced_decode_rows": 2047, "capture_is_speed_valid": False,
                    "sitecustomize": "/usr/lib/python3.12/sitecustomize.py",
                    "pythonpath": runtime.V9_PYTHONPATH},
        "lifecycle": "never calls tail_v2_product_runner._campaign; never starts/restores production; raw capture streamed one window at a time by the future executor"}
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    args.output.with_suffix(".sha256").write_text(runtime.sha(args.output) + "  " + args.output.name + "\n")


if __name__ == "__main__":
    main()
