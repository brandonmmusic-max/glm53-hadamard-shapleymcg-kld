#!/usr/bin/env python3
"""Seal the two-arm full-model CF32 runtime manifest (coupled_full, identity_full).

CPU-only.  Authenticates every input (v10 image receipt, serving recipe, stock
carrier, identity and coupled checkpoint manifests, design files, roles and
teacher bytes), derives both arms' environments and ``docker create`` argv, and
writes the sealed manifest plus its SHA-256.  Nothing is launched here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from glm53_nvfp4 import full_coupled_build_support as support
from glm53_nvfp4 import p8_coupled_three_layer_runtime as carrier_receipts
from glm53_nvfp4 import p8_decode_protocol as protocol
from glm53_nvfp4 import p8_full_coupled_runtime as runtime

REPO = Path(__file__).resolve().parents[1]
PREREGISTRATION = REPO / "experiments/p8-full-coupled-k4-cf32-v1.json"
MANIFEST_SCHEMA = "glm53.p8-full-coupled-cf32-runtime.v1"
RUNTIME_LABELS = {"tp": 4, "dcp": 1, "ep": False, "mtp": False, "graphs": True,
                  "attention_backend": "B12X_MLA_SPARSE", "kv_dtype": "nvfp4_ds_mla",
                  "forced_decode_rows": 2047, "capture_is_speed_valid": False,
                  "sitecustomize": "/usr/lib/python3.12/sitecustomize.py",
                  "pythonpath": runtime.V10_PYTHONPATH}


def build_manifest(args: argparse.Namespace, *, docker_inspect=None) -> dict:
    production = support.production_off()
    preregistration = json.loads(PREREGISTRATION.read_text())
    if (preregistration.get("schema") != "glm53.p8-full-coupled-cf32-preregistration.v1"
            or preregistration.get("arms_in_order") != list(runtime.ARMS)):
        raise ValueError("preregistration differs from the executor protocol")
    requested = tuple(args.arm) if args.arm else ("coupled_full",)
    if requested != tuple(runtime.ARMS[: len(requested)]):
        raise ValueError("arms must be a prefix of the preregistered order: coupled_full, identity_full")
    image = runtime.validate_image_receipt(args.image_build_receipt, docker_inspect=docker_inspect)
    stock = carrier_receipts.validate_stock_carrier_receipt(args.stock_carrier_receipt, args.model_root)
    extras = carrier_receipts.validate_stock_carrier_extras(args.stock_carrier_extras_receipt, args.model_root)
    identity = None
    if "identity_full" in requested:
        if not (args.identity_manifest and args.identity_sidecars and args.identity_design):
            raise ValueError("identity_full requires --identity-manifest, --identity-sidecars and --identity-design")
        identity = runtime.validate_identity_manifest(
            args.identity_manifest, sidecar_dir=args.identity_sidecars, design=args.identity_design)
    coupled = runtime.validate_full_coupled_manifest(
        args.coupled_manifest, sidecar_dir=args.coupled_sidecars, transform=args.transform)
    design_shas = {runtime.sha(path): path for path in args.coupled_design}
    if len(design_shas) != len(args.coupled_design):
        raise ValueError("duplicate coupled design file")
    if set(design_shas) != set(coupled["design_by_layer"].values()):
        raise ValueError("coupled design files do not match the manifest's design hashes exactly")
    coupled_designs = [design_shas[digest] for digest in sorted(design_shas)]
    if runtime.sha(args.roles) != runtime.ROLE_SHA256:
        raise ValueError("CF32 role identity differs")
    windows = protocol.load_role_inputs(args.roles, args.teacher_root, verify_teacher_bytes=True)
    ids = [row["id"] for row in windows]
    recipe = json.loads(args.source_recipe.read_text())
    output_root = args.output.parent / (args.output.stem + "-captures")
    if output_root.exists():
        raise ValueError("fresh capture root required")
    campaign_paths = [path for path in args.campaign_path]
    if any(path != path.resolve() for path in campaign_paths) or len(set(campaign_paths)) != len(campaign_paths):
        raise ValueError("campaign paths must be canonical and unique")
    if args.max_new_bytes <= 0:
        raise ValueError("--max-new-bytes must be positive")
    arms = {}
    for arm in requested:
        arm_out = output_root / arm
        if arm == "coupled_full":
            sidecars, designs, transform = args.coupled_sidecars, coupled_designs, args.transform
            design_by_layer = coupled["design_by_layer"]
            bits_by_layer = coupled["bits_by_layer"]
            bpw = coupled["payload"]["stored_bpw_including_metadata"]
            file_bytes = coupled["payload"]["file_bytes"]
        else:
            sidecars, designs, transform = args.identity_sidecars, [args.identity_design], None
            design_by_layer = identity["design_by_layer"]
            bits_by_layer = {layer: 4 for layer in runtime.LAYERS}
            bpw = 4.25
            file_bytes = sum(row["bytes"] for row in identity["files"])
        env = runtime.arm_environment(arm, ids, design_count=len(designs))
        arms[arm] = {
            "environment": env,
            "capture_root": str(arm_out),
            "sidecar_dir": str(sidecars),
            "designs": [{"path": str(path), "sha256": runtime.sha(path)} for path in designs],
            "design_by_layer": {str(layer): digest for layer, digest in sorted(design_by_layer.items())},
            "bits_by_layer": {str(layer): int(bits) for layer, bits in sorted(bits_by_layer.items())},
            "boundary": runtime.BOUNDARIES[arm],
            "bpw": bpw,
            "sidecar_file_bytes": file_bytes,
            "launch_argv": runtime.launch_argv(recipe, image["image_id"], arm, env, arm_out, args.model_root,
                                               sidecars, designs, transform, kv_dtype=args.kv_cache_dtype,
                                               attention_backend=args.attention_backend),
            "runtime_log_gate": "verify_runtime_log; exactly layers 3..44 x ranks 0..3 with the recorded design per layer; no fallback",
        }
    return {
        "schema": MANIFEST_SCHEMA,
        "status": "sealed-before-execution",
        "execution_authority": False,
        "preregistration": {"path": str(PREREGISTRATION), "sha256": runtime.sha(PREREGISTRATION),
                            "arms_in_order": preregistration["arms_in_order"]},
        "production": production,
        "image": image,
        "source_recipe": {"path": str(args.source_recipe), "sha256": runtime.sha(args.source_recipe)},
        "stock_carrier": {
            "path": str(args.model_root),
            "role": "common authenticated base checkpoint for both P8 overlays; not a stock measurement",
            "config_sha256": runtime.sha(args.model_root / "config.json"),
            "index_sha256": runtime.sha(args.model_root / "model.safetensors.index.json"),
            "receipt": {"path": str(args.stock_carrier_receipt), "sha256": runtime.sha(args.stock_carrier_receipt)},
            "extras_receipt": {"path": str(args.stock_carrier_extras_receipt),
                               "sha256": runtime.sha(args.stock_carrier_extras_receipt)},
            "referenced_shard_count": len(stock["shards"]),
            "resolved_total_bytes": stock["index"]["resolved_total_bytes"],
            "unreferenced_safetensors": [row["name"] for row in extras["files"]],
        },
        "roles": {"path": str(args.roles), "sha256": runtime.ROLE_SHA256, "window_ids": ids,
                  "teacher_root": str(args.teacher_root), "teacher_bytes_verified": True},
        "identity_inputs": ({"manifest": {"path": str(args.identity_manifest), "sha256": identity["manifest_sha256"]},
                             "sidecar_dir": str(args.identity_sidecars),
                             "design": {"path": str(args.identity_design), "sha256": identity["design_sha256"]},
                             "files": identity["files"]} if identity is not None else None),
        "coupled_inputs": {"manifest": {"path": str(args.coupled_manifest), "sha256": coupled["manifest_sha256"]},
                           "sidecar_dir": str(args.coupled_sidecars),
                           "designs": [{"path": str(path), "sha256": digest} for digest, path in sorted(design_shas.items())],
                           "transform": {"path": str(args.transform), "sha256": runtime.TRANSFORM_SHA256},
                           "payload": coupled["payload"], "files": coupled["files"]},
        "storage": {"max_new_bytes": args.max_new_bytes,
                    "campaign_paths": [str(path) for path in campaign_paths],
                    "one_window_raw_bytes": 2047 * protocol.VOCAB_LIMIT * 4},
        "arms_in_order": list(requested),
        "arms": arms,
        "runtime": {**RUNTIME_LABELS, "kv_dtype": args.kv_cache_dtype,
                    "attention_backend": args.attention_backend},
        "lifecycle": "never starts/restores production; raw capture streamed one window at a time by the executor",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "image-build-receipt", "source-recipe", "model-root", "stock-carrier-receipt",
                 "stock-carrier-extras-receipt", "coupled-manifest", "coupled-sidecars", "transform",
                 "roles", "teacher-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("identity-manifest", "identity-sidecars", "identity-design"):
        parser.add_argument("--" + name, type=Path, help="required only when --arm identity_full is requested")
    parser.add_argument("--kv-cache-dtype", choices=list(runtime.KV_DTYPES), default="nvfp4_ds_mla",
                        help="KV-cache dtype the arm serves with; recorded in the manifest, seal and conditions")
    parser.add_argument("--attention-backend", choices=list(runtime.ATTENTION_BACKENDS), default="B12X_MLA_SPARSE",
                        help="attention backend; B12X sparse MLA accepts only the NVFP4 cache at this head geometry")
    parser.add_argument("--arm", action="append", choices=list(runtime.ARMS),
                        help="arms to prepare in preregistered order; default coupled_full only")
    parser.add_argument("--coupled-design", type=Path, action="append", required=True,
                        help="every encoder design whose hash appears in the coupled manifest")
    parser.add_argument("--campaign-path", type=Path, action="append", required=True,
                        help="canonical paths whose apparent bytes are charged against the ceiling")
    parser.add_argument("--max-new-bytes", type=int, required=True)
    args = parser.parse_args()
    paths = [value for value in vars(args).values() if isinstance(value, Path)]
    # Repeatable options are not all paths: --arm appends plain arm names, so only Path
    # members of a list argument are subject to the canonical-path rule.
    paths += [item for value in vars(args).values() if isinstance(value, list)
              for item in value if isinstance(item, Path)]
    if any(value != value.resolve() for value in paths):
        raise ValueError("all preparation paths must be absolute and canonical")
    if args.output.exists() or args.output.with_suffix(".sha256").exists():
        raise ValueError("output and seal must be fresh")
    manifest = build_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    args.output.with_suffix(".sha256").write_text(runtime.sha(args.output) + "  " + args.output.name + "\n")
    print(json.dumps({"manifest": str(args.output), "sha256": runtime.sha(args.output)}))


if __name__ == "__main__":
    main()
