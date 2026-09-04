"""Freeze the P8 physical-kernel versus decoded-pseudoquant KLD closure."""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--sidecar-receipt", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    repo = Path(__file__).resolve().parents[1]
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"], text=True
    ).strip()
    payload = {
        "schema": "glm53-p8-kernel-pseudoquant-kld-closure-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-run",
        "role": "opened conditional-fit n=32, developmental closure only",
        "run_ids": {
            "pseudoquant": "p8-identity-mcg-l3-pseudoquant-cf32-v1",
            "kernel": "p8-identity-mcg-l3-native-kernel-cf32-v1",
        },
        "run_order": ["pseudoquant", "kernel"],
        "estimand": "paired equal-window mean KLD(kernel) minus KLD(decoded pseudoquant)",
        "decision_rule": (
            "engineering closure passes when absolute mean delta is at most 0.0014 nats; "
            "paired BCa CI, domain deltas, window wins, and detectable-effect limit remain "
            "reported nonblocking development controls under decision 12"
        ),
        "strict_quality_claim_gate": (
            "unchanged: no claim of beating NVFP4 unless a matched teacher-KLD BCa upper "
            "bound is below zero on an eligible role"
        ),
        "bootstrap": {"method": "paired BCa", "unit": "window", "replicates": 50000, "seed": 20260958},
        "codec": {
            "bits": 4,
            "physical_bpw": 4.25,
            "alphabet": "E4M3",
            "scale": "physical UE8M0/32",
            "law": "procedural MCG alpha 2.0",
            "boundary": "identity",
            "encoder": "Viterbi plus full-Hessian GPTQ-style error feedback; no LDLQ",
        },
        "runtime": {
            "compute": "mxf8f6f4 m16n8k32",
            "image": {"tag": args.image, "id": image_id},
            "isa_cost": "mxf8f6f4 uses twice the MMA issue count of NVFP4",
            "tp": 4,
            "ep": 1,
            "dcp": 1,
            "cuda_graph": False,
            "current_kernel_scope": "materialized dynamic arm for both prefill and decode correctness",
        },
        "inputs": {
            "overlay_receipt": {
                "path": str((args.overlay / "BF16_LAYER_RECEIPT.json").resolve()),
                "sha256": sha256_file(args.overlay / "BF16_LAYER_RECEIPT.json"),
            },
            "sidecar_receipt": {
                "path": str(args.sidecar_receipt.resolve()),
                "sha256": sha256_file(args.sidecar_receipt),
            },
            "roles": {"path": str(args.roles.resolve()), "sha256": sha256_file(args.roles)},
            "sitecustomize": sha256_file(repo / "runtime_patch/sitecustomize.py"),
            "native_runtime": sha256_file(repo / "runtime_patch/p8_native_kernel.py"),
            "launcher": sha256_file(repo / "scripts/run_kld_v3.sh"),
        },
        "protected_boundary": "selection, confirmation, final, and all 28 confirmation logits remain unopened",
        "stopping_rule": "one complete run per arm; preserve all windows with no exclusions or rerolls",
        "ldlq": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
