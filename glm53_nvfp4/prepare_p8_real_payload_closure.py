"""Freeze a real layer-3 P8 payload versus decoded pseudoquant closure run."""
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
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--dense-reference", type=Path, required=True)
    parser.add_argument("--b12x-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260953)
    parser.add_argument("--mode", choices=("small", "split"), default="small")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not 1 <= args.experts <= 72:
        raise ValueError("experts must be in [1, 72]")
    repo = Path(__file__).resolve().parents[1]
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"],
        text=True,
    ).strip()
    payload = {
        "schema": "glm53-p8-real-layer3-device-closure-plan.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-run",
        "decision_question": (
            "Does the actual no-LDLQ layer-3 K4 procedural-MCG stream with physical "
            "UE8M0/32 scales execute through the registered identity-boundary "
            "mxf8f6f4 FC1/FC2 path within the registered dense-pseudoquant tolerance?"
        ),
        "decision_rule": {
            "cosine_min_exclusive": 0.995,
            "relative_l2_max_exclusive": 0.12,
            "finite": True,
        },
        "geometry": {
            "layer": 3,
            "experts": list(range(args.experts)),
            "tokens": [33, 64] if args.mode == "split" else [3],
            "hidden": 4096,
            "intermediate": 2048,
            "top_k": min(4, args.experts),
            "activation": "situ" if args.mode == "split" else "silu",
            "boundary": "identity",
            "kernel": "split-prefill" if args.mode == "split" else "monolithic",
        },
        "codec": {
            "path": str(args.codec.resolve()),
            "sha256": sha256_file(args.codec),
        },
        "dense_reference": {
            "path": str(args.dense_reference.resolve()),
            "sha256": sha256_file(args.dense_reference),
        },
        "image": {"tag": args.image, "id": image_id},
        "sources": {
            "probe": sha256_file(repo / "glm53_nvfp4/probe_p8_mcg_moe.py"),
            "runtime_dynamic": sha256_file(
                repo / "runtime_patch/b12x_h16/b12x/moe/_shared/kernels/dynamic.py"
            ),
            "mcg_decode": sha256_file(
                repo / "runtime_patch/p8_mcg/w4a8_mcg_decode.py"
            ),
            "b12x_test_oracle": sha256_file(
                args.b12x_source / "tests/moe/test_dynamic_w4a8_trellis.py"
            ),
            "b12x_reference": sha256_file(
                args.b12x_source / "tests/_reference/trellis_moe.py"
            ),
        },
        "seed": args.seed,
        "encoder": "Viterbi plus full-Hessian GPTQ-style error feedback; no LDLQ",
        "stored_bpw": 4.25,
        "table_bytes": 0,
        "isa_cost": "mxf8f6f4 uses twice the MMA issue count of NVFP4",
        "role": (
            "developmental real-payload device arithmetic closure; no teacher logits "
            "or protected role is consumed"
        ),
        "stopping_rule": "one registered run; preserve failure without reroll",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
