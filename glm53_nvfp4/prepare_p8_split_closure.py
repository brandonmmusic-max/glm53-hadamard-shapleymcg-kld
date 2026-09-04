"""Freeze the P8 procedural-MCG split-prefill device repeat."""
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
    parser.add_argument("--b12x-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260943)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    repo = Path(__file__).resolve().parents[1]
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"], text=True
    ).strip()
    sources = {
        "runtime_dynamic": sha256_file(repo / "runtime_patch/b12x_h16/b12x/moe/_shared/kernels/dynamic.py"),
        "mcg_decode": sha256_file(repo / "runtime_patch/p8_mcg/w4a8_mcg_decode.py"),
        "split_transformer": sha256_file(repo / "runtime_patch/p8_mcg/patch_split_phases.py"),
        "probe": sha256_file(repo / "glm53_nvfp4/probe_p8_mcg_moe.py"),
        "b12x_test_oracle": sha256_file(args.b12x_source / "tests/moe/test_dynamic_w4a8_trellis.py"),
        "b12x_reference": sha256_file(args.b12x_source / "tests/_reference/trellis_moe.py"),
    }
    payload = {
        "schema": "glm53-p8-mcg-split-closure-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-repeat",
        "observation": "One unsealed M33/M64 K3/K4 bring-up passed; this new-seed repeat is the decision-bearing result.",
        "decision_question": "Do table-free procedural MCG K3/K4 weights execute through both split FC1 and FC2 mxf8f6f4 kernels at M33 and M64 within tolerance?",
        "thresholds": {"cosine_min_exclusive": 0.995, "relative_l2_max_exclusive": 0.12, "finite": True},
        "required_cells": [{"bits": bits, "tokens": tokens} for bits in (3, 4) for tokens in (33, 64)],
        "seed": args.seed,
        "image": {"tag": args.image, "id": image_id},
        "sources": sources,
        "encoder_contract": "Viterbi plus full-Hessian GPTQ-style feedback; no LDLQ; native ties-to-even E4M3",
        "table_bytes": 0,
        "isa_cost": "mxf8f6f4 uses twice the MMA issue count of NVFP4; P8 is quality-oriented",
        "role": "developmental split-prefill arithmetic closure; no teacher logits or protected roles",
        "stopping_rule": "one fresh-seed four-cell repeat; preserve any failure without reroll",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
