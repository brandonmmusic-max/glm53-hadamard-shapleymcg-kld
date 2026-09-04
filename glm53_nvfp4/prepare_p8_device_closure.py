"""Freeze the developmental P8 MCG device-closure repeat before execution."""
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
    parser.add_argument("--seed", type=int, default=20260940)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    repo = Path(__file__).resolve().parents[1]
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"],
        text=True,
    ).strip()
    payload = {
        "schema": "glm53-p8-mcg-device-closure-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-repeat",
        "observation": "An unsealed bring-up run passed decoder and small-M arithmetic thresholds; this fresh-seed repeat freezes the same thresholds and writes immutable receipts.",
        "decision_question": "Does procedural MCG alpha-2 decode exactly to the encoder's E4M3 bytes, and does the K3/K4 stream execute through the SM120 mxf8f6f4 MoE path within the predeclared arithmetic tolerance?",
        "products": ["P8-K3", "P8-K4"],
        "thresholds": {
            "decode_mismatched_words": 0,
            "moe_cosine_min_exclusive": 0.995,
            "moe_relative_l2_max_exclusive": 0.12,
            "finite": True,
        },
        "seed": args.seed,
        "image": {"tag": args.image, "id": image_id},
        "sources": {
            "runtime_dynamic": sha256_file(
                repo / "runtime_patch/b12x_h16/b12x/moe/_shared/kernels/dynamic.py"
            ),
            "decode_probe": sha256_file(repo / "glm53_nvfp4/probe_p8_mcg_decode.py"),
            "moe_probe": sha256_file(repo / "glm53_nvfp4/probe_p8_mcg_moe.py"),
            "b12x_test_oracle": sha256_file(
                args.b12x_source / "tests/moe/test_dynamic_w4a8_trellis.py"
            ),
            "b12x_reference": sha256_file(
                args.b12x_source / "tests/_reference/trellis_moe.py"
            ),
        },
        "encoder_contract": "Viterbi plus full-Hessian GPTQ-style feedback; no LDLQ; direct ties-to-even E4M3 rounding",
        "table_bytes": 0,
        "isa_cost": "mxf8f6f4 uses twice the MMA issue count of NVFP4; P8 is quality-oriented",
        "role": "developmental device closure; no teacher logits or protected roles",
        "stopping_rule": "one decode run and one MoE run at the frozen seed; preserve failures without reroll",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
