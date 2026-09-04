"""SM120 MoE closure probe for native P8 K3/K4 procedural MCG weights."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import torch

from .shard_index import sha256_file
from .trellis_nvfp4 import procedural_state_values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b12x-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--mode", choices=("small", "split"), default="small")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    # Resolve the runtime B12X package first. The source tree is added only to
    # import its pinned test oracle, so it cannot replace the image's kernel.
    runtime_dynamic = importlib.import_module("b12x.moe._shared.kernels.dynamic")
    sys.path.insert(0, str(args.b12x_source))
    test_module = importlib.import_module("tests.moe.test_dynamic_w4a8_trellis")
    reference = importlib.import_module("tests._reference.trellis_moe")

    def build_mcg(
        generator: torch.Generator,
        experts: int,
        out_features: int,
        in_features: int,
        bits: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k16 = in_features // 16
        n16 = out_features // 16
        windows = experts * k16 * n16
        edges = torch.randint(
            0,
            1 << bits,
            (windows, 256),
            generator=generator,
            device="cpu",
        ).to(device)
        payload = reference._pack_native_tiles(edges, bits).reshape(
            experts, k16, n16, 16 * bits
        )
        states = reference._cyclic_states(edges, bits)
        table = (
            procedural_state_values("mcg")
            .float()
            .mul_(2.0)
            .to(torch.float8_e4m3fn)
            .float()
            .to(device)
        )
        values = table[states.reshape(-1)].reshape(windows, 256)
        rows, columns = reference._tile_position_map(device)
        tiles = torch.zeros(
            windows, 16, 16, dtype=torch.float32, device=device
        )
        tiles[:, rows, columns] = values
        weights = (
            tiles.reshape(experts, k16, n16, 16, 16)
            .permute(0, 2, 3, 1, 4)
            .reshape(experts, out_features, in_features)
            .contiguous()
        )
        return payload, weights

    test_module.build_trellis_weight = build_mcg
    cells: list[dict[str, object]] = []
    token_counts = (3,) if args.mode == "small" else (33, 64)
    for bits in (3, 4):
        for tokens in token_counts:
            test_module._BITS = bits
            actual, expected = test_module._run_trellis_dynamic(
                activation="silu" if args.mode == "small" else "situ",
                E=8,
                m=tokens,
                K=512,
                n=256,
                top_k=4,
                seed=args.seed + bits * 100 + tokens,
                tile_m=16 if args.mode == "small" else 64,
                split_materialized=args.mode == "split",
                mac=4 if args.mode == "small" else 64,
            )
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    actual.reshape(1, -1), expected.reshape(1, -1)
                ).item()
            )
            relative_l2 = float(
                ((actual - expected).norm() / expected.norm().clamp_min(1e-9)).item()
            )
            cells.append(
                {
                    "bits": bits,
                    "tokens": tokens,
                    "finite": bool(torch.isfinite(actual).all()),
                    "nonzero": int(torch.count_nonzero(actual).item()),
                    "cosine": cosine,
                    "relative_l2": relative_l2,
                    "output_sha256": hashlib.sha256(
                        actual.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                    ).hexdigest(),
                    "pass": bool(
                        torch.isfinite(actual).all()
                        and cosine > 0.995
                        and relative_l2 < 0.12
                    ),
                }
            )

    dynamic_path = Path(runtime_dynamic.__file__).resolve()
    payload = {
        "schema": "glm53-p8-mcg-native-moe-closure.v2",
        "product": "P8",
        "compute": "mxf8f6f4 m16n8k32 with E4M3 activations and weights",
        "law": "procedural MCG alpha 2.0",
        "ldlq": False,
        "geometry": {
            "mode": args.mode,
            "experts": 8,
            "tokens": list(token_counts),
            "hidden": 512,
            "intermediate": 256,
            "top_k": 4,
            "activation": "silu" if args.mode == "small" else "situ",
            "tile_m": 16 if args.mode == "small" else 64,
            "materialize_intermediate": args.mode == "split",
        },
        "cells": cells,
        "decision": "pass" if all(cell["pass"] for cell in cells) else "fail",
        "runtime_dynamic": {
            "path": str(dynamic_path),
            "sha256": sha256_file(dynamic_path),
        },
        "oracle_source": {
            "path": str(args.b12x_source),
            "test_sha256": sha256_file(
                args.b12x_source / "tests/moe/test_dynamic_w4a8_trellis.py"
            ),
            "reference_sha256": sha256_file(
                args.b12x_source / "tests/_reference/trellis_moe.py"
            ),
        },
        "isa_cost": "mxf8f6f4 uses twice the MMA issue count of NVFP4",
        "scope": (
            "split-prefill device arithmetic closure only; not end-to-end KLD or speed qualification"
            if args.mode == "split"
            else "small-M device arithmetic closure only; not end-to-end KLD or speed qualification"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    if payload["decision"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
