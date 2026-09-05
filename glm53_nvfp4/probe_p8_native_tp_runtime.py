"""Close the reusable TP4 P8 runtime against decoded layer weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from .shard_index import sha256_file


def qdq_e4m3_k32(values: torch.Tensor) -> torch.Tensor:
    blocks = values.float().reshape(*values.shape[:-1], values.shape[-1] // 32, 32)
    maximum = blocks.abs().amax(-1)
    exponent = torch.ceil(
        torch.log2((maximum / 448.0).clamp(min=2.0**-127))
    ).clamp(-127, 128)
    scale = torch.exp2(exponent)
    scale = torch.where(maximum == 0, torch.zeros_like(scale), scale)
    inverse = torch.where(scale == 0, torch.zeros_like(scale), 1.0 / scale)
    quantized = (blocks * inverse[..., None]).clamp(-448.0, 448.0)
    return (
        quantized.to(torch.float8_e4m3fn).float() * scale[..., None]
    ).reshape_as(values)


def load_dense(
    path: Path, *, layer: int, rank: int, experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gate, up, down = [], [], []
    with safe_open(path, framework="pt", device="cpu") as src:
        for expert in range(experts):
            base = f"model.language_model.layers.{layer}.mlp.experts.{expert}"
            gate.append(src.get_tensor(f"{base}.gate_proj.weight")[rank * 512 : (rank + 1) * 512])
            up.append(src.get_tensor(f"{base}.up_proj.weight")[rank * 512 : (rank + 1) * 512])
            down.append(src.get_tensor(f"{base}.down_proj.weight")[:, rank * 512 : (rank + 1) * 512])
    return (
        torch.stack(gate).cuda().float(),
        torch.stack(up).cuda().float(),
        torch.stack(down).cuda().float(),
    )


def reference(
    x: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
) -> torch.Tensor:
    carrier = qdq_e4m3_k32(x)
    out = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(gate.shape[0]):
        locations = (ids == expert).nonzero(as_tuple=False)
        if not len(locations):
            continue
        token = locations[:, 0]
        slot = locations[:, 1]
        xin = carrier.index_select(0, token)
        g = F.linear(xin, gate[expert]).clamp(max=10.0)
        u = F.linear(xin, up[expert]).clamp(-10.0, 10.0)
        middle = qdq_e4m3_k32(F.silu(g) * u)
        partial = F.linear(middle, down[expert])
        out.index_add_(0, token, partial * weights[token, slot, None])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-patch", type=Path, required=True)
    parser.add_argument("--design", type=Path)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260957)
    parser.add_argument(
        "--mode", choices=("materialized", "monolithic"), default="materialized"
    )
    parser.add_argument("--mac", type=int)
    parser.add_argument("--tokens", type=int, nargs="+", default=(3, 33))
    parser.add_argument("--deterministic-output", action="store_true")
    parser.add_argument("--small-m-scheduler", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    sys.path.insert(0, str(args.runtime_patch))
    from p8_native_kernel import P8NativeTPMoE

    torch.manual_seed(args.seed)
    design_sha256 = sha256_file(args.design) if args.design is not None else None
    runtime_kwargs = {
        "device": torch.device("cuda"),
        "tp_rank": args.rank,
        "layer": args.layer,
        "expected_design_sha256": design_sha256,
        "topk": 8,
        "hidden": 4096,
        "intermediate": 512,
        "force_materialized": (
            None if args.small_m_scheduler else args.mode == "materialized"
        ),
        "mac_override": args.mac,
        "deterministic_output": args.deterministic_output,
    }
    if args.small_m_scheduler:
        runtime_kwargs["small_m_scheduler"] = True
    runtime = P8NativeTPMoE(args.sidecar, **runtime_kwargs)
    gate, up, down = load_dense(
        args.dense, layer=args.layer, rank=args.rank, experts=args.experts
    )
    cells: list[dict[str, object]] = []
    for tokens in args.tokens:
        x = (torch.randn(tokens, 4096, device="cuda") * 0.01).to(torch.bfloat16)
        ids = torch.stack(
            [torch.randperm(args.experts, device="cuda")[:8] for _ in range(tokens)]
        ).to(torch.int32)
        weights = torch.softmax(torch.randn(tokens, 8, device="cuda"), -1).float()
        expected = reference(x.float(), ids, weights, gate, up, down)
        actual_runs = []
        output_hashes = []
        for _ in range(args.repeats):
            actual = runtime(x, weights, ids).float()
            torch.cuda.synchronize()
            actual_runs.append(actual)
            output_hashes.append(
                hashlib.sha256(
                    actual.cpu().contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest()
            )
        actual = actual_runs[0]
        cosine = float(F.cosine_similarity(actual.reshape(1, -1), expected.reshape(1, -1)))
        relative_l2 = float((actual - expected).norm() / expected.norm().clamp_min(1e-9))
        bitwise_deterministic = len(set(output_hashes)) == 1
        cells.append(
            {
                "tokens": tokens,
                "cosine": cosine,
                "relative_l2": relative_l2,
                "finite": bool(torch.isfinite(actual).all()),
                "output_sha256": output_hashes[0],
                "output_sha256_runs": output_hashes,
                "bitwise_deterministic": bitwise_deterministic,
                "pass": bool(
                    torch.isfinite(actual).all()
                    and cosine > 0.995
                    and relative_l2 < 0.12
                    and (args.repeats == 1 or bitwise_deterministic)
                ),
            }
        )
    result = {
        "schema": "glm53-p8-native-tp-runtime-closure.v1",
        "decision": "pass" if all(cell["pass"] for cell in cells) else "fail",
        "decision_rule": "each M3/M33 cell cosine > 0.995 and relative L2 < 0.12",
        "cells": cells,
        "sidecar": {"path": str(args.sidecar), "sha256": sha256_file(args.sidecar)},
        "dense": {"path": str(args.dense), "sha256": sha256_file(args.dense)},
        "rank": args.rank,
        "layer": args.layer,
        "design": (
            {"path": str(args.design), "sha256": design_sha256}
            if args.design is not None
            else None
        ),
        "mode": args.mode,
        "max_active_clusters": args.mac,
        "deterministic_output": args.deterministic_output,
        "small_m_scheduler": args.small_m_scheduler,
        "repeats": args.repeats,
        "experts": list(range(args.experts)),
        "physical_bpw": 4.25,
        "compute": "mxf8f6f4 E4M3 x E4M3 with physical UE8M0/32 scales",
        "isa_cost": "mxf8f6f4 uses twice the MMA issue count of NVFP4",
        "ldlq": False,
        "scope": "reusable TP-local runtime arithmetic closure; not end-to-end KLD or speed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if result["decision"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
