"""Close the reusable TP4 P8 runtime against decoded layer weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
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


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


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
        "--mode", choices=("materialized", "monolithic"), default="monolithic"
    )
    parser.add_argument("--mac", type=int)
    parser.add_argument("--tokens", type=int, nargs="+", default=(3, 33))
    parser.add_argument("--deterministic-output", action="store_true")
    parser.add_argument("--small-m-scheduler", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timing-warmups", type=int, default=0)
    parser.add_argument("--timing-repeats", type=int, default=0)
    parser.add_argument("--cuda-graph-warmups", type=int, default=0)
    parser.add_argument("--cuda-graph-replays", type=int, default=0)
    parser.add_argument("--cuda-graph-timing-repeats", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.small_m_scheduler and args.mode != "monolithic":
        raise ValueError(
            "small-M targets the monolithic M1 serving contract; "
            "--mode materialized is a different arithmetic baseline"
        )
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
    # Runtime construction may initialize a different number of CUDA-side
    # objects in different scheduler arms.  Isolate payload generation from
    # that global RNG state so an A/B always receives identical tensors.
    input_generator = torch.Generator(device="cuda").manual_seed(args.seed)
    cells: list[dict[str, object]] = []
    for tokens in args.tokens:
        x = (
            torch.randn(
                tokens, 4096, device="cuda", generator=input_generator
            )
            * 0.01
        ).to(torch.bfloat16)
        ids = torch.stack(
            [
                torch.randperm(
                    args.experts, device="cuda", generator=input_generator
                )[:8]
                for _ in range(tokens)
            ]
        ).to(torch.int32)
        weights = torch.softmax(
            torch.randn(
                tokens, 8, device="cuda", generator=input_generator
            ),
            -1,
        ).float()
        expected = reference(x.float(), ids, weights, gate, up, down)
        actual_runs = []
        output_hashes = []
        for _ in range(args.repeats):
            actual = runtime(x, weights, ids).float()
            torch.cuda.synchronize()
            actual_runs.append(actual)
            output_hashes.append(tensor_sha256(actual))
        actual = actual_runs[0]
        timing = None
        if args.timing_repeats:
            if args.timing_warmups < 0 or args.timing_repeats < 1:
                raise ValueError("timing counts must be nonnegative with at least one repeat")
            for _ in range(args.timing_warmups):
                runtime(x, weights, ids)
            torch.cuda.synchronize()
            samples = []
            for _ in range(args.timing_repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                runtime(x, weights, ids)
                end.record()
                end.synchronize()
                samples.append(float(start.elapsed_time(end)))
            timing = {
                "warmups": args.timing_warmups,
                "repeats": args.timing_repeats,
                "samples_ms": samples,
                "median_ms": statistics.median(samples),
                "min_ms": min(samples),
                "max_ms": max(samples),
            }
        graph_result = None
        if args.cuda_graph_replays or args.cuda_graph_timing_repeats:
            if args.cuda_graph_warmups < 1:
                raise ValueError("CUDA graph capture requires at least one warmup")
            if args.cuda_graph_replays < 5:
                raise ValueError("CUDA graph closure requires at least five replays")
            if args.cuda_graph_timing_repeats < 0:
                raise ValueError("CUDA graph timing count must be nonnegative")
            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                for _ in range(args.cuda_graph_warmups):
                    runtime(x, weights, ids)
            torch.cuda.current_stream().wait_stream(warmup_stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_output = runtime(x, weights, ids)
            graph_hashes = []
            for _ in range(args.cuda_graph_replays):
                graph.replay()
                torch.cuda.synchronize()
                graph_hashes.append(tensor_sha256(graph_output.float()))
            graph_samples = []
            for _ in range(args.cuda_graph_timing_repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                graph_samples.append(float(start.elapsed_time(end)))
            graph_result = {
                "warmups": args.cuda_graph_warmups,
                "replays": args.cuda_graph_replays,
                "output_sha256_runs": graph_hashes,
                "bitwise_deterministic": len(set(graph_hashes)) == 1,
                "matches_eager_output": graph_hashes[0] == output_hashes[0],
                "timing_repeats": args.cuda_graph_timing_repeats,
                "samples_ms": graph_samples,
                "median_ms": statistics.median(graph_samples) if graph_samples else None,
                "min_ms": min(graph_samples) if graph_samples else None,
                "max_ms": max(graph_samples) if graph_samples else None,
            }
        cosine = float(F.cosine_similarity(actual.reshape(1, -1), expected.reshape(1, -1)))
        relative_l2 = float((actual - expected).norm() / expected.norm().clamp_min(1e-9))
        bitwise_deterministic = len(set(output_hashes)) == 1
        cells.append(
            {
                "tokens": tokens,
                "payload_sha256": {
                    "x": tensor_sha256(x),
                    "topk_ids": tensor_sha256(ids),
                    "topk_weights": tensor_sha256(weights),
                    "dense_reference": tensor_sha256(expected),
                },
                "cosine": cosine,
                "relative_l2": relative_l2,
                "finite": bool(torch.isfinite(actual).all()),
                "output_sha256": output_hashes[0],
                "output_sha256_runs": output_hashes,
                "bitwise_deterministic": bitwise_deterministic,
                "timing": timing,
                "cuda_graph": graph_result,
                "pass": bool(
                    torch.isfinite(actual).all()
                    and cosine > 0.995
                    and relative_l2 < 0.12
                    and (args.repeats == 1 or bitwise_deterministic)
                    and (
                        graph_result is None
                        or (
                            graph_result["bitwise_deterministic"]
                            and graph_result["matches_eager_output"]
                        )
                    )
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
        "requested_mode": args.mode,
        "resolved_mode": "small_m" if args.small_m_scheduler else args.mode,
        "mode": args.mode,
        "max_active_clusters": args.mac,
        "deterministic_output": args.deterministic_output,
        "small_m_scheduler": args.small_m_scheduler,
        "repeats": args.repeats,
        "cuda_graph": bool(args.cuda_graph_replays),
        "input_rng": "dedicated CUDA generator seeded after runtime construction",
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
