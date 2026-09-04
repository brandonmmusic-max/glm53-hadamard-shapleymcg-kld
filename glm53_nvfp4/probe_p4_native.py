"""Synthetic P4 probe. Default is CPU-only; device closure must be explicit.

No model path, role corpus, or service is used. --device-closure is a seam for
the parent's separately authorized device review, not part of agent validation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .p4_reference import (
    decode_projection, moe_reference, synthetic_payload, tensor_sha256,
)


def inputs(tokens: int):
    x = ((torch.arange(tokens * 128).reshape(tokens, 128).float() % 29 - 14) / 32).to(torch.bfloat16)
    ids = (torch.arange(tokens * 2).reshape(tokens, 2) + 2) % 3
    weights = torch.tensor([.25, .75]).expand(tokens, 2).contiguous()
    return x, ids, weights


def cpu_probe() -> dict:
    tensors, metadata = synthetic_payload()
    x, ids, weights = inputs(3)
    outputs = [moe_reference(tensors, x.float(), ids, weights) for _ in range(5)]
    hashes = [tensor_sha256(t) for t in outputs]
    return {
        "schema": "glm53-p4-astra-structural-probe.v1",
        "evidence_level": "structural", "fixture": "synthetic-only-seed-20260904",
        "payload_metadata": metadata,
        "payload_sha256": {name: tensor_sha256(value) for name, value in tensors.items()},
        "packed_sha256": {
            "gate": tensor_sha256(decode_projection(tensors["w13_trellis"][0])),
            "up": tensor_sha256(decode_projection(tensors["w13_trellis"][1])),
            "down": tensor_sha256(decode_projection(tensors["w2_trellis"])),
        },
        "cpu_output_sha256_runs": hashes,
        "pass": len(set(hashes)) == 1 and all(bool(torch.isfinite(o).all()) for o in outputs),
        "gpu_executed": False, "protected_roles_opened": [],
        "claim": "CPU reference determinism only; no device, KLD, or speed claim",
    }


def device_probe(device: str, tokens: list[int], repeats: int) -> dict:
    from runtime_patch.p4_native_kernel import P4NativeTPMoE

    tensors, metadata = synthetic_payload()
    runtime = P4NativeTPMoE.from_tensors(
        tensors, metadata, device=torch.device(device), layer=3, tp_rank=0,
        expected_design_sha256=metadata["source_design_sha256"], topk=2)
    # Probe exact nibble outputs of the same decoder used by the hot path.
    decode_cells = []
    with torch.cuda.device(runtime.device):
        lib = runtime.compile()
        stream = torch.cuda.current_stream(runtime.device)
        for name, cpu_stream, device_stream in (
            ("gate", tensors["w13_trellis"][0], runtime.tensors["w13_trellis"][0]),
            ("up", tensors["w13_trellis"][1], runtime.tensors["w13_trellis"][1]),
            ("down", tensors["w2_trellis"], runtime.tensors["w2_trellis"]),
        ):
            expected = decode_projection(cpu_stream)
            packed = torch.empty_like(expected, device=runtime.device)
            e, n, k2 = expected.shape
            runtime._check(lib.p4_decode_probe(device_stream.data_ptr(), packed.data_ptr(),
                                               e, n, k2 * 2, stream.cuda_stream))
            packed.record_stream(stream)
            actual = packed.cpu()
            decode_cells.append({"projection": name, "byte_exact": torch.equal(expected, actual),
                                 "expected_sha256": tensor_sha256(expected), "actual_sha256": tensor_sha256(actual)})
        cells = []
        for m in tokens:
            x, ids, weights = inputs(m)
            expected = moe_reference(tensors, x.float(), ids, weights, return_intermediates=True)
            xd, wd, idd = x.to(runtime.device), weights.to(runtime.device), ids.to(runtime.device)
            hashes, stages = [], []
            for _ in range(repeats):
                result = runtime(xd, wd, idd, return_intermediates=True)
                host = tuple(value.float().cpu() for value in result)
                hashes.append(tensor_sha256(host[0]))
                if not stages:
                    for name, actual, ref in zip(("output", "fc1", "swiglu", "fc2"), host, expected, strict=True):
                        rel = float((actual - ref).norm() / ref.norm().clamp_min(1e-20))
                        maximum = float((actual - ref).abs().max())
                        stages.append({"stage": name, "relative_l2": rel if math.isfinite(rel) else None,
                                       "max_abs": maximum if math.isfinite(maximum) else None,
                                       "finite": bool(torch.isfinite(actual).all()),
                                       "actual_sha256": tensor_sha256(actual),
                                       "reference_sha256": tensor_sha256(ref)})
            cells.append({"tokens": m, "stages": stages, "output_sha256_runs": hashes,
                          "bitwise_deterministic": len(set(hashes)) == 1})
    # Diagnostic limits are declared in source before device execution.
    # They do not replace the parent's real-model arithmetic/KLD protocol.
    passed = all(d["byte_exact"] for d in decode_cells) and all(
        c["bitwise_deterministic"] and all(s["finite"] and s["relative_l2"] is not None
                                         and s["relative_l2"] <= .025 for s in c["stages"])
        for c in cells)
    return {"schema": "glm53-p4-astra-synthetic-device-probe.v1", "evidence_level": "gpu-smoke",
            "device": device, "tokens": tokens, "repeats": repeats, "decode": decode_cells, "cells": cells,
            "decision_rule": "exact decode bytes, finite FC1/SwiGLU/FC2/output, each relative L2 <= .025, repeated output hashes equal",
            "pass": passed, "gpu_executed": True, "protected_roles_opened": [],
            "claim": "synthetic device diagnostic only; no real-model, KLD, or speed qualification"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-closure", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 3, 17, 33])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.repeats < 1 or any(m < 1 for m in args.tokens):
        parser.error("positive tokens and repeats required")
    torch.set_num_threads(1)
    result = {"pass": False, "status": "incomplete", "gpu_requested": args.device_closure}
    try:
        result = device_probe(args.device, args.tokens, args.repeats) if args.device_closure else cpu_probe()
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
