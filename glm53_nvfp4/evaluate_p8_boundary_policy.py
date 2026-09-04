"""Evaluate identity and scaled-H128 decoded P8 payloads per expert."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from .canary_mxfp6_reap import _qdq_e4m3_k32
from .capture import LayerCapture
from .p8_h128 import hadamard128_last
from .shard_index import IndexedCheckpoint, sha256_file


def _forward_identity(hidden: torch.Tensor, weights: dict[str, torch.Tensor]) -> torch.Tensor:
    carrier = _qdq_e4m3_k32(hidden, 1.0, "amax").float()
    gate = F.linear(carrier, weights["gate_proj"]).clamp(max=10.0)
    up = F.linear(carrier, weights["up_proj"]).clamp(-10.0, 10.0)
    middle = _qdq_e4m3_k32(F.silu(gate) * up, 1.0, "amax").float()
    return F.linear(middle, weights["down_proj"])


def _forward_h128(
    hidden: torch.Tensor, weights: dict[str, torch.Tensor], diagonal: torch.Tensor
) -> torch.Tensor:
    carrier = _qdq_e4m3_k32(hidden, 1.0, "amax").float()
    gate = hadamard128_last(F.linear(carrier, weights["gate_proj"])).clamp(max=10.0)
    up = hadamard128_last(F.linear(carrier, weights["up_proj"])).clamp(-10.0, 10.0)
    middle = hadamard128_last(F.silu(gate) * up * diagonal)
    middle = _qdq_e4m3_k32(middle, 1.0, "amax").float()
    return F.linear(middle, weights["down_proj"])


def _nmse(reference: torch.Tensor, actual: torch.Tensor, route: torch.Tensor) -> float:
    target = reference.float() * route[:, None]
    error = (actual.float() - reference.float()) * route[:, None]
    return float((error.double().square().sum() / target.double().square().sum().clamp_min(1e-30)).item())


def _weights(handle, base: str, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: handle.get_tensor(f"{base}.{name}.weight").to(device).float()
        for name in ("gate_proj", "up_proj", "down_proj")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--phase", choices=("selection", "validation"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--h128-dense", type=Path, required=True)
    parser.add_argument("--identity-dense", type=Path, required=True)
    parser.add_argument("--gptq-dense", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--expert-start", type=int, required=True)
    parser.add_argument("--expert-end", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    sample = plan["sampling"][args.phase]
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    source = IndexedCheckpoint(args.source, args.source_index)
    capture = LayerCapture(
        args.capture_root, plan["layer"], args.roles,
        max_samples=sample["count"], sample_offset=sample["offset"],
        data_role=sample["role"], sampling_strategy=sample["strategy"],
    )
    prefix = source.expert_prefix(plan["layer"], args.expert_start).split(
        f"layers.{plan['layer']}."
    )[0]
    started, rows = time.time(), []
    with (
        safe_open(str(args.h128_dense), framework="pt", device="cpu") as h128_handle,
        safe_open(str(args.identity_dense), framework="pt", device="cpu") as identity_handle,
        safe_open(str(args.gptq_dense), framework="pt", device="cpu") as gptq_handle,
        safe_open(str(args.boundary), framework="pt", device="cpu") as boundary_handle,
    ):
        diagonal_all = boundary_handle.get_tensor("down_diagonal").float()
        for local, expert in enumerate(range(args.expert_start, args.expert_end)):
            base = f"{prefix}layers.{plan['layer']}.mlp.experts.{expert}"
            original = {
                name: source.get(f"{base}.{name}.weight").to(device).float()
                for name in ("gate_proj", "up_proj", "down_proj")
            }
            h128 = _weights(h128_handle, base, device)
            identity = _weights(identity_handle, base, device)
            gptq = _weights(gptq_handle, base, device)
            hidden_cpu, route_cpu = capture.samples(expert)
            hidden, route = hidden_cpu.to(device).float(), route_cpu.to(device).float()
            reference = _forward_identity(hidden, original)
            h128_output = _forward_h128(hidden, h128, diagonal_all[local].to(device))
            identity_output = _forward_identity(hidden, identity)
            gptq_output = _forward_identity(hidden, gptq)
            row = {
                "expert": expert,
                "h128_output_nmse": _nmse(reference, h128_output, route),
                "identity_output_nmse": _nmse(reference, identity_output, route),
                "gptq_output_nmse": _nmse(reference, gptq_output, route),
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            del original, h128, identity, gptq, hidden, route, reference
            del h128_output, identity_output, gptq_output
            torch.cuda.empty_cache()
    payload = {
        "schema": "glm53-p8-boundary-policy-evaluation.v1",
        "plan_sha256": sha256_file(args.plan),
        "phase": args.phase,
        "expert_range": [args.expert_start, args.expert_end],
        "sampling": sample,
        "rows": rows,
        "inputs": {
            "h128_dense_sha256": sha256_file(args.h128_dense),
            "identity_dense_sha256": sha256_file(args.identity_dense),
            "gptq_dense_sha256": sha256_file(args.gptq_dense),
            "boundary_sha256": sha256_file(args.boundary),
        },
        "elapsed_seconds": time.time() - started,
        "protected_roles_opened": [],
        "ldlq_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
