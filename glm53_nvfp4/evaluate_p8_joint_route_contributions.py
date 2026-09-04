"""Materialize per-route errors for a joint top-8 P8 policy objective."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from .canary_mxfp6_reap import _qdq_e4m3_k32
from .p8_h128 import hadamard128_last
from .shard_index import IndexedCheckpoint, sha256_file


def _weights(handle, base: str, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: handle.get_tensor(f"{base}.{name}.weight").to(device).float()
        for name in ("gate_proj", "up_proj", "down_proj")
    }


def _identity(carrier: torch.Tensor, weights: dict[str, torch.Tensor]) -> torch.Tensor:
    gate = F.linear(carrier, weights["gate_proj"]).clamp(max=10.0)
    up = F.linear(carrier, weights["up_proj"]).clamp(-10.0, 10.0)
    middle = _qdq_e4m3_k32(F.silu(gate) * up, 1.0, "amax").float()
    return F.linear(middle, weights["down_proj"])


def _h128(
    carrier: torch.Tensor, weights: dict[str, torch.Tensor], diagonal: torch.Tensor
) -> torch.Tensor:
    gate = hadamard128_last(F.linear(carrier, weights["gate_proj"])).clamp(max=10.0)
    up = hadamard128_last(F.linear(carrier, weights["up_proj"])).clamp(-10.0, 10.0)
    middle = hadamard128_last(F.silu(gate) * up * diagonal)
    middle = _qdq_e4m3_k32(middle, 1.0, "amax").float()
    return F.linear(middle, weights["down_proj"])


def _fc1_h128(
    carrier: torch.Tensor,
    h128_weights: dict[str, torch.Tensor],
    identity_weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Use H128 only where its all-expert projection evidence was positive."""
    gate = hadamard128_last(F.linear(carrier, h128_weights["gate_proj"])).clamp(max=10.0)
    up = hadamard128_last(F.linear(carrier, h128_weights["up_proj"])).clamp(-10.0, 10.0)
    middle = _qdq_e4m3_k32(F.silu(gate) * up, 1.0, "amax").float()
    return F.linear(middle, identity_weights["down_proj"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--phase", choices=("selection", "validation"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--identity-dense", type=Path, required=True)
    parser.add_argument("--h128-dense", type=Path, required=True)
    parser.add_argument("--gptq-dense", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--expert-start", type=int, required=True)
    parser.add_argument("--expert-end", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(f"refusing to overwrite {args.receipt}")
    if not 0 <= args.expert_start < args.expert_end <= 288:
        raise ValueError("invalid expert range")

    plan = json.loads(args.plan.read_text())
    manifest = json.loads((args.capture_root / "capture-manifest.json").read_text())
    windows = {item["window_id"]: item for item in manifest["windows"]}
    chosen = plan["sampling"][args.phase]
    indices = [int(windows[item["id"]]["window_index"]) for item in chosen]
    rows_per_window = int(plan["sampling"]["rows_per_window"])
    global_rows = np.concatenate(
        [np.arange(index * rows_per_window, (index + 1) * rows_per_window) for index in indices]
    )
    abi = manifest["file_abi"]
    total_rows, hidden_size = abi["hidden_bf16"]["shape"]
    top_k = abi["topk_ids_u16le"]["shape"][1]
    layer_root = args.capture_root / f"layers/layer-{plan['layer']:03d}"
    hidden_words = np.memmap(
        layer_root / "hidden.bf16.bin", mode="r", dtype="<u2", shape=(total_rows, hidden_size)
    )
    ids_map = np.memmap(
        layer_root / "topk_ids.u16le.bin", mode="r", dtype="<u2", shape=(total_rows, top_k)
    )
    route_map = np.memmap(
        layer_root / "topk_weights.f32le.bin", mode="r", dtype="<f4", shape=(total_rows, top_k)
    )
    ids = np.asarray(ids_map[global_rows])
    routes = np.asarray(route_map[global_rows])

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    source = IndexedCheckpoint(args.source, args.source_index)
    prefix = source.expert_prefix(plan["layer"], args.expert_start).split(
        f"layers.{plan['layer']}."
    )[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started, files = time.time(), []
    with (
        safe_open(str(args.identity_dense), framework="pt", device="cpu") as identity_handle,
        safe_open(str(args.h128_dense), framework="pt", device="cpu") as h128_handle,
        safe_open(str(args.gptq_dense), framework="pt", device="cpu") as gptq_handle,
        safe_open(str(args.boundary), framework="pt", device="cpu") as boundary_handle,
    ):
        diagonal = boundary_handle.get_tensor("down_diagonal").float()
        for local, expert in enumerate(range(args.expert_start, args.expert_end)):
            token, slot = np.nonzero(ids == expert)
            if len(token) == 0:
                raise RuntimeError(f"expert {expert} has no routes in {args.phase}")
            weights_route = torch.from_numpy(routes[token, slot].copy()).float()
            base = f"{prefix}layers.{plan['layer']}.mlp.experts.{expert}"
            original = {
                name: source.get(f"{base}.{name}.weight").to(device).float()
                for name in ("gate_proj", "up_proj", "down_proj")
            }
            identity_weights = _weights(identity_handle, base, device)
            h128_weights = _weights(h128_handle, base, device)
            gptq_weights = _weights(gptq_handle, base, device)
            contributions = {
                name: []
                for name in (
                    "reference",
                    "identity_error",
                    "h128_delta",
                    "fc1_h128_delta",
                    "gptq_error",
                )
            }
            for start in range(0, len(token), args.batch_size):
                stop = min(start + args.batch_size, len(token))
                words = np.array(hidden_words[global_rows[token[start:stop]]], copy=True)
                hidden = torch.from_numpy(words).view(torch.bfloat16).to(device).float()
                carrier = _qdq_e4m3_k32(hidden, 1.0, "amax").float()
                reference = _identity(carrier, original)
                identity_output = _identity(carrier, identity_weights)
                h128_output = _h128(carrier, h128_weights, diagonal[local].to(device))
                fc1_h128_output = _fc1_h128(carrier, h128_weights, identity_weights)
                gptq_output = _identity(carrier, gptq_weights)
                route = weights_route[start:stop].to(device)[:, None]
                contributions["reference"].append((reference * route).to(torch.float16).cpu())
                identity_error = (identity_output - reference) * route
                contributions["identity_error"].append(identity_error.to(torch.float16).cpu())
                contributions["h128_delta"].append(
                    ((h128_output - identity_output) * route).to(torch.float16).cpu()
                )
                contributions["fc1_h128_delta"].append(
                    ((fc1_h128_output - identity_output) * route).to(torch.float16).cpu()
                )
                contributions["gptq_error"].append(
                    ((gptq_output - reference) * route).to(torch.float16).cpu()
                )
            tensors = {
                "token_rows": torch.from_numpy(token.astype(np.int32)),
                **{name: torch.cat(parts).contiguous() for name, parts in contributions.items()},
            }
            output = args.output_dir / f"expert-{expert:03d}.safetensors"
            if output.exists():
                raise FileExistsError(f"refusing to overwrite {output}")
            save_file(
                tensors,
                str(output),
                metadata={
                    "schema": "glm53-p8-joint-route-contribution.v2",
                    "phase": args.phase,
                    "expert": str(expert),
                    "ldlq": "false",
                },
            )
            row = {"expert": expert, "routes": len(token), "path": str(output), "sha256": sha256_file(output)}
            files.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            del original, identity_weights, h128_weights, gptq_weights, tensors
            torch.cuda.empty_cache()
    receipt = {
        "schema": "glm53-p8-joint-route-contribution-receipt.v1",
        "plan_sha256": sha256_file(args.plan),
        "phase": args.phase,
        "expert_range": [args.expert_start, args.expert_end],
        "window_ids": [item["id"] for item in chosen],
        "tokens": len(global_rows),
        "files": files,
        "elapsed_seconds": time.time() - started,
        "protected_roles_opened": [],
        "ldlq_used": False,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"receipt": str(args.receipt), "sha256": sha256_file(args.receipt)}, sort_keys=True))


if __name__ == "__main__":
    main()
