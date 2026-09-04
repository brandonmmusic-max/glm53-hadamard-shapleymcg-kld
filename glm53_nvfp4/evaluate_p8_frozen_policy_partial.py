"""Accumulate one expert range for a frozen P8 policy over complete windows."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .canary_mxfp6_reap import _qdq_e4m3_k32
from .evaluate_p8_joint_route_contributions import _fc1_h128, _h128, _identity, _weights
from .shard_index import IndexedCheckpoint, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--identity-dense", type=Path, required=True)
    parser.add_argument("--h128-dense", type=Path, required=True)
    parser.add_argument("--gptq-dense", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--expert-start", type=int, required=True)
    parser.add_argument("--expert-end", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    policy = json.loads(args.policy.read_text())
    if sha256_file(args.policy) != plan["policy_sha256"] or policy.get("ldlq_used") is not False:
        raise RuntimeError("frozen policy identity mismatch")
    states = policy["expert_states"]
    manifest = json.loads((args.capture_root / "capture-manifest.json").read_text())
    windows = {item["window_id"]: item for item in manifest["windows"]}
    indices = [windows[item["id"]]["window_index"] for item in plan["windows"]]
    rows_per_window = plan["rows_per_window"]
    global_rows = np.concatenate(
        [np.arange(index * rows_per_window, (index + 1) * rows_per_window) for index in indices]
    )
    total_rows, hidden_size = manifest["file_abi"]["hidden_bf16"]["shape"]
    top_k = manifest["file_abi"]["topk_ids_u16le"]["shape"][1]
    layer_root = args.capture_root / f"layers/layer-{plan['layer']:03d}"
    hidden_words = np.memmap(layer_root / "hidden.bf16.bin", mode="r", dtype="<u2", shape=(total_rows, hidden_size))
    ids_map = np.memmap(layer_root / "topk_ids.u16le.bin", mode="r", dtype="<u2", shape=(total_rows, top_k))
    route_map = np.memmap(layer_root / "topk_weights.f32le.bin", mode="r", dtype="<f4", shape=(total_rows, top_k))
    ids, routes = np.asarray(ids_map[global_rows]), np.asarray(route_map[global_rows])

    reference_sum = torch.zeros((len(global_rows), hidden_size), dtype=torch.float32)
    candidate_error = torch.zeros_like(reference_sum)
    gptq_error = torch.zeros_like(reference_sum)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    source = IndexedCheckpoint(args.source, args.source_index)
    prefix = source.expert_prefix(plan["layer"], args.expert_start).split(f"layers.{plan['layer']}.")[0]
    started, route_count = time.time(), 0
    with (
        safe_open(str(args.identity_dense), framework="pt", device="cpu") as identity_handle,
        safe_open(str(args.h128_dense), framework="pt", device="cpu") as h128_handle,
        safe_open(str(args.gptq_dense), framework="pt", device="cpu") as gptq_handle,
        safe_open(str(args.boundary), framework="pt", device="cpu") as boundary_handle,
    ):
        diagonal = boundary_handle.get_tensor("down_diagonal").float()
        for local, expert in enumerate(range(args.expert_start, args.expert_end)):
            token, slot = np.nonzero(ids == expert)
            route_count += len(token)
            base = f"{prefix}layers.{plan['layer']}.mlp.experts.{expert}"
            original = {name: source.get(f"{base}.{name}.weight").to(device).float() for name in ("gate_proj", "up_proj", "down_proj")}
            identity_weights = _weights(identity_handle, base, device)
            h128_weights = _weights(h128_handle, base, device)
            gptq_weights = _weights(gptq_handle, base, device)
            for start in range(0, len(token), args.batch_size):
                stop = min(start + args.batch_size, len(token))
                index = torch.from_numpy(token[start:stop].astype(np.int64))
                words = np.array(hidden_words[global_rows[token[start:stop]]], copy=True)
                hidden = torch.from_numpy(words).view(torch.bfloat16).to(device).float()
                carrier = _qdq_e4m3_k32(hidden, 1.0, "amax").float()
                reference = _identity(carrier, original)
                if states[expert] == 0:
                    candidate = _identity(carrier, identity_weights)
                elif states[expert] == 1:
                    candidate = _h128(carrier, h128_weights, diagonal[local].to(device))
                elif states[expert] == 2:
                    candidate = _fc1_h128(carrier, h128_weights, identity_weights)
                else:
                    raise RuntimeError(f"invalid policy state for expert {expert}")
                gptq = _identity(carrier, gptq_weights)
                route = torch.from_numpy(routes[token[start:stop], slot[start:stop]].copy()).to(device)[:, None]
                reference_sum.index_add_(0, index, (reference * route).cpu())
                candidate_error.index_add_(0, index, ((candidate - reference) * route).cpu())
                gptq_error.index_add_(0, index, ((gptq - reference) * route).cpu())
            del original, identity_weights, h128_weights, gptq_weights
            torch.cuda.empty_cache()
    save_file(
        {
            "reference_sum": reference_sum.to(torch.float16),
            "candidate_error": candidate_error.to(torch.float16),
            "gptq_error": gptq_error.to(torch.float16),
        },
        str(args.output),
        metadata={
            "schema": "glm53-p8-frozen-policy-partial.v1",
            "expert_start": str(args.expert_start),
            "expert_end": str(args.expert_end),
            "policy_sha256": plan["policy_sha256"],
            "ldlq": "false",
        },
    )
    receipt = {
        "output": str(args.output), "sha256": sha256_file(args.output),
        "expert_range": [args.expert_start, args.expert_end], "routes": route_count,
        "tokens": len(global_rows), "elapsed_seconds": time.time() - started,
    }
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
