"""Compare physical P8 scale contracts on disjoint fit captures."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from .canary_mxfp6_reap import _qdq_e4m3_k32
from .capture import LayerCapture
from .paired_role_analysis import bca_mean_interval
from .shard_index import IndexedCheckpoint, sha256_file


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _nmse(reference: torch.Tensor, actual: torch.Tensor) -> float:
    return float(
        ((actual.float() - reference.float()).double().square().sum()
         / reference.float().double().square().sum().clamp_min(1e-30)).item()
    )


def _routed_nmse(reference: torch.Tensor, actual: torch.Tensor, route: torch.Tensor) -> float:
    scale = route.float()[:, None]
    return _nmse(reference.float() * scale, actual.float() * scale)


def _forward(hidden: torch.Tensor, weights: dict[str, torch.Tensor]) -> torch.Tensor:
    carrier = _qdq_e4m3_k32(hidden, 1.0, "amax").float()
    gate = F.linear(carrier, weights["gate_proj"]).clamp(max=10.0)
    up = F.linear(carrier, weights["up_proj"]).clamp(-10.0, 10.0)
    middle = _qdq_e4m3_k32(F.silu(gate) * up, 1.0, "amax").float()
    return F.linear(middle, weights["down_proj"])


def _gmean(values: np.ndarray) -> float:
    return float(math.exp(np.log(values).mean()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    if plan.get("algorithm_exclusion") != "LDLQ and BlockLDLQ are excluded":
        raise RuntimeError("scale-contract screen requires frozen no-LDLQ rule")
    if plan.get("role") != "fit-developmental":
        raise RuntimeError("scale-contract screen may only use the fit role")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    source = IndexedCheckpoint(args.source, args.source_index)
    sample = plan["sampling"]
    capture = LayerCapture(
        args.capture_root,
        plan["layer"],
        args.roles,
        max_samples=sample["count"],
        sample_offset=sample["offset"],
        data_role=sample["role"],
        sampling_strategy=sample["strategy"],
    )
    prefix = source.expert_prefix(plan["layer"], plan["experts"][0]).split(
        f"layers.{plan['layer']}."
    )[0]
    rows: list[dict[str, object]] = []
    started = time.time()
    with safe_open(str(args.dense), framework="pt", device="cpu") as dense, safe_open(
        str(args.codec), framework="pt", device="cpu"
    ) as codec:
        for expert in plan["experts"]:
            base = f"{prefix}layers.{plan['layer']}.mlp.experts.{expert}"
            original = {
                projection: source.get(f"{base}.{projection}.weight").to(device).float()
                for projection in PROJECTIONS
            }
            native = {
                projection: dense.get_tensor(f"{base}.{projection}.weight").to(device).float()
                for projection in PROJECTIONS
            }
            # Exact value seen by an identity-SFB MMA after a fully-scaled
            # E4M3 decode in the prologue.
            identity = {
                projection: weight.to(torch.float8_e4m3fn).float()
                for projection, weight in native.items()
            }
            scale_codes = {
                projection: codec.get_tensor(f"{base}.{projection}.scale_ue8m0")
                for projection in PROJECTIONS
            }
            hidden_cpu, route_cpu = capture.samples(expert)
            hidden, route = hidden_cpu.to(device).float(), route_cpu.to(device).float()
            reference_output = _forward(hidden, original)
            native_output = _forward(hidden, native)
            identity_output = _forward(hidden, identity)
            projection_metrics = {}
            for projection in PROJECTIONS:
                projection_metrics[projection] = {
                    "native_sfb_weight_nmse": _nmse(original[projection], native[projection]),
                    "identity_sfb_weight_nmse": _nmse(original[projection], identity[projection]),
                    "identity_vs_native_extra_nmse": _nmse(native[projection], identity[projection]),
                    "fully_scaled_e4m3_exact_fraction": float(
                        (identity[projection] == native[projection]).float().mean().item()
                    ),
                    "ue8m0_code_min": int(scale_codes[projection].min().item()),
                    "ue8m0_code_max": int(scale_codes[projection].max().item()),
                }
            row = {
                "expert": expert,
                "projection_weight_nmse_first": projection_metrics,
                "native_sfb_full_output_nmse": _routed_nmse(reference_output, native_output, route),
                "identity_sfb_full_output_nmse": _routed_nmse(reference_output, identity_output, route),
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            del original, native, identity, scale_codes, hidden, route
            del reference_output, native_output, identity_output
            torch.cuda.empty_cache()

    native_values = np.asarray([row["native_sfb_full_output_nmse"] for row in rows])
    identity_values = np.asarray([row["identity_sfb_full_output_nmse"] for row in rows])
    log_ratio = np.log(identity_values / native_values)
    bootstrap = plan["bootstrap"]
    rng = np.random.default_rng(bootstrap["seed"])
    choices = rng.integers(0, len(log_ratio), size=(bootstrap["replicates"], len(log_ratio)))
    interval = bca_mean_interval(log_ratio, log_ratio[choices].mean(1))
    relative_change = _gmean(identity_values) / _gmean(native_values) - 1.0
    threshold = float(plan["decision_thresholds"]["maximum_identity_sfb_relative_regression"])
    wins = int((identity_values <= native_values).sum())
    identity_accepted = relative_change <= threshold and wins >= int(
        plan["decision_thresholds"]["minimum_identity_sfb_nonregressions"]
    )
    result = {
        "schema": "glm53-p8-scale-contract-screen.v1",
        "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)},
        "inputs": {
            "source_index_sha256": sha256_file(args.source_index),
            "capture_manifest_sha256": sha256_file(args.capture_root / "capture-manifest.json"),
            "roles_sha256": sha256_file(args.roles),
            "dense": {"path": str(args.dense), "sha256": sha256_file(args.dense)},
            "codec": {"path": str(args.codec), "sha256": sha256_file(args.codec)},
        },
        "projection_weight_nmse_first": [
            {"expert": row["expert"], "metrics": row["projection_weight_nmse_first"]}
            for row in rows
        ],
        "full_output": {
            "native_sfb_geometric_mean_nmse": _gmean(native_values),
            "identity_sfb_geometric_mean_nmse": _gmean(identity_values),
            "identity_sfb_relative_change": relative_change,
            "identity_sfb_nonregressions": wins,
            "mean_log_ratio_ci95_bca": interval,
        },
        "per_expert_full_output": [
            {
                "expert": row["expert"],
                "native_sfb_nmse": row["native_sfb_full_output_nmse"],
                "identity_sfb_nmse": row["identity_sfb_full_output_nmse"],
            }
            for row in rows
        ],
        "decision_rule": plan["decision_rule"],
        "decision": "use-fully-scaled-e4m3-identity-sfb" if identity_accepted else "use-native-ue8m0-sfb",
        "scope": "adaptive fit-only engineering contract; no end-to-end KLD claim",
        "window_uncertainty_policy": "recorded control, not an engineering blocker",
        "protected_roles_opened": [],
        "ldlq": False,
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output), "decision": result["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
