"""Optimize and validate a tiny P8 mask on the actual routed MoE sum."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from .paired_role_analysis import bca_mean_interval
from .shard_index import sha256_file


def coordinate_policy(
    identity_error: torch.Tensor,
    rows: list[torch.Tensor],
    deltas: list[torch.Tensor],
    starts: list[torch.Tensor],
    maximum_passes: int,
) -> tuple[torch.Tensor, float, list[dict[str, object]]]:
    """Deterministically minimize joint SSE over binary expert choices."""
    candidates = []
    for start_index, start in enumerate(starts):
        mask = start.bool().clone()
        residual = identity_error.clone()
        for expert, selected in enumerate(mask.tolist()):
            if selected:
                residual.index_add_(0, rows[expert], deltas[expert])
        flips_by_pass = []
        for _ in range(maximum_passes):
            flips = 0
            for expert in range(len(rows)):
                index, delta = rows[expert], deltas[expert]
                change = -delta if bool(mask[expert]) else delta
                current = residual.index_select(0, index)
                difference = float((2.0 * current * change + change.square()).double().sum())
                if difference < -1e-12:
                    residual.index_add_(0, index, change)
                    mask[expert] = ~mask[expert]
                    flips += 1
            flips_by_pass.append(flips)
            if flips == 0:
                break
        candidates.append(
            {
                "start_index": start_index,
                "mask": mask,
                "sse": float(residual.double().square().sum()),
                "flips_by_pass": flips_by_pass,
            }
        )
    winner = min(candidates, key=lambda item: (item["sse"], item["mask"].tolist()))
    trace = [
        {key: value for key, value in item.items() if key != "mask"}
        for item in candidates
    ]
    return winner["mask"], winner["sse"], trace


def _load_phase(root: Path, phase: str, tokens: int, hidden: int = 4096):
    reference = torch.zeros((tokens, hidden), dtype=torch.float32)
    identity = torch.zeros_like(reference)
    gptq = torch.zeros_like(reference)
    rows, deltas, hashes = [], [], []
    for expert in range(288):
        path = root / phase / f"expert-{expert:03d}.safetensors"
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if metadata.get("ldlq") != "false" or int(metadata.get("expert", -1)) != expert:
                raise RuntimeError(f"invalid joint contribution: {path}")
            index = handle.get_tensor("token_rows").long()
            reference.index_add_(0, index, handle.get_tensor("reference").float())
            identity.index_add_(0, index, handle.get_tensor("identity_error").float())
            gptq.index_add_(0, index, handle.get_tensor("gptq_error").float())
            rows.append(index)
            deltas.append(handle.get_tensor("h128_delta").float())
        hashes.append({"path": str(path), "sha256": sha256_file(path)})
    return reference, identity, gptq, rows, deltas, hashes


def _residual(base: torch.Tensor, rows, deltas, mask: torch.Tensor) -> torch.Tensor:
    value = base.clone()
    for expert, selected in enumerate(mask.tolist()):
        if selected:
            value.index_add_(0, rows[expert], deltas[expert])
    return value


def _window_nmse(reference: torch.Tensor, error: torch.Tensor, rows_per_window: int) -> list[float]:
    values = []
    for start in range(0, len(reference), rows_per_window):
        stop = start + rows_per_window
        numerator = error[start:stop].double().square().sum()
        denominator = reference[start:stop].double().square().sum().clamp_min(1e-30)
        values.append(float(numerator / denominator))
    return values


def _mask_bytes(mask: torch.Tensor) -> str:
    packed = np.packbits(mask.numpy().astype(np.uint8), bitorder="little")
    return packed.tobytes().hex()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--contributions", type=Path, required=True)
    parser.add_argument("--previous-policy", type=Path, required=True)
    parser.add_argument("--policy-output", type=Path, required=True)
    parser.add_argument("--analysis-output", type=Path, required=True)
    args = parser.parse_args()
    if args.policy_output.exists() or args.analysis_output.exists():
        raise FileExistsError("refusing to overwrite joint policy outputs")
    plan = json.loads(args.plan.read_text())
    previous = json.loads(args.previous_policy.read_text())
    tokens = len(plan["sampling"]["selection"]) * plan["sampling"]["rows_per_window"]

    selection = _load_phase(args.contributions, "selection", tokens)
    selection_reference, selection_identity, selection_gptq, rows, deltas, selection_hashes = selection
    prior_mask = torch.zeros(288, dtype=torch.bool)
    prior_mask[previous["h128_experts"]] = True
    starts = [torch.zeros(288, dtype=torch.bool), torch.ones(288, dtype=torch.bool), prior_mask]
    mask, selection_sse, trace = coordinate_policy(
        selection_identity,
        rows,
        deltas,
        starts,
        int(plan["optimizer"]["maximum_passes"]),
    )
    selection_errors = {
        "identity": selection_identity,
        "h128": _residual(selection_identity, rows, deltas, torch.ones(288, dtype=torch.bool)),
        "previous": _residual(selection_identity, rows, deltas, prior_mask),
        "joint": _residual(selection_identity, rows, deltas, mask),
        "gptq": selection_gptq,
    }

    validation = _load_phase(args.contributions, "validation", tokens)
    validation_reference, validation_identity, validation_gptq, val_rows, val_deltas, validation_hashes = validation
    validation_errors = {
        "identity": validation_identity,
        "h128": _residual(validation_identity, val_rows, val_deltas, torch.ones(288, dtype=torch.bool)),
        "previous": _residual(validation_identity, val_rows, val_deltas, prior_mask),
        "joint": _residual(validation_identity, val_rows, val_deltas, mask),
        "gptq": validation_gptq,
    }
    rows_per_window = int(plan["sampling"]["rows_per_window"])
    selection_nmse = {
        name: _window_nmse(selection_reference, value, rows_per_window)
        for name, value in selection_errors.items()
    }
    validation_nmse = {
        name: _window_nmse(validation_reference, value, rows_per_window)
        for name, value in validation_errors.items()
    }
    joint = np.asarray(validation_nmse["joint"])
    gptq = np.asarray(validation_nmse["gptq"])
    log_ratio = np.log(joint / gptq)
    spec = plan["bootstrap"]
    rng = np.random.default_rng(spec["seed"])
    indices = rng.integers(0, len(log_ratio), size=(spec["replicates"], len(log_ratio)))
    bootstrap = log_ratio[indices].mean(axis=1)
    interval = bca_mean_interval(log_ratio, bootstrap)
    improvement = 1.0 - float(joint.sum() / gptq.sum())
    wins = int((joint < gptq).sum())
    passed = improvement >= 0.10 and wins >= 6 and interval[1] < 0

    h128_experts = [expert for expert, selected in enumerate(mask.tolist()) if selected]
    policy = {
        "schema": "glm53-p8-joint-routed-sum-policy.v1",
        "plan_sha256": sha256_file(args.plan),
        "phase": "selection",
        "h128_experts": h128_experts,
        "h128_count": len(h128_experts),
        "identity_count": 288 - len(h128_experts),
        "expert_mask_hex_little_endian": _mask_bytes(mask),
        "expert_mask_bytes": 36,
        "checkpoint_family_law_ids_bytes": 3,
        "policy_bytes": 39,
        "selection_sse": selection_sse,
        "optimizer_trace": trace,
        "protected_roles_opened": [],
        "ldlq_used": False,
    }
    args.policy_output.parent.mkdir(parents=True, exist_ok=True)
    args.policy_output.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")

    domains = [item["domain"] for item in plan["sampling"]["validation"]]
    domain_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    window_rows = []
    for item, candidate, control in zip(plan["sampling"]["validation"], joint, gptq, strict=True):
        row = {
            "window": item["id"],
            "domain": item["domain"],
            "joint_nmse": float(candidate),
            "gptq_nmse": float(control),
            "log_ratio": float(math.log(candidate / control)),
        }
        window_rows.append(row)
        domain_rows[item["domain"]].append(row)
    analysis = {
        "schema": "glm53-p8-joint-routed-sum-policy-analysis.v1",
        "plan_sha256": sha256_file(args.plan),
        "policy_sha256": sha256_file(args.policy_output),
        "tokens_per_phase": tokens,
        "selection_window_nmse": selection_nmse,
        "validation_window_nmse": validation_nmse,
        "validation_joint_vs_gptq": {
            "relative_improvement": improvement,
            "window_wins": wins,
            "mean_log_ratio": float(log_ratio.mean()),
            "mean_log_ratio_ci95_bca": [float(interval[0]), float(interval[1])],
            "mean_log_ratio_ci95_percentile": [float(x) for x in np.quantile(bootstrap, (0.025, 0.975))],
        },
        "validation_by_domain": {
            domain: {
                "joint_geometric_mean_nmse": float(np.exp(np.mean(np.log([row["joint_nmse"] for row in values])))),
                "gptq_geometric_mean_nmse": float(np.exp(np.mean(np.log([row["gptq_nmse"] for row in values])))),
            }
            for domain, values in sorted(domain_rows.items())
        },
        "window_results": window_rows,
        "decision": "pass-freeze-for-unopened-kld" if passed else "fail-joint-route-policy",
        "decision_rule": plan["validation_rule"],
        "input_files": selection_hashes + validation_hashes,
        "protected_roles_opened": [],
        "confirmation_logits_opened": False,
        "ldlq_used": False,
    }
    args.analysis_output.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "decision": analysis["decision"],
        "policy_sha256": analysis["policy_sha256"],
        "analysis_sha256": sha256_file(args.analysis_output),
        "h128_count": len(h128_experts),
        **analysis["validation_joint_vs_gptq"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
