"""Optimize identity/full-H128/FC1-H128 choices on the routed MoE sum."""
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


def coordinate_policy_three(
    identity_error: torch.Tensor,
    rows: list[torch.Tensor],
    full_deltas: list[torch.Tensor],
    fc1_deltas: list[torch.Tensor],
    starts: list[torch.Tensor],
    maximum_passes: int,
) -> tuple[torch.Tensor, float, list[dict[str, object]]]:
    variants = [None, full_deltas, fc1_deltas]
    candidates = []
    for start_index, start in enumerate(starts):
        states = start.to(torch.uint8).clone()
        residual = identity_error.clone()
        for expert, state in enumerate(states.tolist()):
            if state:
                residual.index_add_(0, rows[expert], variants[state][expert])
        changes_by_pass = []
        for _ in range(maximum_passes):
            changes = 0
            for expert, current_state in enumerate(states.tolist()):
                index = rows[expert]
                current_delta = (
                    torch.zeros_like(full_deltas[expert])
                    if current_state == 0
                    else variants[current_state][expert]
                )
                current = residual.index_select(0, index)
                best_state, best_difference, best_change = current_state, 0.0, None
                for candidate_state in range(3):
                    if candidate_state == current_state:
                        continue
                    candidate_delta = (
                        torch.zeros_like(current_delta)
                        if candidate_state == 0
                        else variants[candidate_state][expert]
                    )
                    change = candidate_delta - current_delta
                    difference = float((2.0 * current * change + change.square()).double().sum())
                    if difference < best_difference - 1e-12 or (
                        abs(difference - best_difference) <= 1e-12
                        and candidate_state < best_state
                    ):
                        best_state, best_difference, best_change = candidate_state, difference, change
                if best_state != current_state:
                    residual.index_add_(0, index, best_change)
                    states[expert] = best_state
                    changes += 1
            changes_by_pass.append(changes)
            if changes == 0:
                break
        candidates.append(
            {
                "start_index": start_index,
                "states": states,
                "sse": float(residual.double().square().sum()),
                "changes_by_pass": changes_by_pass,
            }
        )
    winner = min(candidates, key=lambda item: (item["sse"], item["states"].tolist()))
    trace = [{key: value for key, value in item.items() if key != "states"} for item in candidates]
    return winner["states"], winner["sse"], trace


def _load_phase(root: Path, phase: str, tokens: int, hidden: int = 4096):
    reference = torch.zeros((tokens, hidden), dtype=torch.float32)
    identity = torch.zeros_like(reference)
    gptq = torch.zeros_like(reference)
    rows, full_deltas, fc1_deltas, hashes = [], [], [], []
    for expert in range(288):
        path = root / phase / f"expert-{expert:03d}.safetensors"
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if (
                metadata.get("schema") != "glm53-p8-joint-route-contribution.v2"
                or metadata.get("ldlq") != "false"
                or int(metadata.get("expert", -1)) != expert
            ):
                raise RuntimeError(f"invalid V2 joint contribution: {path}")
            index = handle.get_tensor("token_rows").long()
            reference.index_add_(0, index, handle.get_tensor("reference").float())
            identity.index_add_(0, index, handle.get_tensor("identity_error").float())
            gptq.index_add_(0, index, handle.get_tensor("gptq_error").float())
            rows.append(index)
            full_deltas.append(handle.get_tensor("h128_delta").float())
            fc1_deltas.append(handle.get_tensor("fc1_h128_delta").float())
        hashes.append({"path": str(path), "sha256": sha256_file(path)})
    return reference, identity, gptq, rows, full_deltas, fc1_deltas, hashes


def _residual(base, rows, full_deltas, fc1_deltas, states):
    value = base.clone()
    for expert, state in enumerate(states.tolist()):
        if state == 1:
            value.index_add_(0, rows[expert], full_deltas[expert])
        elif state == 2:
            value.index_add_(0, rows[expert], fc1_deltas[expert])
    return value


def _window_nmse(reference, error, rows_per_window):
    result = []
    for start in range(0, len(reference), rows_per_window):
        stop = start + rows_per_window
        result.append(float(
            error[start:stop].double().square().sum()
            / reference[start:stop].double().square().sum().clamp_min(1e-30)
        ))
    return result


def _pack_two_bits(states: torch.Tensor) -> str:
    values = states.numpy().astype(np.uint8)
    packed = np.zeros((len(values) + 3) // 4, dtype=np.uint8)
    for index, value in enumerate(values):
        packed[index // 4] |= value << (2 * (index % 4))
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
        raise FileExistsError("refusing to overwrite V2 joint policy outputs")
    plan = json.loads(args.plan.read_text())
    if plan.get("schema") != "glm53-p8-joint-routed-sum-fc1-policy-plan.v2":
        raise RuntimeError("wrong joint policy plan")
    previous = json.loads(args.previous_policy.read_text())
    tokens = len(plan["sampling"]["selection"]) * plan["sampling"]["rows_per_window"]

    selection = _load_phase(args.contributions, "selection", tokens)
    sel_ref, sel_identity, sel_gptq, rows, full, fc1, selection_hashes = selection
    prior = torch.zeros(288, dtype=torch.uint8)
    prior[previous["h128_experts"]] = 1
    starts = [
        torch.zeros(288, dtype=torch.uint8),
        torch.ones(288, dtype=torch.uint8),
        torch.full((288,), 2, dtype=torch.uint8),
        prior,
    ]
    states, selection_sse, trace = coordinate_policy_three(
        sel_identity, rows, full, fc1, starts, plan["optimizer"]["maximum_passes"]
    )
    selection_errors = {
        "identity": sel_identity,
        "full_h128": _residual(sel_identity, rows, full, fc1, torch.ones(288, dtype=torch.uint8)),
        "fc1_h128": _residual(sel_identity, rows, full, fc1, torch.full((288,), 2, dtype=torch.uint8)),
        "joint": _residual(sel_identity, rows, full, fc1, states),
        "gptq": sel_gptq,
    }
    del selection, sel_gptq, rows, full, fc1

    validation = _load_phase(args.contributions, "validation", tokens)
    val_ref, val_identity, val_gptq, rows, full, fc1, validation_hashes = validation
    validation_errors = {
        "identity": val_identity,
        "full_h128": _residual(val_identity, rows, full, fc1, torch.ones(288, dtype=torch.uint8)),
        "fc1_h128": _residual(val_identity, rows, full, fc1, torch.full((288,), 2, dtype=torch.uint8)),
        "joint": _residual(val_identity, rows, full, fc1, states),
        "gptq": val_gptq,
    }
    rows_per_window = plan["sampling"]["rows_per_window"]
    selection_nmse = {name: _window_nmse(sel_ref, value, rows_per_window) for name, value in selection_errors.items()}
    validation_nmse = {name: _window_nmse(val_ref, value, rows_per_window) for name, value in validation_errors.items()}
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

    state_names = ["identity", "full-h128", "fc1-h128-identity-down"]
    counts = {name: int((states == index).sum()) for index, name in enumerate(state_names)}
    policy = {
        "schema": "glm53-p8-joint-routed-sum-fc1-policy.v2",
        "plan_sha256": sha256_file(args.plan),
        "phase": "selection",
        "expert_states": states.tolist(),
        "state_names": state_names,
        "state_counts": counts,
        "expert_states_hex_two_bit": _pack_two_bits(states),
        "expert_state_bytes": 72,
        "checkpoint_family_law_ids_bytes": 3,
        "policy_bytes": 75,
        "selection_sse": selection_sse,
        "optimizer_trace": trace,
        "protected_roles_opened": [],
        "ldlq_used": False,
    }
    args.policy_output.parent.mkdir(parents=True, exist_ok=True)
    args.policy_output.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")

    domain_rows = defaultdict(list)
    window_rows = []
    for item, candidate, control in zip(plan["sampling"]["validation"], joint, gptq, strict=True):
        row = {
            "window": item["id"], "domain": item["domain"],
            "joint_nmse": float(candidate), "gptq_nmse": float(control),
            "log_ratio": float(math.log(candidate / control)),
        }
        window_rows.append(row)
        domain_rows[item["domain"]].append(row)
    analysis = {
        "schema": "glm53-p8-joint-routed-sum-fc1-policy-analysis.v2",
        "plan_sha256": sha256_file(args.plan),
        "policy_sha256": sha256_file(args.policy_output),
        "state_counts": counts,
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
        "decision": "pass-freeze-for-unopened-kld" if passed else "fail-fc1-joint-route-policy",
        "decision_rule": plan["validation_rule"],
        "input_files": selection_hashes + validation_hashes,
        "protected_roles_opened": [],
        "confirmation_logits_opened": False,
        "ldlq_used": False,
    }
    args.analysis_output.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "decision": analysis["decision"], "state_counts": counts,
        "policy_sha256": analysis["policy_sha256"],
        "analysis_sha256": sha256_file(args.analysis_output),
        **analysis["validation_joint_vs_gptq"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
