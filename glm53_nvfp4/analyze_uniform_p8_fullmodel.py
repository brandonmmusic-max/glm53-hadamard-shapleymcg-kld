"""Analyze the preregistered uniform all-layer P8 conditional-fit KLD run."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .paired_role_analysis import bca_mean_interval, load_window_means
from .shard_index import sha256_file


def analyze(
    *,
    plan: dict[str, object],
    role_rows: list[dict[str, object]],
    candidate: dict[str, float],
    control: dict[str, float],
) -> dict[str, object]:
    expected = [str(row["id"]) for row in role_rows]
    if len(expected) != len(set(expected)):
        raise RuntimeError("role contains duplicate window ids")
    if set(candidate) != set(expected) or set(control) != set(expected):
        raise RuntimeError("candidate and control must contain the exact preregistered role")
    for label, values in (("candidate", candidate), ("control", control)):
        if any(not math.isfinite(values[key]) or values[key] < 0.0 for key in expected):
            raise RuntimeError(f"{label} contains a non-finite or negative KLD")

    candidate_values = np.asarray([candidate[key] for key in expected], dtype=np.float64)
    control_values = np.asarray([control[key] for key in expected], dtype=np.float64)
    delta = candidate_values - control_values
    bootstrap_spec = plan["bootstrap"]
    rng = np.random.default_rng(int(bootstrap_spec["seed"]))
    indexes = rng.integers(
        0,
        len(expected),
        size=(int(bootstrap_spec["replicates"]), len(expected)),
    )
    candidate_bootstrap = candidate_values[indexes].mean(axis=1)
    control_bootstrap = control_values[indexes].mean(axis=1)
    delta_bootstrap = delta[indexes].mean(axis=1)
    ratio_bootstrap = candidate_bootstrap / control_bootstrap

    candidate_mean = float(candidate_values.mean())
    control_mean = float(control_values.mean())
    mean_delta = float(delta.mean())
    delta_interval = bca_mean_interval(delta, delta_bootstrap)
    if delta_interval[1] < 0.0:
        direction = "candidate-lower-kld"
    elif delta_interval[0] > 0.0:
        direction = "candidate-higher-kld"
    else:
        direction = "contextual-comparison-inconclusive"

    domain_rows: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(role_rows):
        domain_rows[str(row["domain"])].append(index)
    domains: dict[str, object] = {}
    for domain, positions in sorted(domain_rows.items()):
        c = candidate_values[positions]
        b = control_values[positions]
        d = c - b
        domains[domain] = {
            "windows": len(positions),
            "candidate_mean_kld": float(c.mean()),
            "context_control_mean_kld": float(b.mean()),
            "mean_delta_kld": float(d.mean()),
            "relative_improvement": float(1.0 - c.mean() / b.mean()),
            "candidate_window_wins": int((d < 0.0).sum()),
        }

    return {
        "schema": "glm53-p8-uniform-all42-fullmodel-kld-analysis.v1",
        "status": "measurement-complete",
        "evidence_level": "opened-role developmental full-model measurement",
        "role": "conditional-fit",
        "experimental_unit": "window",
        "windows": len(expected),
        "candidate_mean_kld": candidate_mean,
        "candidate_mean_kld_ci95_bca": list(
            bca_mean_interval(candidate_values, candidate_bootstrap)
        ),
        "context_control_mean_kld": control_mean,
        "context_control_mean_kld_ci95_bca": list(
            bca_mean_interval(control_values, control_bootstrap)
        ),
        "mean_delta_kld": mean_delta,
        "relative_improvement": float(1.0 - candidate_mean / control_mean),
        "delta_ci95_bca": list(delta_interval),
        "delta_ci95_percentile": [
            float(value) for value in np.quantile(delta_bootstrap, (0.025, 0.975))
        ],
        "ratio_ci95_percentile": [
            float(value) for value in np.quantile(ratio_bootstrap, (0.025, 0.975))
        ],
        "candidate_window_wins": int((delta < 0.0).sum()),
        "directional_context": direction,
        "context_control_limitation": (
            "The decoded-GPTQ comparator is not equal-rate and does not transform the "
            "same all-layer set; this row cannot establish a strict product win."
        ),
        "bootstrap": {
            "replicates": int(bootstrap_spec["replicates"]),
            "seed": int(bootstrap_spec["seed"]),
            "unit": "window",
        },
        "domains": domains,
        "window_records": [
            {
                "window_id": key,
                "domain": str(row["domain"]),
                "candidate_mean_kld": float(candidate[key]),
                "context_control_mean_kld": float(control[key]),
                "delta_kld": float(candidate[key] - control[key]),
            }
            for key, row in zip(expected, role_rows, strict=True)
        ],
        "protected_roles_opened": [],
        "strict_quality_claim": False,
        "ldlq_used": False,
    }


def _record_hashes(run_dir: Path) -> dict[str, str]:
    return {path.stem: sha256_file(path) for path in sorted(run_dir.glob("*.parquet"))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--context-control-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    roles = json.loads(args.roles.read_text())
    if sha256_file(args.roles) != plan["role"]["sha256"]:
        raise RuntimeError("role manifest does not match the preregistered identity")
    role_rows = roles["roles"]["conditional-fit"]
    if len(role_rows) != plan["role"]["windows"]:
        raise RuntimeError("role window count does not match the preregistered plan")
    payload = analyze(
        plan=plan,
        role_rows=role_rows,
        candidate=load_window_means(args.candidate_run),
        control=load_window_means(args.context_control_run),
    )
    payload.update(
        {
            "plan_sha256": sha256_file(args.plan),
            "roles_sha256": sha256_file(args.roles),
            "analysis_code_sha256": sha256_file(Path(__file__)),
            "candidate_record_sha256": _record_hashes(args.candidate_run),
            "context_control_record_sha256": _record_hashes(args.context_control_run),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
