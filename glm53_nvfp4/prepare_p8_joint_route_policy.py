"""Freeze a fit-only policy experiment on the weighted top-8 MoE sum."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


def _file(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _windows(
    roles_path: Path, offset: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    fit = json.loads(roles_path.read_text())["roles"]["fit"]
    by_domain: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in fit:
        by_domain[item["domain"]].append(item)
    if len(by_domain) != 4 or any(len(items) < offset + 4 for items in by_domain.values()):
        raise RuntimeError("joint policy has insufficient unused fit windows")
    selection, validation = [], []
    for domain in sorted(by_domain):
        items = sorted(by_domain[domain], key=lambda item: item["rank_sha256"])
        selection.extend(items[offset : offset + 2])
        validation.extend(items[offset + 2 : offset + 4])
    return selection, validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--previous-policy", type=Path, required=True)
    parser.add_argument("--domain-window-offset", type=int, default=0)
    parser.add_argument("--include-fc1-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    previous = json.loads(args.previous_policy.read_text())
    if previous.get("ldlq_used") is not False:
        raise RuntimeError("previous policy violates the no-LDLQ boundary")
    if args.domain_window_offset < 0:
        raise ValueError("domain window offset must be nonnegative")
    selection, validation = _windows(args.roles, args.domain_window_offset)
    payload = {
        "schema": (
            "glm53-p8-joint-routed-sum-fc1-policy-plan.v2"
            if args.include_fc1_only
            else "glm53-p8-joint-routed-sum-policy-plan.v1"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "layer": 3,
        "codec": "K4 procedural-MCG E4M3 UE8M0/K32, identity or scaled-H128 boundary",
        "physical_codec_bpw": 4.250443113070947,
        "policy_bytes": 75 if args.include_fc1_only else 39,
        "algorithm_exclusion": "LDLQ and BlockLDLQ are excluded",
        "objective": "minimum squared error of the route-weighted sum of all top-8 expert outputs, including co-routed cross-expert error terms",
        "arms": (
            ["identity", "full-h128", "fc1-h128-identity-down"]
            if args.include_fc1_only
            else ["identity", "full-h128"]
        ),
        "reference": "BF16 source weights through the same software E4M3/K32 input and post-SwiGLU carriers",
        "sampling": {
            "selection": selection,
            "validation": validation,
            "rows_per_window": 2048,
            "domains": 4,
            "windows_per_domain_per_phase": 2,
            "domain_window_offset": args.domain_window_offset,
        },
        "optimizer": {
            "method": "deterministic binary coordinate descent",
            "expert_order": list(range(288)),
            "maximum_passes": 8,
            "starts": (
                ["all-identity", "all-full-h128", "all-fc1-h128", "previous-39-byte-policy"]
                if args.include_fc1_only
                else ["all-identity", "all-h128", "previous-39-byte-policy"]
            ),
            "tie_break": "lowest selection SSE, then lexicographically smallest bit mask",
        },
        "validation_rule": "advance only if frozen policy improves joint routed-sum NMSE versus decoded GPTQ NVFP4 by at least 10 percent, improves at least 6 of 8 windows, and the paired BCa 95 percent upper bound on window log NMSE ratio is below zero",
        "next_kld_boundary": "a pass may be frozen for one previously unopened role with verified teacher logits; the already reused conditional-fit32 role cannot qualify it",
        "protected_boundary": "selection, confirmation, final, and all 28 confirmation logits remain unopened during policy fit and validation",
        "bootstrap": {"method": "paired BCa", "unit": "fit window", "replicates": 50000, "seed": 2026090515},
        "inputs": {
            "roles": _file(args.roles),
            "capture_manifest": _file(args.capture_root / "capture-manifest.json"),
            "source_index": _file(args.source_index),
            "previous_policy": _file(args.previous_policy),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
