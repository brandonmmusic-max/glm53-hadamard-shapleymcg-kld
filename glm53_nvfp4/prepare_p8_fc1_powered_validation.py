"""Freeze powered fit validation for the selected three-state P8 policy."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


def _file(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--pilot-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    policy = json.loads(args.policy.read_text())
    pilot = json.loads(args.pilot_analysis.read_text())
    if (
        policy.get("schema") != "glm53-p8-joint-routed-sum-fc1-policy.v2"
        or pilot.get("decision") != "fail-fc1-joint-route-policy"
        or policy.get("ldlq_used") is not False
    ):
        raise RuntimeError("powered validation inputs do not match the frozen FC1 policy")
    fit = json.loads(args.roles.read_text())["roles"]["fit"]
    by_domain = defaultdict(list)
    for item in fit:
        by_domain[item["domain"]].append(item)
    windows = []
    for domain in sorted(by_domain):
        items = sorted(by_domain[domain], key=lambda item: item["rank_sha256"])
        if len(items) != 16:
            raise RuntimeError(f"expected 16 fit windows for {domain}, got {len(items)}")
        windows.extend(items[8:16])
    payload = {
        "schema": "glm53-p8-fc1-three-state-powered-validation-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "layer": 3,
        "status": "frozen-policy-powered-fit-validation",
        "policy_sha256": sha256_file(args.policy),
        "policy_state_counts": policy["state_counts"],
        "physical_codec_bpw": 4.250443113070947,
        "policy_bytes": 75,
        "algorithm_exclusion": "LDLQ and BlockLDLQ are excluded",
        "windows": windows,
        "rows_per_window": 2048,
        "objective": "route-weighted top-8 MoE sum NMSE with co-routed cross-expert terms",
        "decision_rule": "advance the unchanged policy to one unopened KLD role only if joint NMSE improves at least 10 percent versus decoded GPTQ NVFP4, at least 24 of 32 windows improve, and the paired BCa 95 percent upper bound on window log NMSE ratio is below zero",
        "stopping_rule": "one evaluation of the frozen policy on all 32 remaining fit windows; no exclusions, substitutions, policy changes, or rerolls",
        "protected_boundary": "selection, confirmation, final, and all 28 confirmation logits remain unopened",
        "bootstrap": {"method": "paired BCa", "unit": "fit window", "replicates": 50000, "seed": 2026090516},
        "inputs": {
            "roles": _file(args.roles),
            "capture_manifest": _file(args.capture_root / "capture-manifest.json"),
            "policy": _file(args.policy),
            "pilot_analysis": _file(args.pilot_analysis),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
