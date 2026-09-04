"""Freeze the identity-MCG P8 fallback build authorized by all-expert attribution."""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


def _file(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attribution", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    attribution = json.loads(args.attribution.read_text())
    if attribution.get("decision") != "pass-build-identity-p8-fallback":
        raise RuntimeError("identity fallback was not authorized by the all-expert gate")
    root = Path(__file__).resolve().parents[1]
    payload = {
        "schema": "glm53-p8-identity-mcg-layer3-build-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "layer": 3,
        "experts": 288,
        "build_ranges": [[0, 72], [72, 144], [144, 216], [216, 288]],
        "calibration": {"role": "fit", "strategy": "domain-balanced", "offset": 320, "count": 128},
        "candidate": "K4 procedural-MCG alpha2, E4M3, UE8M0/32, identity boundary",
        "physical_bpw": 4.25,
        "encoder": "Viterbi plus full-Hessian GPTQ-style inter-group output-error feedback with static in-group activation order and two scale refits",
        "algorithm_exclusion": "LDLQ and BlockLDLQ are excluded",
        "stopping_rule": "build every expert exactly once; fail on any non-bit-exact reference decode",
        "next_gate": "evaluate identity and scaled-H128 decoded payloads on a new disjoint fit slice, then freeze one expert bit from the preregistered policy",
        "protected_boundary": "selection, confirmation, final, and all 28 confirmation logits remain unopened",
        "inputs": {
            "attribution": _file(args.attribution),
            "source_index": _file(args.source_index),
            "capture_manifest": _file(args.capture_root / "capture-manifest.json"),
            "roles": _file(args.roles),
        },
        "code": {
            "builder": _file(root / "glm53_nvfp4/quantize_p8_identity_mcg_layer.py"),
            "codec": _file(root / "glm53_nvfp4/trellis_mxf.py"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
