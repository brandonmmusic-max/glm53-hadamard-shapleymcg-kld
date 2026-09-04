"""Freeze the full layer-3 scaled-H128 MCG build and pseudoquant KLD gate."""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


def _file(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-analysis", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--conditional-roles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    screen = json.loads(args.screen_analysis.read_text())
    if screen.get("decision") != "pass-build-layer3-pseudoquant-kld-overlay":
        raise RuntimeError("full build requires the frozen scaled-H128 MCG screen pass")
    conditional = json.loads(args.conditional_roles.read_text())
    rows = conditional["roles"]["conditional-fit"]
    if len(rows) != 32 or any(conditional["roles"][role] for role in ("fit", "selection", "confirmation", "final")):
        raise RuntimeError("conditional role file must expose exactly 32 conditional-fit windows and nothing else")
    root = Path(__file__).resolve().parents[1]
    candidate_bpw = 4.25 + (16.0 * 2048) / (2 * 2048 * 4096 + 4096 * 2048)
    payload = {
        "schema": "glm53-p8-scaled-h128-mcg-layer3-kld-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "layer": 3,
        "experts": 288,
        "hidden_size": 4096,
        "intermediate_size": 2048,
        "calibration": {"role": "fit", "strategy": "domain-balanced", "offset": 320, "count": 128},
        "candidate": "K4 procedural-MCG alpha2, E4M3, UE8M0/32, H128 balance-0.5",
        "candidate_physical_bpw": candidate_bpw,
        "encoder": "Viterbi plus full-Hessian GPTQ-style inter-group output-error feedback with static native-group activation order and two joint scale-refit rounds",
        "algorithm_exclusion": "LDLQ and BlockLDLQ are excluded",
        "build_ranges": [[0, 72], [72, 144], [144, 216], [216, 288]],
        "pseudoquant_execution": "decoded BF16 weights with exact E4M3 K32 input/post-SwiGLU carriers and per-expert H128 balance boundary",
        "control": "decoded full-Hessian GPTQ NVFP4 at 4.5 bpw, same BF16 loader and same E4M3 K32 activation carrier, identity boundary",
        "primary_estimand": "paired equal-window mean KLD(candidate)-KLD(control) on conditional-fit n=32",
        "decision_rule": "pass gate 4a only if mean delta is at most -0.0014 nats and the paired BCa 95 percent upper bound is below zero",
        "bootstrap": {"method": "paired BCa", "unit": "window", "replicates": 50000, "seed": 2026090511},
        "stopping_rule": "one run per arm on the exact 32 windows; no exclusions, substitutions, or rerolls",
        "protected_boundary": "selection, confirmation, final, and all 28 confirmation logits remain unopened",
        "isa_cost": "P8 mxf8f6f4 issues twice the MMA instructions of NVFP4; P8 is a quality product and P4 is the speed product",
        "runtime_gate": "this plan can pass only pseudoquant gate 4a; device closure, speed, and determinism remain separate later gates",
        "inputs": {
            "screen_analysis": _file(args.screen_analysis),
            "source_index": _file(args.source_index),
            "capture_manifest": _file(args.capture_root / "capture-manifest.json"),
            "roles": _file(args.roles),
            "conditional_roles": _file(args.conditional_roles),
        },
        "code": {
            "builder": _file(root / "glm53_nvfp4/quantize_p8_scaled_mcg_layer.py"),
            "transform": _file(root / "glm53_nvfp4/p8_h128.py"),
            "codec": _file(root / "glm53_nvfp4/trellis_mxf.py"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
