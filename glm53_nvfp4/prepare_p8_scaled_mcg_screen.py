"""Freeze scaled-H128 plus procedural-MCG K4 Viterbi expert screen."""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


EXPERTS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 71]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--replication-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    prior = json.loads(args.replication_result.read_text())
    if prior["decision"] != "pass-advance-to-mcg-codec-screen" or prior["selected_arm"] != "balance-0.5":
        raise ValueError("MCG screen requires the frozen balance-0.5 activation pass")
    repo = Path(__file__).resolve().parents[1]
    logical_weights = 2 * 2048 * 4096 + 4096 * 2048
    rotation_bpw = 16.0 * 2048 / logical_weights
    payload = {
        "schema": "glm53-p8-scaled-h128-mcg-screen-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "fit-developmental-new-disjoint-slices",
        "layer": 3,
        "experts": EXPERTS,
        "sampling": {
            "calibration": {"role": "fit", "strategy": "domain-balanced", "offset": 320, "count": 128},
            "evaluation": {"role": "fit", "strategy": "domain-balanced", "offset": 448, "count": 128},
        },
        "candidate": {
            "product": "P8",
            "rate": "K4",
            "alphabet": "E4M3",
            "block_size": 32,
            "scale": "UE8M0 per 32 weights",
            "law": "procedural MCG alpha 2.0",
            "encoder": "native-tile Viterbi with full-Hessian GPTQ-style inter-group feedback and static in-group activation order",
            "boundary": "uncoupled H128 with balance-0.5 positive diagonal learned from REAP calibration routes",
            "diagonal": "d=exp(0.5*(log down-column RMS-log post-SwiGLU RMS)), centered per 128-block and clamped [0.25,4]",
            "physical_bpw": 4.25 + rotation_bpw,
            "rotation_metadata_bpw": rotation_bpw,
            "runtime_table_bytes": 0,
            "ldlq": False,
        },
        "control": {
            "weights": "GPTQ NVFP4 E2M1 group16 with E4M3/16 secondary scales",
            "physical_bpw": 4.5,
            "screen_carrier": "same E4M3 K32 activation carrier to isolate weight codec plus selected H128 boundary",
        },
        "metrics_order": ["effective per-projection weight NMSE", "routed full-expert output NMSE"],
        "estimand": "paired per-expert log full-output NMSE ratio, scaled-H128 MCG P8 versus GPTQ NVFP4 under the common E4M3 carrier",
        "decision_rule": "advance to a layer-3 pseudoquant end-to-end KLD overlay only if candidate physical bpw is no greater than control, geometric-mean full-expert output NMSE improves at least 10 percent, at least 12 of 16 experts improve, the paired 95 percent BCa upper bound on mean log ratio is below zero, codec decode is bit exact, and all three per-projection effective weight NMSE values are reported",
        "bootstrap": {"method": "BCa", "unit": "expert", "replicates": 50000, "seed": 2026090506},
        "interpretation_boundary": "a pass qualifies a layer-3 pseudoquant KLD build only; P8 device closure and performance remain blocked until the kernel consumes these exact rotations and payloads",
        "isa_cost": "P8 mxf8f6f4 issues twice the MMA instructions of NVFP4; P8 is a quality product, not the NVFP4 speed product",
        "protected_boundary": "selection, confirmation, final, and all 28 confirmation logits remain unopened",
        "inputs": {
            "activation_replication": {"path": str(args.replication_result), "sha256": sha256_file(args.replication_result)},
            "source_index": {"path": str(args.source_index), "sha256": sha256_file(args.source_index)},
            "capture_manifest": {"path": str(args.capture_root / "capture-manifest.json"), "sha256": sha256_file(args.capture_root / "capture-manifest.json")},
            "roles": {"path": str(args.roles), "sha256": sha256_file(args.roles)},
        },
        "analysis_lock": {
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            "screen_code_sha256": sha256_file(repo / "glm53_nvfp4/screen_p8_scaled_mcg.py"),
            "analysis_code_sha256": sha256_file(repo / "glm53_nvfp4/analyze_p8_scaled_mcg.py"),
            "transform_code_sha256": sha256_file(repo / "glm53_nvfp4/p8_h128.py"),
            "codec_code_sha256": sha256_file(repo / "glm53_nvfp4/trellis_mxf.py"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
