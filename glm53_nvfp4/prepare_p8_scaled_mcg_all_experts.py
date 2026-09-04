"""Freeze all-288 expert attribution for the scaled-H128 MCG codec."""
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
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--failed-kld", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    failed = json.loads(args.failed_kld.read_text())
    if failed.get("decision") != "fail-pseudoquant-kld" or failed.get("ldlq_used"):
        raise RuntimeError("all-expert attribution requires the frozen no-LDLQ KLD failure")
    root = Path(__file__).resolve().parents[1]
    payload = {
        "schema": "glm53-p8-scaled-h128-mcg-all-experts-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "layer": 3,
        "experts": list(range(288)),
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
            "encoder": "Viterbi plus full-Hessian GPTQ-style inter-group feedback, static in-group activation order, and two scale refits",
            "boundary": "uncoupled H128 with balance-0.5 positive diagonal",
            "physical_bpw": 4.251302083333333,
            "ldlq": False,
        },
        "control": {
            "weights": "full-Hessian GPTQ NVFP4 E2M1 group16",
            "physical_bpw": 4.5,
            "activation_carrier": "same dynamic E4M3 K32 carrier",
        },
        "estimand": "paired per-expert log routed-output NMSE ratio on disjoint fit routes",
        "bootstrap": {"method": "paired BCa", "unit": "expert", "replicates": 50000, "seed": 2026090512},
        "policy_preregistration": {
            "h128_eligible": "candidate full-output NMSE at least 5 percent below GPTQ control on evaluation routes",
            "fallback": "identity procedural-MCG P8, to be built and evaluated separately before a hybrid KLD run",
            "metadata": "one bit per expert plus three checkpoint-family law identifiers; total must be <=4096 bytes",
            "no_kld_reuse_claim": "the already opened conditional-fit32 panel may be used only for adaptive tuning and cannot qualify the redesigned policy",
        },
        "decision_rule": "report all 288 expert effects and bootstrap; advance to identity-P8 fallback build if at least 144 experts are H128-eligible and the BCa upper bound on the all-expert mean log ratio is below zero",
        "protected_boundary": "selection, confirmation, final, and all 28 confirmation logits remain unopened",
        "isa_cost": "P8 mxf8f6f4 is twice NVFP4 MMA issue count; P8 is quality, P4 is speed",
        "inputs": {
            "failed_kld": _file(args.failed_kld),
            "source_index": _file(args.source_index),
            "capture_manifest": _file(args.capture_root / "capture-manifest.json"),
            "roles": _file(args.roles),
        },
        "code": {
            "screen": _file(root / "glm53_nvfp4/screen_p8_scaled_mcg.py"),
            "transform": _file(root / "glm53_nvfp4/p8_h128.py"),
            "codec": _file(root / "glm53_nvfp4/trellis_mxf.py"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
