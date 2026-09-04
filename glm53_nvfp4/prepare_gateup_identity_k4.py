"""Freeze gate/up identity-K4 endpoint ridge tuning."""
from __future__ import annotations

import argparse, json
from datetime import datetime, timezone
from pathlib import Path
from .shard_index import sha256_file


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prior-analysis", type=Path, required=True)
    p.add_argument("--source-index", type=Path, required=True)
    p.add_argument("--capture-manifest", type=Path, required=True)
    p.add_argument("--roles", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    payload = {
        "schema": "glm53-gateup-identity-k4-tuning-plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "adaptive fit-only tuning; no teacher logits",
        "layer": 3,
        "experts": [2, 18, 34, 50, 66, 82, 98, 114],
        "sampling": {"strategy": "domain-balanced", "fit": {"offset": 0, "count": 256}, "evaluation": {"offset": 1664, "count": 128}},
        "candidate": {"scope": "unrotated gate_proj and up_proj concatenated with shared global scale", "ridge_ratio_grid": [1.0, 3.0, 10.0], "sweeps": 1, "stored_bpw": 4.5, "runtime": "direct native NVFP4 endpoint", "ldlq": False},
        "control": "same concatenated full-Hessian GPTQ NVFP4 endpoint",
        "measurement": "routed full-expert output NMSE with BF16 down_proj to isolate gate/up",
        "selection_rule": "select minimum geometric-mean NMSE; validate only if at least 3 percent better and at least 6/8 experts win",
        "protected_boundary": "teacher-logit roles remain unopened",
        "prior": {"path": str(a.prior_analysis), "sha256": sha256_file(a.prior_analysis)},
        "inputs": {"source_index": {"path": str(a.source_index), "sha256": sha256_file(a.source_index)}, "capture_manifest": {"path": str(a.capture_manifest), "sha256": sha256_file(a.capture_manifest)}, "roles": {"path": str(a.roles), "sha256": sha256_file(a.roles)}},
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(sha256_file(a.output))
if __name__ == "__main__": main()
