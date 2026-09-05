"""Close a V8 gate whose post-score wrapper was interrupted.

This recovery is deliberately narrow: all three immutable KLD manifests and
the separately produced analysis must already exist.  It never launches a
model, evaluates a window, edits a manifest, or recomputes KLD.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


def _file(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    execution = json.loads(args.execution.read_text())
    analysis = json.loads(args.analysis.read_text())
    if (
        execution.get("schema")
        != "glm53-rotation-v8.shared-mid-butterfly-gate-execution.v1"
        or execution.get("stage") != "tune32"
        or execution.get("run_order") != ["stock", "zero", "candidate"]
        or execution.get("protected_roles_opened") != []
        or analysis.get("schema")
        != "glm53-rotation-v8.shared-mid-butterfly-gate-analysis.v1"
        or analysis.get("execution_sha256") != sha256_file(args.execution)
        or analysis.get("protected_roles_opened") != []
    ):
        raise RuntimeError("inputs are not the frozen completed V8 tune gate")
    runs = {}
    for arm in execution["run_order"]:
        run_id = execution["run_ids"][arm]
        manifest_path = args.run_root / f"run-{run_id}.json"
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("status") != "complete"
            or manifest.get("role") != execution["runtime_role"]
            or len(manifest.get("windows", {})) != 32
        ):
            raise RuntimeError(f"incomplete KLD manifest for {arm}")
        session = args.run_root / "sessions" / run_id
        evidence = {
            name: _file(session / name)
            for name in (
                "server-ready.log",
                "server-final.log",
                "role-eval.log",
            )
        }
        if arm != "stock":
            evidence["rotation-forward.log"] = _file(session / "rotation-forward.log")
            evidence["rotation-runtime-proof.json"] = _file(
                session / "rotation-runtime-proof.json"
            )
            proof = json.loads((session / "rotation-runtime-proof.json").read_text())
            if proof.get("status") != "pass" or proof.get("mid_forward_pairs") != 4:
                raise RuntimeError(f"rotation runtime proof failed for {arm}")
        runs[arm] = {
            "manifest": _file(manifest_path),
            "mean_kld": manifest["mean_of_window_means"],
            "session_evidence": evidence,
        }
    candidate_session = args.run_root / "sessions" / execution["run_ids"]["candidate"]
    if sha256_file(candidate_session / "server-final.log") != sha256_file(
        candidate_session / "rotation-forward.log"
    ):
        raise RuntimeError("recovered candidate runtime log is not a bitwise copy")
    receipt = {
        "schema": "glm53-rotation-v8.shared-mid-butterfly-gate-execution-receipt.v1",
        "status": "complete-with-post-score-wrapper-recovery",
        "stage": execution["stage"],
        "recovered_at": datetime.now(timezone.utc).isoformat(),
        "execution": _file(args.execution),
        "runs": runs,
        "analysis": _file(args.analysis),
        "decision": analysis["decision"],
        "recovery": {
            "reason": "the active run_kld_v3.sh file was edited while its long-lived shell was executing; after all candidate KLD rows and the complete manifest were written, the shifted lazy-read source resumed at --config-id and exited 127",
            "data_action": "no KLD row or manifest was rerun, replaced, or edited",
            "runtime_proof_action": "copied candidate server-final.log byte-for-byte to the expected rotation-forward.log name and ran the existing verifier against that same log",
            "analysis_action": "ran the frozen analyzer once over the three already complete manifests",
        },
        "protected_roles_opened": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
