"""Execute the sealed matched stock/candidate conditional-fit32 pair."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


SCHEMA = "glm53-rotation-v6.blocklocal-h16-layer3-cf32-execution.v2"


def _verify_file(item: dict[str, object]) -> None:
    path = Path(str(item["path"]))
    if (
        not path.is_file()
        or path.stat().st_size != item["bytes"]
        or sha256_file(path) != item["sha256"]
    ):
        raise RuntimeError(f"sealed input drift: {path}")


def verify_plan(plan_path: Path) -> dict[str, object]:
    plan = json.loads(plan_path.read_text())
    if (
        plan.get("schema") != SCHEMA
        or plan.get("decision_before_result") is not True
        or plan.get("run_order") != ["stock", "candidate"]
        or plan.get("protected_roles_opened") != []
    ):
        raise RuntimeError("invalid execution plan")
    for item in plan["inputs"].values():
        _verify_file(item)
    if sha256_file(Path(plan["inputs"]["runner"]["path"])) != sha256_file(Path(__file__)):
        raise RuntimeError("the running executor is not the sealed executor")
    return plan


def _verify_run(path: Path, run_id: str, roles_sha: str) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if (
        payload.get("status") != "complete"
        or payload.get("run_id") != run_id
        or payload.get("role") != "conditional-fit"
        or payload.get("roles_sha256") != roles_sha
        or len(payload.get("windows", {})) != 32
        or payload.get("config_id")
        != "nvfp4-v5-identity-gate-up-humming-bf16-instanttensor-tp4-ep4-dcp4-eager-nomtp-kvfp8"
    ):
        raise RuntimeError(f"run did not close exactly: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = verify_plan(args.plan)
    outputs = plan["outputs"]
    receipt_path = Path(outputs["execution_receipt"])
    analysis_path = Path(outputs["paired_analysis"])
    if receipt_path.exists() or analysis_path.exists():
        raise FileExistsError("refusing to overwrite execution or analysis output")

    current_image = subprocess.check_output(
        ["docker", "image", "inspect", plan["runtime"]["image"], "--format", "{{.Id}}"],
        text=True,
    ).strip()
    if current_image != plan["runtime"]["image_id"]:
        raise RuntimeError("runtime image digest drift")

    run_root = Path(outputs["run_root"])
    roles_path = Path(plan["inputs"]["roles"]["path"])
    roles_sha = str(plan["inputs"]["roles"]["sha256"])
    run_kld = Path(plan["inputs"]["run_kld"]["path"])
    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(
        {
            "GLM53_REPO": str(repo),
            "GLM53_ROLES": str(roles_path),
            "GLM53_LOAD_FORMAT": "instanttensor",
            "GLM53_DCP_SIZE": "4",
            "GLM53_RUNTIME_IMAGE": str(plan["runtime"]["image"]),
            "GLM53_RUNTIME_IMAGE_ID": str(plan["runtime"]["image_id"]),
            "GLM53_RUNTIME_PATCH_MANIFEST": str(plan["inputs"]["runtime_manifest"]["path"]),
            "GLM53_MODEL_AUX_MOUNT_ROOT": str(plan["runtime"]["model_aux_mount_root"]),
        }
    )
    started = datetime.now(timezone.utc).isoformat()
    runs: dict[str, dict[str, object]] = {}
    for arm in plan["run_order"]:
        run_id = str(plan["run_ids"][arm])
        manifest = run_root / f"run-{run_id}.json"
        records = run_root / "records" / run_id
        if manifest.exists() or records.exists():
            raise FileExistsError(f"fresh-run boundary violated for {run_id}")
        subprocess.run(
            [
                "bash", str(run_kld), "conditional-fit", run_id,
                str(plan["models"][arm]), f"rotation-v6-{arm}", "identity", "3",
                "", "", "", "humming", "gate-up",
            ],
            check=True,
            cwd=repo,
            env=env,
        )
        closed = _verify_run(manifest, run_id, roles_sha)
        runs[arm] = {
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "records": str(records),
            "mean_kld": closed["mean_of_window_means"],
        }

    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable, "-m", "glm53_nvfp4.paired_role_analysis",
            "--candidate-run", str(run_root / "records" / plan["run_ids"]["candidate"]),
            "--stock-run", str(run_root / "records" / plan["run_ids"]["stock"]),
            "--roles", str(roles_path), "--role", "conditional-fit",
            "--output", str(analysis_path),
        ],
        check=True,
        cwd=repo,
    )
    analysis = json.loads(analysis_path.read_text())
    advance = analysis["mean_delta_kld"] < 0.0
    receipt = {
        "schema": "glm53-rotation-v6.blocklocal-h16-layer3-cf32-execution-receipt.v1",
        "status": "complete",
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "plan": str(args.plan.resolve()),
        "plan_sha256": sha256_file(args.plan),
        "runs": runs,
        "analysis": str(analysis_path),
        "analysis_sha256": sha256_file(analysis_path),
        "adaptive_decision": "advance" if advance else "do-not-advance",
        "protected_roles_opened": [],
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
