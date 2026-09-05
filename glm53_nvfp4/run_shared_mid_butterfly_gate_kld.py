"""Execute a sealed V8 three-arm tune/conditional-fit gate."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


def _verify(item: dict) -> None:
    path = Path(item["path"])
    if not path.is_file() or path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
        raise RuntimeError(f"sealed input drift: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution", type=Path, required=True)
    args = parser.parse_args()
    execution = json.loads(args.execution.read_text())
    if (
        execution.get("schema") != "glm53-rotation-v8.shared-mid-butterfly-gate-execution.v1"
        or execution.get("decision_before_result") is not True
        or execution.get("protected_roles_opened") != []
        or execution.get("run_order") != ["stock", "zero", "candidate"]
    ):
        raise RuntimeError("invalid gate execution")
    for item in execution["inputs"].values():
        _verify(item)
    for group in ("rotations", "physical_builds"):
        for item in execution[group].values():
            _verify(item)
    if sha256_file(Path(execution["inputs"]["runner"]["path"])) != sha256_file(Path(__file__)):
        raise RuntimeError("runner differs from sealed input")
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", execution["runtime"]["image"], "--format", "{{.Id}}"],
        text=True,
    ).strip()
    if image_id != execution["runtime"]["image_id"]:
        raise RuntimeError("runtime image drift")
    run_root = Path(execution["outputs"]["run_root"])
    analysis_path = Path(execution["outputs"]["analysis"])
    receipt_path = Path(execution["outputs"]["execution_receipt"])
    if analysis_path.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite gate outputs")
    for run_id in execution["run_ids"].values():
        if (run_root / f"run-{run_id}.json").exists() or (run_root / "records" / run_id).exists():
            raise FileExistsError(f"fresh-run boundary violated: {run_id}")
    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(
        {
            "GLM53_REPO": str(repo),
            "GLM53_ROLES": execution["inputs"]["roles"]["path"],
            "GLM53_LOAD_FORMAT": "instanttensor",
            "GLM53_DCP_SIZE": "4",
            "GLM53_RUNTIME_IMAGE": execution["runtime"]["image"],
            "GLM53_RUNTIME_IMAGE_ID": execution["runtime"]["image_id"],
            "GLM53_RUNTIME_PATCH_MANIFEST": execution["inputs"]["runtime_manifest"]["path"],
            "GLM53_MODEL_AUX_MOUNT_ROOT": execution["runtime"]["model_aux_mount_root"],
            "GLM53_ROTATION_PLACEMENT": "humming-inner",
        }
    )
    started = datetime.now(timezone.utc).isoformat()
    runs = {}
    for arm in execution["run_order"]:
        learned = arm != "stock"
        rotation = execution["rotations"][arm]["path"] if learned else ""
        run_id = execution["run_ids"][arm]
        subprocess.run(
            [
                "bash", execution["inputs"]["run_kld"]["path"],
                execution["runtime_role"], run_id, execution["models"][arm],
                f"rotation-v8-{execution['selected_arm']}-{arm}-{execution['stage']}",
                "learned" if learned else "identity", "3", rotation,
                "", "", "humming", "mid-only" if learned else "gate-up",
            ],
            cwd=repo,
            env=env,
            check=True,
        )
        manifest_path = run_root / f"run-{run_id}.json"
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("status") != "complete"
            or manifest.get("role") != execution["runtime_role"]
            or len(manifest.get("windows", {})) != 32
        ):
            raise RuntimeError(f"run did not close: {run_id}")
        runs[arm] = {
            "manifest": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "mean_kld": manifest["mean_of_window_means"],
        }
    subprocess.run(
        [
            execution["inputs"]["analysis_python"]["path"],
            "-m", "glm53_nvfp4.analyze_shared_mid_butterfly_gate",
            "--execution", str(args.execution),
            "--roles", execution["inputs"]["roles"]["path"],
            "--run-root", str(run_root),
            "--output", str(analysis_path),
        ],
        cwd=repo,
        check=True,
    )
    analysis = json.loads(analysis_path.read_text())
    receipt = {
        "schema": "glm53-rotation-v8.shared-mid-butterfly-gate-execution-receipt.v1",
        "status": "complete",
        "stage": execution["stage"],
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "execution": str(args.execution.resolve()),
        "execution_sha256": sha256_file(args.execution),
        "runs": runs,
        "analysis": str(analysis_path),
        "analysis_sha256": sha256_file(analysis_path),
        "decision": analysis["decision"],
        "protected_roles_opened": [],
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
