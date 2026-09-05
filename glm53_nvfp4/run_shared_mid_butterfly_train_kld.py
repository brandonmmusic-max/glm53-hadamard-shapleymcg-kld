"""Execute the sealed V8 stock plus nine-arm train32 KLD grid."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


def _verify_file(item: dict) -> None:
    path = Path(item["path"])
    if (
        not path.is_file()
        or path.stat().st_size != item["bytes"]
        or sha256_file(path) != item["sha256"]
    ):
        raise RuntimeError(f"sealed input drift: {path}")


def _verify_run(path: Path, run_id: str, roles_sha: str, learned: bool) -> dict:
    payload = json.loads(path.read_text())
    scope = "learned-mid-only" if learned else "identity-gate-up"
    expected_config = f"nvfp4-v5-{scope}-humming-bf16-instanttensor-tp4-ep4-dcp4-eager-nomtp-kvfp8"
    if (
        payload.get("status") != "complete"
        or payload.get("run_id") != run_id
        or payload.get("role") != "fit"
        or payload.get("roles_sha256") != roles_sha
        or len(payload.get("windows", {})) != 32
        or payload.get("config_id") != expected_config
    ):
        raise RuntimeError(f"run did not close exactly: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution", type=Path, required=True)
    args = parser.parse_args()
    execution = json.loads(args.execution.read_text())
    if (
        execution.get("schema") != "glm53-rotation-v8.shared-mid-butterfly-train32-execution.v1"
        or execution.get("decision_before_result") is not True
        or execution.get("protected_roles_opened") != []
    ):
        raise RuntimeError("invalid V8 execution plan")
    for item in execution["inputs"].values():
        _verify_file(item)
    for candidate in execution["candidates"].values():
        for key in ("rotation", "build", "config", "index"):
            _verify_file(candidate[key])
    if sha256_file(Path(execution["inputs"]["runner"]["path"])) != sha256_file(Path(__file__)):
        raise RuntimeError("running executor is not the sealed executor")
    current_image = subprocess.check_output(
        ["docker", "image", "inspect", execution["runtime"]["image"], "--format", "{{.Id}}"],
        text=True,
    ).strip()
    if current_image != execution["runtime"]["image_id"]:
        raise RuntimeError("runtime image digest drift")
    output = execution["outputs"]
    receipt_path = Path(output["execution_receipt"])
    analysis_path = Path(output["analysis"])
    if receipt_path.exists() or analysis_path.exists():
        raise FileExistsError("refusing to overwrite V8 execution outputs")
    run_root = Path(output["run_root"])
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
    closed_runs = {}
    run_kld = execution["inputs"]["run_kld"]["path"]
    roles_sha = execution["inputs"]["roles"]["sha256"]
    for arm in execution["run_order"]:
        learned = arm != "stock"
        run_id = execution["run_ids"][arm]
        rotation = execution["candidates"][arm]["rotation"]["path"] if learned else ""
        subprocess.run(
            [
                "bash", run_kld, "fit", run_id, execution["models"][arm],
                f"rotation-v8-shared-mid-butterfly-l3-{arm}",
                "learned" if learned else "identity", "3", rotation,
                "", "", "humming", "mid-only" if learned else "gate-up",
            ],
            cwd=repo,
            env=env,
            check=True,
        )
        manifest_path = run_root / f"run-{run_id}.json"
        manifest = _verify_run(manifest_path, run_id, roles_sha, learned)
        closed_runs[arm] = {
            "manifest": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "mean_kld": manifest["mean_of_window_means"],
        }

    subprocess.run(
        [
            execution["inputs"]["analysis_python"]["path"],
            "-m", "glm53_nvfp4.analyze_shared_mid_butterfly_train",
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
        "schema": "glm53-rotation-v8.shared-mid-butterfly-train32-execution-receipt.v1",
        "status": "complete",
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "execution": str(args.execution.resolve()),
        "execution_sha256": sha256_file(args.execution),
        "runs": closed_runs,
        "analysis": str(analysis_path),
        "analysis_sha256": sha256_file(analysis_path),
        "selected_arm": analysis["selected_arm"],
        "adaptive_decision": analysis["adaptive_decision"],
        "protected_roles_opened": [],
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
