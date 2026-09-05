from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from glm53_nvfp4.prepare_blocklocal_h16_cf32_execution import build
from glm53_nvfp4.run_blocklocal_h16_cf32 import verify_plan


def _write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) if not isinstance(payload, str) else payload)
    return path


def _args(tmp_path: Path) -> argparse.Namespace:
    stock = tmp_path / "stock"
    candidate = tmp_path / "candidate"
    stock.mkdir()
    candidate.mkdir()
    for root in (stock, candidate):
        _write(root / "model.safetensors.index.json", {})
        _write(root / "config.json", {})
    receipt = {
        "bf16_layers": [3], "redirected_tensors": 864,
        "carrier": str(stock.resolve()), "required_load_format": "instanttensor",
        "chunks": [{"path": str((tmp_path / "chunk.safetensors").resolve())}],
    }
    receipt_path = _write(candidate / "BF16_LAYER_RECEIPT.json", receipt)
    full = {
        "schema": "glm53-rotation-v6.blocklocal-h16-layer3-full-build.v1",
        "status": "pass", "experts": 288, "protected_roles_opened": [],
        "candidate_receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    }
    roles = {"roles": {"conditional-fit": [{"id": str(i)} for i in range(32)],
                       "fit": [], "selection": [], "confirmation": [], "final": []}}
    files = {name: _write(tmp_path / name, name) for name in
             ("hypothesis", "runtime", "runner", "run_kld", "role_eval", "analysis_code")}
    return argparse.Namespace(
        hypothesis_plan=files["hypothesis"], full_build=_write(tmp_path / "full.json", full),
        candidate=candidate, stock=stock, roles=_write(tmp_path / "roles.json", roles),
        runtime_manifest=files["runtime"], runtime_image="image:test", image_id="sha256:abc",
        model_aux_mount_root=tmp_path,
        runner=files["runner"], run_kld=files["run_kld"], role_eval=files["role_eval"],
        paired_analysis_code=files["analysis_code"], run_root=tmp_path / "runs",
        execution_receipt=tmp_path / "execution.json", paired_analysis=tmp_path / "paired.json",
    )


def test_freezer_closes_all_expert_overlay(tmp_path):
    payload = build(_args(tmp_path))
    assert payload["run_order"] == ["stock", "candidate"]
    assert payload["runtime"]["mtp"] == "disabled"
    assert payload["protected_roles_opened"] == []


def test_freezer_rejects_incomplete_expert_coverage(tmp_path):
    args = _args(tmp_path)
    full = json.loads(args.full_build.read_text())
    full["experts"] = 287
    args.full_build.write_text(json.dumps(full))
    with pytest.raises(RuntimeError, match="not closed"):
        build(args)


def test_executor_rejects_sealed_input_drift(tmp_path):
    args = _args(tmp_path)
    payload = build(args)
    payload["inputs"]["runner"] = {
        "path": str(Path(__import__("glm53_nvfp4.run_blocklocal_h16_cf32", fromlist=["x"]).__file__)),
        "bytes": Path(__import__("glm53_nvfp4.run_blocklocal_h16_cf32", fromlist=["x"]).__file__).stat().st_size,
        "sha256": hashlib.sha256(Path(__import__("glm53_nvfp4.run_blocklocal_h16_cf32", fromlist=["x"]).__file__).read_bytes()).hexdigest(),
    }
    plan = _write(tmp_path / "plan.json", payload)
    args.roles.write_text("drift")
    with pytest.raises(RuntimeError, match="sealed input drift"):
        verify_plan(plan)
