import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest

from glm53_nvfp4 import p8_coupled_three_layer_runtime as runtime


def _preparer():
    path = Path(__file__).parents[1] / "scripts/prepare_p8_coupled_three_layer_cf32_runtime.py"
    spec = importlib.util.spec_from_file_location("p8_three_layer_preparer", path)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _safe(path: Path, metadata: dict[str, str], payload: bytes = b"x") -> tuple[int, str]:
    header = json.dumps({"__metadata__": metadata}, separators=(",", ":")).encode()
    pad = (-len(header)) % 8
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(header) + pad) + header + b" " * pad + payload)
    return path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest()


def _coupled_tree(tmp_path: Path):
    design, transform = tmp_path / "design.json", tmp_path / "transform.json"
    design.write_bytes(b"design")
    transform.write_bytes(b"transform")
    old_d, old_t = runtime.COUPLED_DESIGN_SHA256, runtime.TRANSFORM_SHA256
    runtime.COUPLED_DESIGN_SHA256 = hashlib.sha256(b"design").hexdigest()
    runtime.TRANSFORM_SHA256 = hashlib.sha256(b"transform").hexdigest()
    postwrites, loaders = {}, {}
    old_s = runtime.EXL3_SCALE_SOURCE_SHA256
    runtime.EXL3_SCALE_SOURCE_SHA256 = "scale-source"
    for layer in runtime.LAYERS:
        rows = []
        for rank in runtime.RANKS:
            path = tmp_path / "sidecars" / f"layer-{layer:03d}" / f"p8-layer-{layer:03d}-tp4-rank-{rank}.safetensors"
            size, digest = _safe(path, {"schema": "glm53-p8-coupled-h512-h128-tp4-rank.v1",
                "boundary": "coupled-h512-h128-suh-svh-v1", "full_coupled": "true",
                "layer": str(layer), "rank": str(rank), "world_size": "4",
                "source_design_sha256": runtime.COUPLED_DESIGN_SHA256,
                "encoder_transform_sha256": runtime.TRANSFORM_SHA256})
            rows.append({"rank": rank, "path": str(path), "bytes": size, "sha256": digest, "source_exact": True})
        receipt = {"schema": "glm53-p8-coupled-tp4-postwrite-closure.v1", "status": "pass",
            "layer": layer, "world_size": 4, "source_design_sha256": runtime.COUPLED_DESIGN_SHA256,
            "evidence_level": "postwrite-source-exact-structural-closure",
            "chunk_retirement_gate_closed": "postwrite-all-tensor-source-closure",
            "exl3_scale_source_sha256": runtime.EXL3_SCALE_SOURCE_SHA256, "ranks": rows}
        p = tmp_path / "receipts" / f"layer-{layer:03d}-postwrite-abi-v2.json"
        p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(receipt))
        postwrites[layer] = p
        loader = {"decision": "pass", "loader_abi_gate_closed": True,
            "identities": {"design": runtime.COUPLED_DESIGN_SHA256, "transform": runtime.TRANSFORM_SHA256,
                           "runtime_manifest": runtime.V9_MANIFEST_SHA256, "postwrite": runtime.sha(p)},
            "protocol": {"schema": "glm53.p8-full-coupled-real-sidecar-loader-closure.v1",
                         "geometry": {"layer": layer, "ranks": list(runtime.RANKS), "world_size": 4, "fc1_tile_n": 128}},
            "ranks": [{"layer": layer, "rank": rank, "mode": "full-coupled", "identity_fallback": False}
                      for rank in runtime.RANKS]}
        lp = tmp_path / "receipts" / f"layer-{layer:03d}-loader.json"; lp.write_text(json.dumps(loader)); loaders[layer] = lp
    return design, transform, postwrites, loaders, old_d, old_t, old_s


def test_coupled_sidecars_all_layers_and_ranks(tmp_path):
    design, transform, postwrites, loaders, old_d, old_t, old_s = _coupled_tree(tmp_path)
    try:
        rows = runtime.validate_coupled_sidecars(tmp_path, design, transform, postwrites, loaders)
        assert {(r["layer"], r["rank"]) for r in rows} == set(__import__("itertools").product(runtime.LAYERS, runtime.RANKS))
    finally:
        runtime.COUPLED_DESIGN_SHA256, runtime.TRANSFORM_SHA256, runtime.EXL3_SCALE_SOURCE_SHA256 = old_d, old_t, old_s


def test_coupled_missing_rank_fails(tmp_path):
    design, transform, postwrites, loaders, old_d, old_t, old_s = _coupled_tree(tmp_path)
    try:
        receipt = tmp_path / "receipts/layer-020-postwrite-abi-v2.json"
        value = json.loads(receipt.read_text()); value["ranks"].pop(); receipt.write_text(json.dumps(value))
        with pytest.raises(ValueError, match="missing or duplicates"):
            runtime.validate_coupled_sidecars(tmp_path, design, transform, postwrites, loaders)
    finally:
        runtime.COUPLED_DESIGN_SHA256, runtime.TRANSFORM_SHA256, runtime.EXL3_SCALE_SOURCE_SHA256 = old_d, old_t, old_s


def test_coupled_identity_fallback_fails(tmp_path):
    design, transform, postwrites, loaders, old_d, old_t, old_s = _coupled_tree(tmp_path)
    try:
        target = tmp_path / "sidecars/layer-022/p8-layer-022-tp4-rank-2.safetensors"
        size, digest = _safe(target, {"schema": "glm53-p8-mcg-tp4-rank.v2", "boundary": "identity"})
        receipt = tmp_path / "receipts/layer-022-postwrite-abi-v2.json"
        value = json.loads(receipt.read_text()); value["ranks"][2].update(bytes=size, sha256=digest); receipt.write_text(json.dumps(value))
        with pytest.raises(ValueError, match="metadata/fallback"):
            runtime.validate_coupled_sidecars(tmp_path, design, transform, postwrites, loaders)
    finally:
        runtime.COUPLED_DESIGN_SHA256, runtime.TRANSFORM_SHA256, runtime.EXL3_SCALE_SOURCE_SHA256 = old_d, old_t, old_s


def test_coupled_loader_identity_fallback_fails(tmp_path):
    design, transform, postwrites, loaders, old_d, old_t, old_s = _coupled_tree(tmp_path)
    try:
        loader = loaders[20]
        value = json.loads(loader.read_text()); value["ranks"][1]["identity_fallback"] = True
        loader.write_text(json.dumps(value))
        with pytest.raises(ValueError, match="real-loader closure differs"):
            runtime.validate_coupled_sidecars(tmp_path, design, transform, postwrites, loaders)
    finally:
        runtime.COUPLED_DESIGN_SHA256, runtime.TRANSFORM_SHA256, runtime.EXL3_SCALE_SOURCE_SHA256 = old_d, old_t, old_s


def _log(arm: str) -> str:
    boundary = "identity" if arm == "identity_p8" else "coupled-h512-h128-suh-svh-v1"
    full = "false" if arm == "identity_p8" else "true"
    lines = []
    for layer in runtime.LAYERS:
        for rank in runtime.RANKS:
            lines += [f"GLM53_P8_NATIVE_WEIGHTS_READY layer={layer} rank={rank} x boundary={boundary} full_coupled={full}",
                      f"GLM53_P8_NATIVE_FORWARD layer={layer} rank={rank} x boundary={boundary}"]
    return "\n".join(lines)


def test_log_gate_rejects_missing_pair_and_wrong_boundary():
    text = _log("coupled_p8")
    assert runtime.verify_runtime_log(text, "coupled_p8")["full_coupled"] is True
    with pytest.raises(ValueError, match="missing rank/layer"):
        runtime.verify_runtime_log(text.replace("GLM53_P8_NATIVE_WEIGHTS_READY layer=22 rank=3", "REMOVED", 1), "coupled_p8")
    with pytest.raises(ValueError, match="fallback"):
        runtime.verify_runtime_log(text.replace("boundary=coupled-h512-h128-suh-svh-v1", "boundary=identity", 1), "coupled_p8")


def test_arm_environment_stock_is_explicitly_off_and_coupled_exact():
    stock = runtime.arm_environment("stock", ["conditional-fit-0001"])
    assert stock["GLM53_P8_NATIVE"] == "" and stock["GLM53_P8_NATIVE_TRANSFORM"] == ""
    coupled = runtime.arm_environment("coupled_p8", ["conditional-fit-0001"])
    assert coupled["GLM53_P8_NATIVE_LAYERS"] == "3,20,22"
    assert coupled["GLM53_P8_FC1_TILE_N"] == "128"
    assert coupled["GLM53_P8_FUSED_SCRATCH"] == ""
    assert coupled["GLM53_P8_NATIVE_SIDECAR_DIR"] == "/tmp/p8-three-layer-sidecars"
    assert coupled["GLM53_P8_NATIVE_TRANSFORM"] == "/p8-design/transform.json"
    assert coupled["PYTHONPATH"] == runtime.V9_PYTHONPATH


def test_launch_removes_stale_runtime_and_uses_exact_coupled_flat_path(tmp_path):
    module = _preparer()
    recipe = {"Config": {"Cmd": ["-lc", "/opt/venv/bin/python -m vllm.entrypoints.cli.main serve /old --port 1 --served-model-name old --tensor-parallel-size 4 --decode-context-parallel-size 1 --attention-backend B12X_MLA_SPARSE --kv-cache-dtype nvfp4_ds_mla --max-num-seqs 1 --quantization modelopt"],
                         "Env": ["PYTHONPATH=/runtime-patch:/stale", "KEEP=1"]},
              "HostConfig": {"ShmSize": 64, "Runtime": "runc",
                             "Binds": ["/host/stale:/runtime-patch:ro", "/host/old:/model:ro", "/host/cache:/cache:rw"]}}
    env = runtime.arm_environment("coupled_p8", ["conditional-fit-0001"])
    argv = module.launch_argv(recipe, "sha256:" + "1" * 64, "coupled_p8", env, tmp_path / "captures",
        tmp_path / "model", tmp_path / "identity", tmp_path / "coupled", tmp_path / "identity.json",
        tmp_path / "coupled.json", tmp_path / "transform.json")
    joined = "\n".join(argv)
    command = argv[-1]
    assert "/host/stale:/runtime-patch:ro" not in joined
    assert "PYTHONPATH=/runtime-patch:/stale" not in joined
    assert f"PYTHONPATH={runtime.V9_PYTHONPATH}" in argv
    assert "rm -rf" not in command
    assert command.startswith("set -e; mkdir /tmp/p8-three-layer-sidecars;")
    assert command.count("ln -s /p8-coupled-sidecars/layer-") == 12
    assert env["GLM53_P8_NATIVE_SIDECAR_DIR"] == "/tmp/p8-three-layer-sidecars"
    assert f"{tmp_path / 'captures' / 'captures'}:/p8-captures:rw" in argv


def test_launch_normalizes_real_source_leading_exec_to_exactly_one(tmp_path):
    module = _preparer()
    command = "exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve /old --port 1 --served-model-name old --tensor-parallel-size 4 --decode-context-parallel-size 1 --attention-backend B12X_MLA_SPARSE --kv-cache-dtype nvfp4_ds_mla --max-num-seqs 1 --quantization modelopt"
    recipe = {"Config": {"Cmd": ["-lc", command], "Env": []},
              "HostConfig": {"ShmSize": 64, "Runtime": "runc", "Binds": []}}
    env = runtime.arm_environment("stock", ["conditional-fit-0001"])
    argv = module.launch_argv(
        recipe, "sha256:" + "1" * 64, "stock", env, tmp_path / "stock",
        tmp_path / "model", tmp_path / "identity", tmp_path / "coupled",
        tmp_path / "identity.json", tmp_path / "coupled.json", tmp_path / "transform.json")
    emitted = argv[-1]
    assert emitted.startswith("exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve /model")
    assert emitted.split().count("exec") == 1


def test_launch_rejects_nonleading_or_duplicate_exec(tmp_path):
    module = _preparer()
    command = "/opt/venv/bin/python -m vllm.entrypoints.cli.main exec serve /old --port 1 --served-model-name old --tensor-parallel-size 4 --decode-context-parallel-size 1 --attention-backend B12X_MLA_SPARSE --kv-cache-dtype nvfp4_ds_mla --max-num-seqs 1 --quantization modelopt"
    recipe = {"Config": {"Cmd": ["-lc", command], "Env": []},
              "HostConfig": {"ShmSize": 64, "Runtime": "runc", "Binds": []}}
    with pytest.raises(ValueError, match="serving prefix"):
        module.launch_argv(
            recipe, "sha256:" + "1" * 64, "stock",
            runtime.arm_environment("stock", ["conditional-fit-0001"]), tmp_path / "stock",
            tmp_path / "model", tmp_path / "identity", tmp_path / "coupled",
            tmp_path / "identity.json", tmp_path / "coupled.json", tmp_path / "transform.json")


def test_production_guard_uses_user_backend_and_system_timer(monkeypatch):
    module, calls = _preparer(), []
    def run(argv, **_kwargs):
        calls.append(argv); return SimpleNamespace(stdout="inactive\n", returncode=3)
    class Sock:
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def setsockopt(self, *_args): pass
        def bind(self, address): assert address == ("127.0.0.1", 8000)
    monkeypatch.setattr(module.socket, "socket", lambda: Sock())
    state = module.production_off(run)
    assert calls == [["systemctl", "--user", "is-active", "klc-backend.service"],
                     ["systemctl", "is-active", "klc-model-stack.timer"]]
    assert state["units"]["klc-backend.service"]["scope"] == "user"
    assert state["units"]["klc-model-stack.timer"]["scope"] == "system"


def test_capture_receipt_rejects_non_v9_parent(tmp_path):
    receipt = {"schema": "glm53.p8-coupled-cf32-capture-image.v1", "status": "complete",
        "parent_image_id": "sha256:" + "0" * 64, "image_id": "sha256:" + "1" * 64,
        "runtime_manifest_sha256": runtime.V9_MANIFEST_SHA256, "tail_v2_sha256": runtime.TAIL_V2_SHA256,
        "gpu_used": False, "speed_measurement_valid": False, "installed_sha256": {}}
    path = tmp_path / "receipt.json"; path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="receipt differs"):
        runtime.validate_capture_image_receipt(path)


def test_capture_receipt_reuses_attested_v9_without_child(tmp_path):
    receipt = {"schema": "glm53.p8-coupled-cf32-capture-image.v1", "status": "complete",
        "parent_image_id": runtime.V9_IMAGE, "image_id": runtime.V9_IMAGE, "reuse_existing_image": True,
        "image_built": False, "added_image_bytes": 0,
        "runtime_manifest_sha256": runtime.V9_MANIFEST_SHA256, "tail_v2_sha256": runtime.TAIL_V2_SHA256,
        "gpu_used": False, "speed_measurement_valid": False,
        "installed_sha256": {**{f"p8_decode_capture/{k}": v for k, v in runtime.CAPTURE_SOURCES.items()},
                             "sampler.py": runtime.SAMPLER_PATCHED_SHA256,
                             "warmup.py": runtime.WARMUP_PATCHED_SHA256},
        "full_installed_sha256": {
            "/usr/lib/python3.12/sitecustomize.py": runtime.V9_SITECUSTOMIZE_SHA256,
            "/opt/p8-coupled-runtime/image-manifest.json": runtime.V9_MANIFEST_SHA256,
            "/opt/infernal-invocation/vllm/vllm/v1/worker/gpu/sample/sampler.py": runtime.SAMPLER_PATCHED_SHA256,
            "/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/sample/sampler.py": runtime.SAMPLER_PATCHED_SHA256,
            "/opt/infernal-invocation/vllm/vllm/v1/worker/gpu/warmup.py": runtime.WARMUP_PATCHED_SHA256,
            "/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/warmup.py": runtime.WARMUP_PATCHED_SHA256}}
    path = tmp_path / "receipt.json"; path.write_text(json.dumps(receipt))
    assert runtime.validate_capture_image_receipt(path)["image_id"] == runtime.V9_IMAGE
