import hashlib
import json
from pathlib import Path

import pytest

from glm53_nvfp4 import p8_full_coupled_cf32_executor as executor
from glm53_nvfp4 import p8_full_coupled_runtime as runtime


DESIGN_A = "a" * 64
DESIGN_B = "b" * 64


def _recipe() -> dict:
    command = (
        "exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve /models/GLM-5.3-Flash-NVFP4 "
        "--tensor-parallel-size 4 --decode-context-parallel-size 1 --attention-backend B12X_MLA_SPARSE "
        "--kv-cache-dtype nvfp4_ds_mla --max-num-seqs 1 --quantization modelopt --port 8000 "
        "--served-model-name glm53-reference"
    )
    return {
        "Config": {"Cmd": ["-lc", command], "Env": ["PYTHONPATH=/old", "OTHER=1"]},
        "HostConfig": {"ShmSize": 68719476736, "Runtime": "nvidia",
                       "Binds": ["/models:/model:ro", "/x/patch:/runtime-patch:ro", "/data/teacher:/teacher:ro",
                                 "/old/sidecars:/p8-sidecars:ro", "/old/design:/p8-design/design.json:ro"]},
    }


def test_launch_argv_rewrites_only_the_declared_fields(tmp_path: Path):
    env = runtime.arm_environment("coupled_full", ["w1"], design_count=2)
    argv = runtime.launch_argv(_recipe(), "sha256:" + "1" * 64, "coupled_full", env, tmp_path / "out",
                               Path("/models/carrier"), Path("/full/sidecars"),
                               [Path("/d/v4.json"), Path("/d/v3.json")], Path("/d/transform.json"))
    assert argv[:2] == ["docker", "create"] and argv[argv.index("--name") + 1] == "glm53-p8-full-cf32-coupled_full"
    volumes = [argv[i + 1] for i, token in enumerate(argv) if token == "--volume"]
    assert "/models/carrier:/model:ro" in volumes and "/full/sidecars:/p8-sidecars:ro" in volumes
    assert "/d/v4.json:/p8-design/design-0.json:ro" in volumes and "/d/v3.json:/p8-design/design-1.json:ro" in volumes
    assert "/d/transform.json:/p8-design/transform.json:ro" in volumes
    assert "/data/teacher:/teacher:ro" in volumes
    assert not any(v.startswith("/old/") for v in volumes) and not any("/runtime-patch" in v for v in volumes)
    envs = [argv[i + 1] for i, token in enumerate(argv) if token == "--env"]
    assert "OTHER=1" in envs and f"PYTHONPATH={runtime.V10_PYTHONPATH}" in envs and "PYTHONPATH=/old" not in envs
    assert "GLM53_P8_NATIVE_LAYERS=" + runtime.LAYER_SPEC in envs
    shell = argv[-1]
    assert shell.startswith("exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve /model ")
    assert "--port 8032" in shell and "--served-model-name glm53-p8-full-cf32-coupled_full" in shell
    identity_env = runtime.arm_environment("identity_full", ["w1"], design_count=1)
    identity = runtime.launch_argv(_recipe(), "sha256:" + "1" * 64, "identity_full", identity_env, tmp_path / "out",
                                   Path("/models/carrier"), Path("/identity/sidecars"), [Path("/d/native6.json")], None)
    assert not any("transform" in v for v in identity)
    with pytest.raises(ValueError, match="requires the encoder transform"):
        runtime.launch_argv(_recipe(), "sha256:" + "1" * 64, "coupled_full", env, tmp_path / "out",
                            Path("/m"), Path("/s"), [Path("/d/a"), Path("/d/b")], None)
    with pytest.raises(ValueError, match="design mounts differ"):
        runtime.launch_argv(_recipe(), "sha256:" + "1" * 64, "coupled_full", env, tmp_path / "out",
                            Path("/m"), Path("/s"), [Path("/d/a")], Path("/d/t"))


def test_launch_argv_rejects_topology_drift(tmp_path: Path):
    recipe = _recipe()
    recipe["Config"]["Cmd"][1] = recipe["Config"]["Cmd"][1].replace("--decode-context-parallel-size 1", "--decode-context-parallel-size 4")
    env = runtime.arm_environment("identity_full", ["w1"], design_count=1)
    with pytest.raises(ValueError, match="topology differs"):
        runtime.launch_argv(recipe, "sha256:" + "1" * 64, "identity_full", env, tmp_path, Path("/m"), Path("/s"), [Path("/d")], None)
    recipe = _recipe()
    recipe["Config"]["Cmd"][1] += " --enable-expert-parallel"
    with pytest.raises(ValueError, match="TP4/DCP1/noEP"):
        runtime.launch_argv(recipe, "sha256:" + "1" * 64, "identity_full", env, tmp_path, Path("/m"), Path("/s"), [Path("/d")], None)


def _common_log() -> list[str]:
    rows = ["Using V2 Model Runner", "tensor_parallel_size=4", "decode_context_parallel_size=1",
            "speculative_config=None", "kv_cache_dtype=nvfp4_ds_mla", "quantization=modelopt_mixed"]
    from glm53_nvfp4 import p8_decode_protocol as protocol
    rows += [f"GLM53_P8_DECODE_CAPTURE_V2_READY tp_rank={rank} max_num_reqs=1 real_vocab={protocol.VOCAB_LIMIT} expected_outputs=2047"
             for rank in range(4)]
    rows += [f"GLM53_P8_DECODE_CAPTURE_V2_WARMUP_SCOPE_CLOSED tp_rank={rank} registrations=1 samples=2" for rank in range(4)]
    return rows


def _native_log(arm: str, designs: dict[int, str]) -> list[str]:
    boundary, full = runtime.BOUNDARIES[arm], runtime.FULL_COUPLED_FLAG[arm]
    rows = [f"GLM53_P8_NATIVE_PATCH_ACTIVE layers={runtime.LAYER_SPEC} tp=4 design_sha256_allowlist=x"]
    for layer in runtime.LAYERS:
        for rank in runtime.RANKS:
            rows.append(f"GLM53_P8_NATIVE_WEIGHTS_READY layer={layer} rank={rank} sidecar=/p8-sidecars/f design_sha256={designs[layer]} "
                        f"released_carrier_bytes=1 boundary={boundary} full_coupled={full} ldlq=false")
            rows.append(f"GLM53_P8_NATIVE_FORWARD layer={layer} rank={rank} stream=K4 boundary={boundary} ldlq=false")
    return rows


def test_audit_runtime_log_binds_designs_image_and_completion_order(monkeypatch):
    monkeypatch.setattr(executor, "verify_graphs", lambda _text: {"status": "pass"})
    designs = {layer: (DESIGN_B if layer in (3, 20, 22) else DESIGN_A) for layer in runtime.LAYERS}
    text = "\n".join(_common_log() + _native_log("coupled_full", designs))
    audit = executor.audit_runtime_log(text, "coupled_full", design_by_layer=designs, image_id="sha256:" + "9" * 64, bpw=4.2539798595)
    assert audit["conditions"]["layers"] == "3-44" and audit["conditions"]["bpw"] == 4.2539798595
    assert audit["conditions"]["moe_backend"].endswith("coupled-h512-h128-suh-svh-v1")
    with pytest.raises(ValueError, match="loaded design"):
        executor.audit_runtime_log(text, "coupled_full", design_by_layer={l: DESIGN_A for l in runtime.LAYERS},
                                   image_id="sha256:" + "9" * 64, bpw=4.25)
    completed = text + "\n" + "\n".join(f"GLM53_P8_DECODE_CAPTURE_V2_COMPLETE window=conditional-fit-{i:04d} rows=2047 tp_rank=0" for i in (5, 9))
    audit = executor.audit_runtime_log(completed, "coupled_full", design_by_layer=designs, image_id="x", bpw=4.25,
                                       completed=["conditional-fit-0005", "conditional-fit-0009"])
    assert audit["completed_window_ids"] == ["conditional-fit-0005", "conditional-fit-0009"]
    with pytest.raises(ValueError, match="completion inventory differs"):
        executor.audit_runtime_log(completed, "coupled_full", design_by_layer=designs, image_id="x", bpw=4.25,
                                   completed=["conditional-fit-0009", "conditional-fit-0005"])
    with pytest.raises(ValueError, match="runtime marker missing"):
        executor.audit_runtime_log(text.replace("kv_cache_dtype=nvfp4_ds_mla", "kv_cache_dtype=fp8_ds_mla"),
                                   "coupled_full", design_by_layer=designs, image_id="x", bpw=4.25)


def test_storage_gate_charges_campaign_paths_and_reserves_one_raw(tmp_path: Path, monkeypatch):
    campaign = tmp_path / "campaign"; campaign.mkdir(); (campaign / "blob").write_bytes(b"x" * 4096)
    output = tmp_path / "eval"
    manifest = {"storage": {"max_new_bytes": 4096 + executor.RAW_BYTES, "campaign_paths": [str(campaign)],
                            "one_window_raw_bytes": executor.RAW_BYTES}}
    state = executor._storage_state(manifest, output)
    assert state["campaign_apparent_bytes"] == 4096 and str(output) in state["campaign_paths"]
    monkeypatch.setattr(executor, "MIN_FREE_BYTES", 0)
    executor._require_storage(state, reserve_raw=True)
    manifest["storage"]["max_new_bytes"] = 4096 + executor.RAW_BYTES - 1
    with pytest.raises(ValueError, match="ceiling would be exceeded"):
        executor._require_storage(executor._storage_state(manifest, output), reserve_raw=True)
    executor._require_storage(executor._storage_state(manifest, output), reserve_raw=False)
    monkeypatch.setattr(executor, "MIN_FREE_BYTES", 10 ** 18)
    with pytest.raises(ValueError, match="does not fit"):
        executor._require_storage(executor._storage_state(manifest, output), reserve_raw=False)


def test_image_receipt_validation_requires_v10_pass_and_presence(tmp_path: Path):
    image = "sha256:" + "7" * 64
    receipt = {"schema": runtime.IMAGE_BUILD_SCHEMA, "status": "complete", "parent_image_id": runtime.IMAGE_PARENT,
               "image_id": image, "verification": {"status": "pass"}, "gpu_used": False,
               "labels": {"org.klc.experiment": runtime.IMAGE_EXPERIMENT_LABEL, "org.klc.parent.digest": runtime.IMAGE_PARENT},
               "source_commit": "c" * 40, "source_tree": "t" * 40}
    path = tmp_path / "receipt.json"; path.write_text(json.dumps(receipt))
    result = runtime.validate_image_receipt(path, docker_inspect=lambda name: image)
    assert result["image_id"] == image and result["receipt"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="not present"):
        runtime.validate_image_receipt(path, docker_inspect=lambda name: "sha256:" + "8" * 64)
    receipt["labels"]["org.klc.experiment"] = "glm53-p8-coupled-h512-h128-suh-svh-v9"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="receipt differs"):
        runtime.validate_image_receipt(path, docker_inspect=lambda name: image)


def test_analysis_requires_matched_conditions_and_reports_strict_claim():
    from glm53_nvfp4 import analyze_p8_full_coupled_cf32 as analysis
    domains = ["axis1_general", "axis2_legal", "axis3_code_agentic", "axis4_reasoning_termination"]
    windows = [{"id": f"conditional-fit-{i:04d}", "domain": domains[i % 4]} for i in range(32)]
    base = {"attention": "B12X_MLA_SPARSE", "kv_dtype": "nvfp4_ds_mla", "activation_precision": "E4M3",
            "layers": "3-44", "image_id": "sha256:x"}
    def rows(offset):
        return [{"window_id": w["id"], "domain": w["domain"], "mean_kld": 0.04 + offset + 0.001 * (i % 3),
                 "true_decode_mean_kld": 0.038 + offset + 0.001 * (i % 3)} for i, w in enumerate(windows)]
    arms = {"coupled_full": {"conditions": {**base, "moe_backend": "c", "bpw": 4.2539798595}, "windows": rows(-0.002)},
            "identity_full": {"conditions": {**base, "moe_backend": "i", "bpw": 4.25}, "windows": rows(0.0)}}
    result = analysis.analyze(windows, arms)
    assert result["decision"] == "pass" and result["strict_claim"] is True
    assert abs(result["comparison"]["mean_delta_kld"] + 0.002) < 1e-12 and result["comparison"]["paired_window_wins"] == 32
    arms["identity_full"]["conditions"]["image_id"] = "sha256:y"
    with pytest.raises(ValueError, match="not matched"):
        analysis.analyze(windows, arms)
