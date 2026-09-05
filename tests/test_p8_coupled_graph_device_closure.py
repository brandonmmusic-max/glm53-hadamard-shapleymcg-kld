from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_p8_coupled_graph_device_closure.py"


def _module():
    spec = importlib.util.spec_from_file_location("p8_coupled_graph_closure", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_is_cpu_only_frozen_protocol() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)], text=True, capture_output=True, check=True
    )
    record = json.loads(completed.stdout)
    assert record["protocol_sha256"] == (
        "3281b800664e7987732a681959c561b0c74905271e34a3da07e1ea004a0a2843"
    )
    protocol = record["protocol"]
    assert protocol["geometry"]["m_cases"] == [1, 2, 64, 65]
    assert protocol["repetitions_per_case"] == 5
    assert protocol["image_rebuild_required"] is False
    assert "not KLD" in protocol["claim_boundary"]
    assert protocol["frozen_inputs"]["payload_sha256"] == _module().FROZEN_PAYLOAD_SHA256


def test_v9_eager_receipts_and_immutable_inputs_are_exact() -> None:
    module = _module()
    m1 = ROOT / "evidence/preparation/p8-v9/m1-device-closure-v9/result.json"
    prefill = ROOT / "evidence/preparation/p8-v9/prefill-device-closure-v9/result.json"
    assert module.M1.sha256_file(m1) == module.M1_RESULT_SHA256
    assert module.M1.sha256_file(prefill) == module.PREFILL_RESULT_SHA256
    for path in (m1, prefill):
        result = json.loads(path.read_text())
        assert result["decision"] == "pass"
        assert result["image_id"] == module.V9_IMAGE
        assert result["identities"] == {
            "sidecar": module.V9_SIDECAR_SHA256,
            "design": module.V9_DESIGN_SHA256,
            "transform": module.V9_TRANSFORM_SHA256,
            "runtime_manifest": module.V9_RUNTIME_MANIFEST_SHA256,
        }


@pytest.mark.parametrize(
    "m,expected",
    [
        (1, {"small_m": True, "materialized": True, "fused_scratch_zero": False,
             "fc1_tile_n": 128, "tile_m": 16}),
        (2, {"small_m": False, "materialized": True, "fused_scratch_zero": False,
             "fc1_tile_n": 128, "tile_m": 64}),
        (64, {"small_m": False, "materialized": True, "fused_scratch_zero": False,
              "fc1_tile_n": 128, "tile_m": 64}),
        (65, {"small_m": False, "materialized": True, "fused_scratch_zero": False,
              "fc1_tile_n": 128, "tile_m": 64}),
    ],
)
def test_dispatch_contract_covers_both_actual_wrapper_arms(m: int, expected: dict) -> None:
    assert _module()._expected_dispatch(m) == expected


def _m1_observation_fixture(module):
    rows_capacity = 128
    selected = [slot * 16 for slot in range(8)]
    input_payload = torch.arange(4096, dtype=torch.int32).remainder(251).to(torch.uint8)
    input_scale = torch.arange(128, dtype=torch.int32).remainder(251).to(torch.uint8)
    middle_payload = torch.arange(8 * 512, dtype=torch.int32).remainder(251).to(torch.uint8).reshape(8, 512)
    middle_scale = torch.arange(8 * 16, dtype=torch.int32).remainder(251).to(torch.uint8).reshape(8, 16)
    intermediate = torch.zeros(rows_capacity * (128 + 4), dtype=torch.int32)
    payload_plane = intermediate[: rows_capacity * 128].view(torch.uint8).reshape(rows_capacity, 512)
    scale_plane = intermediate[rows_capacity * 128 :]
    for slot, row in enumerate(selected):
        payload_plane[row].copy_(middle_payload[slot])
        for slice_id in range(4):
            word = middle_scale[slot, slice_id * 4 : (slice_id + 1) * 4].clone().view(torch.int32)
            scale_plane[slice_id * rows_capacity + row] = word[0]
    routes = torch.arange(8 * 32, dtype=torch.float32).reshape(8, 32) / 97
    final = torch.arange(32, dtype=torch.float32).reshape(1, 32).to(torch.bfloat16)

    class Runtime:
        debug_dispatch = module._expected_dispatch(1)
        debug_tensors = {
            "packed_a": input_payload.clone(),
            "scale_flat": input_scale.clone(),
            "intermediate_u32": intermediate,
            "route_output": routes.clone(),
            "token_map": torch.zeros(1, dtype=torch.int32),
            "row_counts": torch.zeros(288, dtype=torch.int32),
            "expert_tile_base": torch.zeros(289, dtype=torch.int32),
        }

    reference = {
        "input_payload": input_payload.reshape(1, 4096),
        "input_scale": input_scale.reshape(1, 128),
        "middle_payload": middle_payload,
        "middle_scale": middle_scale,
        "routes": routes,
        "final": final,
    }
    ids = torch.tensor(module.M1.EXPERT_IDS, dtype=torch.int32).reshape(1, 8)
    return Runtime(), final, ids, reference


def test_observation_requires_every_carrier_and_hashes_route_and_final() -> None:
    module = _module()
    runtime, output, ids, reference = _m1_observation_fixture(module)
    receipt = module._observe(runtime, output, m=1, ids_cpu=ids, reference=reference)
    assert all(receipt["exact_activation_carriers"].values())
    assert set(receipt["hashes"]) == {
        "input_payload", "input_scale", "middle_payload", "middle_scale",
        "route_output", "final_output",
    }
    runtime.debug_tensors["scale_flat"][0] ^= 1
    with pytest.raises(RuntimeError, match="carrier byte mismatch"):
        module._observe(runtime, output, m=1, ids_cpu=ids, reference=reference)


def test_capture_sequence_warms_then_eager_then_captures_and_replays() -> None:
    source = SCRIPT.read_text()
    warm = source.index("warm_stream = torch.cuda.Stream()")
    eager = source.index("eager_output = runtime(", warm)
    capture = source.index("graph = torch.cuda.CUDAGraph()", eager)
    replay = source.index("graph.replay()", capture)
    assert warm < eager < capture < replay
    assert "for repeat in range(REPETITIONS):" in source
    assert "observed[\"hashes\"] != eager[\"hashes\"]" in source


def test_partial_receipt_is_atomic_and_replaces_prior_progress(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "partial.json"
    module._write_partial(path, {"status": "running", "replays": [0]})
    assert json.loads(path.read_text()) == {"status": "running", "replays": [0]}
    module._write_partial(path, {"status": "running", "replays": [0, 1]})
    assert json.loads(path.read_text()) == {"status": "running", "replays": [0, 1]}
    assert not (tmp_path / "partial.json.tmp").exists()


def test_evidence_retry_preserves_original_whole_receipt_failure_predicate() -> None:
    source = SCRIPT.read_text()
    append = source.index("graph_runs.append(observed)")
    durable = source.index("_write_partial(partial_path, partial)", append)
    compare = source.index("comparable = [", durable)
    failure = source.index("five graph replays are not bitwise deterministic", compare)
    assert append < durable < compare < failure
    assert '{key: value for key, value in run.items() if key != "repeat"}' in source


def test_probe_command_reuses_v9_without_network_or_image_build(tmp_path: Path) -> None:
    module = _module()
    args = argparse.Namespace(
        gpu_device="0",
        sidecar=Path("/fixture/sidecar.safetensors"),
        design=Path("/fixture/design.json"),
        transform=Path("/fixture/transform.json"),
        output=tmp_path,
    )
    command = module.build_probe_command(args, ROOT)
    joined = " ".join(str(item) for item in command)
    assert "--network=none" in command
    assert module.V9_IMAGE in command
    assert f"{ROOT}:/work:ro" in command
    assert "/work/scripts/run_p8_coupled_graph_device_closure.py" in command
    assert "--harness-sha256" in command
    assert "docker build" not in joined and "systemctl" not in joined


def test_probe_fails_closed_without_explicit_harness_hash(tmp_path: Path) -> None:
    module = _module()
    args = module.parse_args([
        "--probe", "--image-id", module.V9_IMAGE,
        "--sidecar", "/x", "--design", "/y", "--transform", "/z",
        "--runtime-manifest", "/m", "--output", str(tmp_path / "r.json"),
    ])
    with pytest.raises(ValueError, match="harness_sha256"):
        module._require(
            args,
            ("sidecar", "design", "transform", "runtime_manifest", "output",
             "image_id", "harness_sha256"),
        )
