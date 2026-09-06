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
SCRIPT = ROOT / "scripts/run_p8_coupled_prefill_device_closure.py"


def _module():
    spec = importlib.util.spec_from_file_location("p8_coupled_prefill_closure", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _synthetic_grouped_schedule(module, m: int):
    ids = torch.tensor(module.EXPERT_IDS, dtype=torch.int32).repeat(m, 1)
    flat = ids.reshape(-1)
    counts = torch.bincount(flat.to(torch.int64), minlength=288).to(torch.int32)
    spans = (counts.to(torch.int64) + 63) // 64
    bases = torch.cat((torch.zeros(1, dtype=torch.int64), spans.cumsum(0))).to(
        torch.int32
    )
    token_map = torch.full((int(bases[-1]) * 64,), -1, dtype=torch.int32)
    for expert in module.EXPERT_IDS:
        routes = (flat == expert).nonzero().reshape(-1)
        begin = int(bases[expert]) * 64
        token_map[begin : begin + routes.numel()] = routes.to(torch.int32)
    return ids, counts, bases, token_map


def test_default_invocation_is_frozen_protocol_only() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)], text=True, capture_output=True, check=True
    )
    record = json.loads(result.stdout)
    assert record["protocol_sha256"] == (
        "7c6db9db4728ed6c6e626c976f8bfe77a9729a8cc48a4c04f98fcf734a5c0fb8"
    )
    protocol = record["protocol"]
    assert protocol["geometry"]["m_cases"] == [2, 64, 65]
    assert protocol["m1_protocol_sha256"] == (
        "9556885c32385574530acf7b07c1d5636b46852ca9714c27883f98f17368efe1"
    )
    assert protocol["numeric_gates"] == {
        "route_cosine_strictly_greater_than": 0.995,
        "route_relative_l2_strictly_less_than": 0.12,
        "final_cosine_strictly_greater_than": 0.995,
        "final_relative_l2_strictly_less_than": 0.12,
    }


@pytest.mark.parametrize(
    "m,expected_tiles,expected_partial",
    [(2, 1, 2), (64, 1, 0), (65, 2, 1)],
)
def test_grouped_schedule_maps_every_route_and_freezes_tail_geometry(
    m: int, expected_tiles: int, expected_partial: int
) -> None:
    module = _module()
    ids, counts, bases, token_map = _synthetic_grouped_schedule(module, m)
    physical, receipt = module.map_route_order_physical_rows(
        counts, bases, token_map, ids.reshape(-1)
    )
    assert len(physical) == m * 8
    assert len(set(physical)) == m * 8
    assert receipt["active_experts"] == 8
    for expert in module.EXPERT_IDS:
        span = receipt["active_spans"][str(expert)]
        assert span == {
            "rows": m,
            "tiles": expected_tiles,
            "partial_rows": expected_partial,
        }


def test_grouped_schedule_rejects_count_route_and_prefix_corruption() -> None:
    module = _module()
    ids, counts, bases, token_map = _synthetic_grouped_schedule(module, 2)
    bad_counts = counts.clone()
    bad_counts[module.EXPERT_IDS[0]] += 1
    with pytest.raises(RuntimeError, match="row_counts"):
        module.map_route_order_physical_rows(
            bad_counts, bases, token_map, ids.reshape(-1)
        )
    bad_map = token_map.clone()
    bad_map[int(bases[module.EXPERT_IDS[0]]) * 64] = 1
    with pytest.raises(RuntimeError, match="different expert"):
        module.map_route_order_physical_rows(
            counts, bases, bad_map, ids.reshape(-1)
        )
    bad_base = bases.clone()
    bad_base[module.EXPERT_IDS[0] + 1] += 1
    with pytest.raises(RuntimeError, match="tile span"):
        module.map_route_order_physical_rows(
            counts, bad_base, token_map, ids.reshape(-1)
        )


def test_prototype_expansion_preserves_route_order_and_numeric_reduction(monkeypatch) -> None:
    module = _module()
    sys.path.insert(0, str(ROOT / "runtime_patch"))
    import p8_coupled_scales

    prototype_index = torch.tensor([0, 1, 0], dtype=torch.long)
    cache = {
        "input_payload": torch.arange(2 * 4096, dtype=torch.int32)
        .remainder(251).to(torch.uint8).reshape(2, 4096),
        "input_scale": torch.arange(2 * 128, dtype=torch.int32)
        .remainder(251).to(torch.uint8).reshape(2, 128),
        "middle_payload": torch.arange(2 * 8 * 512, dtype=torch.int32)
        .remainder(251).to(torch.uint8).reshape(2, 8, 512),
        "middle_scale": torch.arange(2 * 8 * 16, dtype=torch.int32)
        .remainder(251).to(torch.uint8).reshape(2, 8, 16),
        "routes": torch.arange(2 * 8 * 4096, dtype=torch.float32).reshape(2, 8, 4096),
    }
    weights = torch.full((3, 8), 1 / 8, dtype=torch.float32)
    monkeypatch.setattr(
        p8_coupled_scales,
        "hadamard_blocks",
        lambda value, block: value,
    )
    try:
        actual = module.expand_reference_case(cache, prototype_index, weights)
    finally:
        sys.path.remove(str(ROOT / "runtime_patch"))
    assert torch.equal(actual["input_payload"][0], actual["input_payload"][2])
    assert torch.equal(
        actual["middle_payload"].reshape(3, 8, 512)[1], cache["middle_payload"][1]
    )
    expected = (cache["routes"].index_select(0, prototype_index) * weights[..., None]).sum(1)
    assert torch.equal(actual["final"], expected.to(torch.bfloat16))


def test_carrier_and_numeric_gates_are_distinct_and_m1_is_not_weakened() -> None:
    module = _module()
    assert any("byte equals" in gate for gate in module.PROTOCOL["exact_gates"])
    assert module.PROTOCOL["numeric_gates"] == module.M1.PROTOCOL["numeric_gates"]
    assert module.M1._numeric_pass(
        {"cosine": 0.996, "relative_l2": 0.119, "max_abs": 99.0}, "route"
    )
    assert not module.M1._numeric_pass(
        {"cosine": 0.995, "relative_l2": 0.119, "max_abs": 0.0}, "route"
    )


def test_wrapper_source_forces_actual_m_gt_1_materialized_dispatch() -> None:
    module = _module()
    source = (ROOT / "runtime_patch/p8_native_kernel.py").read_text()
    assert "if self.full_coupled:" in source
    assert "small_m = m == 1" in source
    assert "materialized = True" in source
    assert '"small_m": small_m, "materialized": materialized' in source
    assert module._expected_dispatch() == {
        "small_m": False,
        "materialized": True,
        "fused_scratch_zero": False,
        "fc1_tile_n": 128,
        "tile_m": 64,
    }


def test_runtime_manifest_gate_includes_both_prefill_kernels_and_wrapper() -> None:
    module = _module()
    assert module.MODULE_SOURCES["p8_native_kernel"] == "runtime_patch/p8_native_kernel.py"
    assert set(module.MODULE_SOURCES) >= {
        "p8_coupled_prefill_plan",
        "b12x.moe._shared.kernels.dynamic",
        "b12x.moe._shared.kernels.p8_coupled_prefill_fc1",
        "b12x.moe._shared.kernels.p8_coupled_prefill_fc2",
        "b12x.moe._shared.kernels.p8_coupled_topk",
    }


def test_scratch_forecast_is_bounded_at_all_frozen_cases(monkeypatch) -> None:
    module = _module()
    sys.path.insert(0, str(ROOT / "runtime_patch"))
    try:
        receipts = module._scratch_receipts()
    finally:
        sys.path.remove(str(ROOT / "runtime_patch"))
    assert [item["m"] for item in receipts] == [2, 64, 65]
    assert max(item["upper_bound_bytes"] for item in receipts) == 123_834_624
    assert max(item["upper_bound_bytes"] for item in receipts) < 130 * 1024 * 1024


def test_probe_command_is_network_disabled_read_only_and_prefill_specific(tmp_path: Path) -> None:
    module = _module()
    args = argparse.Namespace(
        gpu_device="0",
        sidecar=Path("/fixture/sidecar.safetensors"),
        sidecar_sha256="1" * 64,
        design=Path("/fixture/design.json"),
        design_sha256="2" * 64,
        transform=Path("/fixture/transform.json"),
        transform_sha256="3" * 64,
        runtime_manifest_sha256="4" * 64,
        output=tmp_path,
        image="sha256:" + "5" * 64,
    )
    command = module.build_probe_command(args, ROOT)
    joined = " ".join(str(item) for item in command)
    assert "--network=none" in command
    assert f"{ROOT}:/work:ro" in command
    assert "/work/scripts/run_p8_coupled_prefill_device_closure.py" in command
    assert "/work/scripts/run_p8_coupled_m1_device_closure.py" not in command
    assert "docker compose" not in joined and "systemctl" not in joined
