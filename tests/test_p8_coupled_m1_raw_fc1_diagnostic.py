from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_p8_coupled_m1_raw_fc1_diagnostic.py"


def _module():
    spec = importlib.util.spec_from_file_location("p8_raw_fc1_diag", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_is_localization_protocol_only() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)], text=True, capture_output=True, check=True
    )
    record = json.loads(result.stdout)
    protocol = record["protocol"]
    assert protocol["evidence_level"] == "gpu-localization-only"
    assert "closure pass" in protocol["forbidden_claims"]
    assert protocol["ldlq"] is False


def test_capture_rows_are_unique_bounded_route_tile_bijection() -> None:
    module = _module()
    rows = module.capture_rows()
    assert rows == [[slot * 16 + tile for tile in range(4)] for slot in range(8)]
    flat = [row for route in rows for row in route]
    assert len(flat) == len(set(flat)) == 32
    assert min(flat) == 0 and max(flat) == 115


def test_raw_fp16_extractor_preserves_projection_and_tile_order() -> None:
    module = _module()
    carrier = torch.zeros(128 * 132, dtype=torch.int32)
    words = carrier[: 128 * 128]
    for slot, route in enumerate(module.capture_rows()):
        for tile, row in enumerate(route):
            values = torch.empty(256, dtype=torch.float16)
            values[:128] = slot * 1000 + tile * 100 + torch.arange(128)
            values[128:] = -(slot * 1000 + tile * 100 + torch.arange(128))
            words[row * 128 : (row + 1) * 128] = values.view(torch.int32)
    gate, up = module.unpack_raw_fc1(carrier)
    assert gate.shape == up.shape == (8, 512)
    assert gate.dtype == up.dtype == torch.float16
    for slot in range(8):
        for tile in range(4):
            expected = (
                slot * 1000 + tile * 100 + torch.arange(128)
            ).to(torch.float16)
            torch.testing.assert_close(gate[slot, tile * 128 : (tile + 1) * 128], expected)
            torch.testing.assert_close(up[slot, tile * 128 : (tile + 1) * 128], -expected)


def test_raw_capture_sentinels_metadata_and_write_counts_fail_closed() -> None:
    module = _module()
    carrier = torch.full((128 * 132,), -1, dtype=torch.int32)
    payload = carrier[: 128 * 128].reshape(128, 128)
    tail = carrier[128 * 128 :]
    tail[32:64].zero_()
    for slot, expert in enumerate(module.EXPERT_IDS):
        for tile, row in enumerate(module.capture_rows()[slot]):
            payload[row].zero_()
            trace_slot = slot * 4 + tile
            tail[trace_slot] = expert | (slot << 9) | (tile << 12)
            tail[32 + trace_slot] += 1
    receipt = module.audit_raw_capture_bounds(carrier)
    assert receipt["selected_rows_all_written"] is True
    assert receipt["nonselected_rows"] == 96
    assert receipt["unused_tail_words"] == 448

    bad_count = carrier.clone()
    bad_count[128 * 128 + 32] = 2
    with pytest.raises(RuntimeError, match="metadata mismatch"):
        module.audit_raw_capture_bounds(bad_count)
    bad_bound = carrier.clone()
    bad_bound[4 * 128] = 0
    with pytest.raises(RuntimeError, match="outside selected"):
        module.audit_raw_capture_bounds(bad_bound)


def test_input_carrier_inverse_permutation_roundtrip() -> None:
    module = _module()
    logical = torch.arange(4096).remainder(120).to(torch.uint8).reshape(1, 4096)
    packed = module.closure.permute_k32_payload(logical)
    scales = torch.full((1, 128), 127, dtype=torch.uint8)
    reconstructed = module.reconstruct_input(packed, scales)
    expected = logical.view(torch.float8_e4m3fn).float()
    assert torch.equal(reconstructed, expected)


def test_prequant_trace_is_exactly_512_bytes_and_fails_partial_sentinel() -> None:
    module = _module()
    values = torch.arange(128, dtype=torch.float32)
    trace = module.unpack_input_prequant_trace(values.view(torch.uint8))
    assert list(trace) == [
        "block40_raw", "block40_normalized", "block62_raw", "block62_normalized"
    ]
    assert trace["block40_raw"].tolist() == list(range(32))
    bad = values.clone()
    bad[127] = float("nan")
    with pytest.raises(RuntimeError, match="completely overwritten"):
        module.unpack_input_prequant_trace(bad.view(torch.uint8))
    with pytest.raises(ValueError, match="exactly 512"):
        module.unpack_input_prequant_trace(torch.zeros(511, dtype=torch.uint8))


def test_input_trace_evidence_freezes_bits_midpoints_and_wire_map() -> None:
    module = _module()
    values = torch.tensor([-1.0, 0.0, 1.0, 1.0625], dtype=torch.float32)
    evidence = module.fp32_evidence(values)
    assert evidence["bits_hex"] == [
        "0xbf800000", "0x00000000", "0x3f800000", "0x3f880000"
    ]
    distances = module.e4m3_midpoint_distance(values)
    assert len(distances) == 4 and all(value >= 0 for value in distances)
    positions = module.logical_to_wire_positions(40)
    assert sorted(positions) == list(range(40 * 32, 41 * 32))
    trace = {
        "block40_raw": torch.arange(32, dtype=torch.float32),
        "block40_normalized": torch.arange(32, dtype=torch.float32) / 8,
        "block62_raw": torch.arange(32, dtype=torch.float32) + 1,
        "block62_normalized": (torch.arange(32, dtype=torch.float32) + 1) / 8,
    }
    payload = torch.zeros((1, 128, 32), dtype=torch.uint8)
    scale = torch.full((1, 128), 127, dtype=torch.uint8)
    rows = module.input_trace_rows(trace, trace, payload, scale, scale)
    assert len(rows) == 64
    assert rows[0]["block"] == 40 and rows[-1]["block"] == 62
    assert rows[0]["raw_device_bits"] == rows[0]["raw_reference_bits"]
    assert rows[0]["device_abs_distance_to_midpoint"] >= 0


def test_runtime_raw_mode_is_opt_in_fail_closed_and_skips_reducer() -> None:
    wrapper = (ROOT / "runtime_patch/p8_native_kernel.py").read_text()
    kernel = (
        ROOT / "runtime_patch/b12x_h16/b12x/moe/_shared/kernels/p8_h128_fc1.py"
    ).read_text()
    assert "diagnostic_raw_fc1: bool = False" in wrapper
    assert "raw FC1 diagnostic requires debug M1 small-M N128 capture" in wrapper
    assert "P8H128FC1RawCaptureKernel" in wrapper
    assert "torch.full((512,), 0xFF" in wrapper
    assert "kernel.p8_input_prequant_diagnostic = True" in wrapper
    assert 'self.debug_tensors["input_prequant_trace"]' in wrapper
    assert '("raw_fc1_diagnostic", int(self.diagnostic_raw_fc1))' in wrapper
    assert "if self.deterministic_output and not self.diagnostic_raw_fc1:" in wrapper
    assert "class P8H128FC1RawCaptureKernel" in kernel
    assert "diagnostic_raw_fc1 = True" in kernel
    assert "capture_row = (" in kernel
    assert "source_m_tile * Int32(self.source_tile_m) + output_tile" in kernel
    assert "return" in kernel[kernel.index("if cutlass.const_expr(self.diagnostic_raw_fc1)") : kernel.index("Full joint FC1 owner")]


def test_metric_gate_is_strict_and_diagnostic_never_passes_closure() -> None:
    module = _module()
    gate = module.PROTOCOL["numeric_gate"]
    assert module.metric_pass({"cosine": 0.996, "relative_l2": 0.119, "max_abs": 1.0})
    assert not module.metric_pass({
        "cosine": gate[
            "each_selected_expert_and_projection_cosine_strictly_greater_than"
        ],
        "relative_l2": 0.119,
        "max_abs": 1.0,
    })
    assert set(module.PROTOCOL["decision_rule"]) == {
        "localized_after_raw_fc1", "localized_at_or_before_raw_fc1"
    }
