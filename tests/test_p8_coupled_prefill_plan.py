from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "runtime_patch"
sys.path.insert(0, str(PATCH))

from p8_coupled_prefill_plan import P8CoupledPrefillGeometry, prefill_strategy


def test_full_coupled_prefill_is_forced_to_split_m64() -> None:
    assert prefill_strategy(1) == "m1-n128"
    assert prefill_strategy(2) == "forced-split-m64"
    assert prefill_strategy(32768) == "forced-split-m64"
    with pytest.raises(ValueError):
        prefill_strategy(0)


@pytest.mark.parametrize(
    "tokens,route_bytes,extra_bytes",
    [
        (2, 262_144, 131_072),
        (4096, 536_870_912, 268_435_456),
        (32768, 4_294_967_296, 2_147_483_648),
    ],
)
def test_fp32_route_scratch_cost_is_exact(
    tokens: int, route_bytes: int, extra_bytes: int
) -> None:
    plan = P8CoupledPrefillGeometry(tokens)
    assert plan.route_scratch_bytes == route_bytes
    assert plan.unavoidable_transform_extra_bytes == extra_bytes


def test_4096_token_current_wrapper_memory_receipt() -> None:
    plan = P8CoupledPrefillGeometry(4096)
    assert plan.route_rows == 32768
    assert plan.physical_tiles_capacity == 800
    assert plan.rows_capacity == 51200
    assert plan.input_h512_units == 32768
    assert plan.fc1_tasks_capacity == 3200
    assert plan.fc2_tasks_capacity == 25600
    assert plan.packed_a_bytes_current == 209_715_200
    assert plan.scale_flat_bytes_current == 1_083_211_776
    assert plan.scale_flat_bytes_logical == 524_288
    assert plan.intermediate_bytes == 27_033_600
    assert plan.output_bytes == 33_554_432
    assert plan.control_upper_bound_bytes == 1_024_000
    assert plan.current_wrapper_scratch_upper_bound == 1_891_409_920


def test_existing_runtime_stays_fail_closed_for_m_greater_than_one() -> None:
    wrapper = (PATCH / "p8_native_kernel.py").read_text()
    assert 'if self.scale_component is not None and m != 1:' in wrapper
    assert 'raise RuntimeError("P8 scale sandwich currently supports M=1 only")' in wrapper


def test_existing_output_reducer_is_already_shape_generic() -> None:
    reducer = (
        PATCH
        / "b12x_h16/b12x/moe/_shared/kernels/p8_coupled_topk.py"
    ).read_text()
    assert "active_m * Int32(self.hidden // 512)" in reducer
    assert "token = unit // blocks_per_token" in reducer
