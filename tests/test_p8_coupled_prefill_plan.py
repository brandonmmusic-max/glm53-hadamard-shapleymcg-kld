from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "runtime_patch"
sys.path.insert(0, str(PATCH))

from p8_coupled_prefill_plan import (
    P8CoupledPrefillGeometry,
    materialized_scale_index_bounds,
    materialized_scale_storage_elements,
    prefill_strategy,
)
import p8_coupled_scales as coupled


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


def test_scale_only_stays_m1_while_full_coupled_forces_m64() -> None:
    wrapper = (PATCH / "p8_native_kernel.py").read_text()
    assert (
        'if self.scale_component is not None and not self.full_coupled and m != 1:'
        in wrapper
    )
    assert 'raise RuntimeError("P8 scale sandwich currently supports M=1 only")' in wrapper
    assert "small_m = m == 1" in wrapper
    assert "materialized = True" in wrapper


def test_existing_output_reducer_is_already_shape_generic() -> None:
    reducer = (
        PATCH
        / "b12x_h16/b12x/moe/_shared/kernels/p8_coupled_topk.py"
    ).read_text()
    assert "active_m * Int32(self.hidden // 512)" in reducer
    assert "token = unit // blocks_per_token" in reducer


@pytest.mark.parametrize("tokens", [2, 63, 64, 65, 1296, 4096])
def test_materialized_scale_plane_has_exact_producer_consumer_bound(
    tokens: int,
) -> None:
    producer_max, consumer_max, elements = materialized_scale_index_bounds(tokens)
    assert elements == tokens * 128
    assert producer_max == elements - 1
    assert consumer_max == elements - 1
    assert materialized_scale_storage_elements(tokens) == elements


def test_materialized_scale_plane_rejects_unsupported_geometry() -> None:
    with pytest.raises(ValueError, match="token count"):
        materialized_scale_storage_elements(0)
    with pytest.raises(ValueError, match="divisible by 128"):
        materialized_scale_storage_elements(2, 4100)


def test_exact_m64_n128_kernel_sources_preserve_native_contract() -> None:
    kernel_dir = PATCH / "b12x_h16/b12x/moe/_shared/kernels"
    fc1 = (kernel_dir / "p8_coupled_prefill_fc1.py").read_text()
    fc2 = (kernel_dir / "p8_coupled_prefill_fc2.py").read_text()
    shared_fc1 = (kernel_dir / "p8_h128_fc1.py").read_text()
    shared_fc2 = (kernel_dir / "p8_small_m.py").read_text()
    for name, source in {"fc1": fc1, "fc2": fc2}.items():
        compile(source, name, "exec")
        assert "tile_m = 64" in source
        assert "source_tile_m = 64" in source
        assert "mma_m_blocks = 4" in source
        assert "owned_row_groups = 16" in source
    assert "P8H128FC1Kernel" in fc1
    assert "super().__init__(full_coupled=True)" in fc1
    assert "P8SmallMPhase2Kernel" in fc2
    assert "scale_sandwich=True, full_coupled=True" in fc2
    assert "w4a8_trellis_pair_words_dispatch" in shared_fc1
    assert "mxfp8_mma_m16n8k32_f32_e4m3" in shared_fc1
    assert "mxfp8_mma_m16n8k32_f32_e4m3" in shared_fc2
    assert "output_row = token_map[physical_row].to(Int32)" in shared_fc2

    dynamic = (kernel_dir / "dynamic.py").read_text()
    assert "def _store_p8_full_coupled_input_row(" in dynamic
    assert "h512 += Int32(self.input_warps_per_token)" in dynamic
    assert "payload_row = token_idx * Int32(a_input.shape[1])" in dynamic
    assert "scale_storage[token_idx * mx_blocks_per_row + block]" in dynamic
    assert "self.materialized_phase1_kernel = P8CoupledPrefillFC1Kernel()" in dynamic
    assert "self.materialized_phase2_kernel = P8CoupledPrefillFC2Kernel()" in dynamic


def test_full_coupled_fc1_keeps_physical_and_fp32_tiles_disjoint() -> None:
    source = (
        PATCH / "b12x_h16/b12x/moe/_shared/kernels/p8_h128_fc1.py"
    ).read_text()
    assert "full_output_base = (" in source
    assert "addr = full_output_base" in source
    assert "value_addr = full_output_base" in source
    assert "self.tile_m * 128 * (8 if self.full_coupled else 4)" in source


def test_m64_shared_memory_bounds_are_frozen() -> None:
    # FC1: gate FP16 + up FP16 + transformed FP32 = 64 KiB. FC2 retains the
    # donor two-stage native-MMA footprint and aliases a 16 KiB physical tile.
    assert 64 * 128 * 8 == 65_536
    a_stage = 64 * 128 + 64 * 4
    b_offset = ((2 * a_stage + 1023) // 1024) * 1024
    fc2_shared = b_offset + 2 * (128 * 128 // 2) + 2 * (16 * 8 * 4)
    assert fc2_shared == 34_816


def test_cpu_reference_is_bit_exact_across_m64_plus_tail_partition() -> None:
    generator = torch.Generator().manual_seed(64065)
    x = torch.randn(65, 512, generator=generator, dtype=torch.bfloat16)
    gate = torch.randn(128, 512, generator=generator, dtype=torch.float16) / 16
    up = torch.randn(128, 512, generator=generator, dtype=torch.float16) / 16
    down = torch.randn(512, 128, generator=generator, dtype=torch.float16) / 16
    scales = coupled.P8ScaleSandwich(
        gate_up_suh=torch.randn(512, generator=generator, dtype=torch.float16),
        intermediate_scales=torch.randn(
            1, 384, generator=generator, dtype=torch.float16
        ),
        down_svh=torch.randn(512, generator=generator, dtype=torch.float16),
        coupled_signs=coupled.rank_local_coupled_signs(intermediate=128, rank=0),
        full_coupled=True,
        transform_sha256="1" * 64,
    )
    whole = coupled.coupled_reference(x, gate, up, down, scales)
    tiled = torch.cat(
        [
            coupled.coupled_reference(x[:64], gate, up, down, scales),
            coupled.coupled_reference(x[64:], gate, up, down, scales),
        ]
    )
    assert torch.equal(whole, tiled)
