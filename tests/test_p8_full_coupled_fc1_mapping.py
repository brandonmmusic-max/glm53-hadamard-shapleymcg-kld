from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL = (
    ROOT
    / "runtime_patch/b12x_h16/b12x/moe/_shared/kernels/p8_h128_fc1.py"
)


def _source_registers() -> list[list[int]]:
    """Semantic output index held by each lane's four activated registers."""

    return [
        [lane * 2, lane * 2 + 1, 64 + lane * 2, 64 + lane * 2 + 1]
        for lane in range(32)
    ]


def _fixed_gather(registers: list[list[int]]) -> list[int]:
    output: list[int] = []
    for lane in range(32):
        src0 = (lane & 15) * 2
        src1 = src0 + 1
        slot = 0 if lane < 16 else 2
        output.extend(
            (
                registers[src0][slot],
                registers[src0][slot + 1],
                registers[src1][slot],
                registers[src1][slot + 1],
            )
        )
    return output


def _legacy_gather(registers: list[list[int]]) -> list[int]:
    """Emulate the pre-fix destination-selected source-register mistake."""

    output: list[int] = []
    for lane in range(32):
        for component in range(4):
            target = lane * 4 + component
            source_segment = target // 64
            source_within = target % 64
            source_lane = source_within // 2
            # The selected variable is evaluated independently in every lane;
            # shfl therefore observes the remote lane's selection, not target's.
            remote_target = source_lane * 4 + component
            remote_slot = (remote_target // 64) * 2 + (remote_target % 2)
            output.append(registers[source_lane][remote_slot])
    return output


def test_full_coupled_shuffle_gather_is_exact_bijection() -> None:
    registers = _source_registers()
    assert _fixed_gather(registers) == list(range(128))

    legacy = _legacy_gather(registers)
    assert sum(actual != expected for expected, actual in enumerate(legacy)) == 64
    assert legacy[:32] == list(range(32))
    assert legacy[32:64] == list(range(96, 128))
    assert legacy[64:96] == list(range(32))
    assert legacy[96:] == list(range(96, 128))


def test_kernel_uses_uniform_source_slots_before_local_segment_selection() -> None:
    source = KERNEL.read_text()
    full = source[source.index("Full joint FC1 owner") : source.index(
        "elif cutlass.const_expr(self.scale_sandwich):",
        source.index("Full joint FC1 owner"),
    )]
    assert "src0 = (lane & Int32(15)) << Int32(1)" in full
    assert "shuffle_sync(activated[0], src0)" in full
    assert "shuffle_sync(activated[3], src1)" in full
    assert "if lane >= Int32(16):" in full
    assert "source_slot" not in full


def test_private_fc1_scale_coordinates_cover_every_output_tile_once() -> None:
    gate_coordinates: list[int] = []
    up_coordinates: list[int] = []
    for output_tile in range(4):
        for segment in range(2):
            for lane in range(32):
                raw_idx = lane * 4
                for component in range(4):
                    index = raw_idx + component
                    projection_slot = (index // 32) & 1
                    base_col = (
                        output_tile * 128
                        + segment * 64
                        + (raw_idx // 64) * 32
                        + (raw_idx % 32)
                    )
                    coordinate = base_col + component
                    (gate_coordinates if projection_slot == 0 else up_coordinates).append(
                        coordinate
                    )

    assert sorted(gate_coordinates) == list(range(512))
    assert sorted(up_coordinates) == list(range(512))

    source = KERNEL.read_text()
    full = source[source.index("Full joint FC1 owner") : source.index(
        "elif cutlass.const_expr(self.scale_sandwich):",
        source.index("Full joint FC1 owner"),
    )]
    base_col = full[full.index("base_col = (") : full.index(")\n", full.index("base_col = (")) + 2]
    assert "output_tile * Int32(128)" in base_col
