"""CPU-visible launch and memory plan for full-coupled P8 prefill.

This module does not enable prefill.  It freezes the only currently legal
shape strategy: M1 keeps the N128 decode owner; every M>1 full-coupled call is
forced through the split M64 materialized pipeline.  The values here are used
by static tests and the implementation review before any SM120 build.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class P8CoupledPrefillGeometry:
    tokens: int
    experts: int = 288
    hidden: int = 4096
    intermediate: int = 512
    topk: int = 8
    tile_m: int = 64
    tile_n: int = 128

    def __post_init__(self) -> None:
        if self.tokens <= 1:
            raise ValueError("prefill geometry requires M>1")
        if (
            self.experts,
            self.hidden,
            self.intermediate,
            self.topk,
            self.tile_m,
            self.tile_n,
        ) != (288, 4096, 512, 8, 64, 128):
            raise ValueError("full-coupled prefill is frozen to GLM TP4 M64/N128")

    @property
    def route_rows(self) -> int:
        return self.tokens * self.topk

    @property
    def physical_tiles_capacity(self) -> int:
        # Preserve the current wrapper's conservative grouped-routing bound.
        return self.experts + (self.route_rows + self.tile_m - 1) // self.tile_m

    @property
    def rows_capacity(self) -> int:
        return self.physical_tiles_capacity * self.tile_m

    @property
    def input_h512_units(self) -> int:
        return self.tokens * (self.hidden // 512)

    @property
    def fc1_tasks_capacity(self) -> int:
        return self.physical_tiles_capacity * (self.intermediate // self.tile_n)

    @property
    def fc2_tasks_capacity(self) -> int:
        return self.physical_tiles_capacity * (self.hidden // self.tile_n)

    @property
    def route_scratch_bytes(self) -> int:
        return self.route_rows * self.hidden * 4

    @property
    def output_bytes(self) -> int:
        return self.tokens * self.hidden * 2

    @property
    def packed_a_bytes_current(self) -> int:
        return self.rows_capacity * self.hidden

    @property
    def scale_flat_bytes_current(self) -> int:
        # Exact allocation in p8_native_kernel.py. This is intentionally
        # reported separately because it is far larger than the logical
        # [M,H/32] activation scale plane under shared-input materialization.
        return (
            (self.experts + self.route_rows + 1)
            * self.tile_m
            * (self.hidden // 8)
        )

    @property
    def scale_flat_bytes_logical(self) -> int:
        return materialized_scale_storage_elements(self.tokens, self.hidden)

    @property
    def intermediate_bytes(self) -> int:
        # E4M3 payload plus one UE8M0 byte per K32, per physical row.
        return self.rows_capacity * (
            self.intermediate + self.intermediate // 32
        )

    @property
    def control_upper_bound_bytes(self) -> int:
        # Integer task/control arrays are small beside tensor scratch. Keep a
        # conservative four bytes times 10 task-capacity planes.
        max_tasks = max(self.fc1_tasks_capacity, self.fc2_tasks_capacity)
        return 4 * 10 * max_tasks

    @property
    def current_wrapper_scratch_upper_bound(self) -> int:
        return (
            self.packed_a_bytes_current
            + self.scale_flat_bytes_current
            + self.intermediate_bytes
            + self.route_scratch_bytes
            + self.output_bytes
            + self.control_upper_bound_bytes
        )

    @property
    def unavoidable_transform_extra_bytes(self) -> int:
        # Versus the deterministic BF16 route scratch, full coupling widens
        # the same route-output plane to FP32. All H512 CTA exchange is shared.
        return self.route_rows * self.hidden * 2


def prefill_strategy(tokens: int) -> str:
    if tokens <= 0:
        raise ValueError("token count must be positive")
    return "m1-n128" if tokens == 1 else "forced-split-m64"


def materialized_scale_storage_elements(tokens: int, hidden: int = 4096) -> int:
    """Exact shared-input UE8M0 byte count for materialized W4A8.

    The front end writes ``token * (H/32) + block``.  FC1 reads one packed
    u32 at ``token * (H/32) + 4*k128`` for each K128 tile, covering the same
    four bytes.  Therefore both producer and consumer have inclusive maximum
    index ``tokens * (H/32) - 1``; grouped route rows never index this plane.
    """

    if tokens <= 0:
        raise ValueError("token count must be positive")
    if hidden <= 0 or hidden % 128:
        raise ValueError("hidden must be positive and divisible by 128")
    return tokens * (hidden // 32)


def materialized_scale_index_bounds(
    tokens: int, hidden: int = 4096
) -> tuple[int, int, int]:
    """Return inclusive producer/consumer maxima and required byte count."""

    elements = materialized_scale_storage_elements(tokens, hidden)
    producer_max = (tokens - 1) * (hidden // 32) + (hidden // 32 - 1)
    last_k128 = hidden // 128 - 1
    consumer_max = (tokens - 1) * (hidden // 32) + 4 * last_k128 + 3
    return producer_max, consumer_max, elements


__all__ = [
    "P8CoupledPrefillGeometry",
    "materialized_scale_index_bounds",
    "materialized_scale_storage_elements",
    "prefill_strategy",
]
