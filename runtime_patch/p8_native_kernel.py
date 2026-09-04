"""Experimental TP-local P8 procedural-MCG MoE runtime for GLM-5.3.

This is a direct device path: the K4 trellis stream is decoded to E4M3 inside
the MMA kernel and physical UE8M0/32 scales are consumed by the tensor core.
It intentionally implements only the frozen TP4, layer-3, identity-boundary
development contract used by the kernel-versus-pseudoquant KLD gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cutlass
import cutlass.cute as cute
import torch
from cutlass.base_dsl.compiler import OptLevel
from cutlass.cute.runtime import make_ptr
from safetensors import safe_open

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x.moe._shared.kernels.dynamic import MoEDynamicKernelBackend
from b12x.moe.fused_moe._impl import (
    _DynamicMoEW4A8Launch,
    _e8m0_scale_to_w4a8_sfb_inplace,
    current_cuda_stream,
)


def _gptr(dtype, tensor: torch.Tensor, align: int = 16):
    return make_ptr(
        dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=align
    )


def _fake_i32(shape: tuple[int, ...]):
    return cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, shape, assumed_align=4
    )


def _fake_f32(shape: tuple[int, ...]):
    return cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, shape, assumed_align=16
    )


@dataclass
class _CompiledArm:
    compiled: object
    tile_m: int
    materialized: bool
    mac: int


class P8NativeTPMoE:
    """Own one TP rank's physical P8 payload and launch compiled MoE kernels."""

    def __init__(
        self,
        sidecar: Path,
        *,
        device: torch.device,
        tp_rank: int,
        topk: int = 8,
        hidden: int = 4096,
        intermediate: int = 512,
        swiglu_limit: float = 10.0,
    ) -> None:
        self.device = torch.device(device)
        self.tp_rank = int(tp_rank)
        self.topk = int(topk)
        self.hidden = int(hidden)
        self.intermediate = int(intermediate)
        self.swiglu_limit = float(swiglu_limit)
        with safe_open(sidecar, framework="pt", device="cpu") as src:
            metadata = src.metadata() or {}
            required = {
                "schema": "glm53-p8-identity-mcg-tp4-rank.v1",
                "layer": "3",
                "rank": str(self.tp_rank),
                "world_size": "4",
                "bits": "4",
                "alphabet": "e4m3",
                "scale": "ue8m0-k32",
                "law": "procedural-mcg-alpha2",
                "boundary": "identity",
                "ldlq": "false",
            }
            if any(metadata.get(key) != value for key, value in required.items()):
                raise RuntimeError(f"invalid P8 native sidecar metadata: {metadata}")
            w13 = src.get_tensor("w13_trellis")
            w2 = src.get_tensor("w2_trellis")
            w13_scale = src.get_tensor("w13_scale_ue8m0")
            w2_scale = src.get_tensor("w2_scale_ue8m0")
        experts = int(w13.shape[1])
        if tuple(w13.shape) != (2, experts, hidden // 16, intermediate // 16, 64):
            raise RuntimeError(f"unexpected W13 trellis shape {tuple(w13.shape)}")
        if tuple(w2.shape) != (experts, intermediate // 16, hidden // 16, 64):
            raise RuntimeError(f"unexpected W2 trellis shape {tuple(w2.shape)}")
        if tuple(w13_scale.shape) != (experts, 2 * intermediate, hidden // 32):
            raise RuntimeError(f"unexpected W13 scale shape {tuple(w13_scale.shape)}")
        if tuple(w2_scale.shape) != (experts, hidden, intermediate // 32):
            raise RuntimeError(f"unexpected W2 scale shape {tuple(w2_scale.shape)}")
        self.experts = experts
        self.w13_stream = w13.to(device=self.device).contiguous().view(torch.int32).reshape(-1)
        self.w2_stream = w2.to(device=self.device).contiguous().view(torch.int32).reshape(-1)
        w13_scale = w13_scale.to(device=self.device).contiguous()
        w2_scale = w2_scale.to(device=self.device).contiguous()
        self.w13_sfb = _e8m0_scale_to_w4a8_sfb_inplace(
            w13_scale,
            weight_E=experts,
            rows=2 * intermediate,
            k_dim=hidden,
            gated_half_rows=intermediate,
        ).reshape(-1)
        self.w2_sfb = _e8m0_scale_to_w4a8_sfb_inplace(
            w2_scale,
            weight_E=experts,
            rows=hidden,
            k_dim=intermediate,
        ).reshape(-1)
        # These are descriptor carriers only; trellis staging never reads them.
        self.w13_dummy = torch.zeros(
            experts, 2 * intermediate, hidden // 2, dtype=torch.uint8, device=self.device
        )
        self.w2_dummy = torch.zeros(
            experts, hidden, intermediate // 2, dtype=torch.uint8, device=self.device
        )
        self.sentinel = torch.zeros(1, dtype=torch.uint8, device=self.device)
        self.zero_lut = torch.zeros(1, dtype=torch.uint8, device=self.device)
        self.zero_rotation = torch.zeros(1, dtype=torch.float16, device=self.device)
        self.ones = torch.ones(experts, dtype=torch.float32, device=self.device)
        self._compiled: dict[bool, _CompiledArm] = {}

    def _compile(self, materialized: bool) -> _CompiledArm:
        cached = self._compiled.get(materialized)
        if cached is not None:
            return cached
        tile_m = 64 if materialized else 16
        mac = 64 if materialized else 4
        kernel = MoEDynamicKernelBackend(
            16,
            (tile_m, 128),
            activation="silu",
            quant_recipe="w4a8_trellis",
            w4a8_repacked=True,
            num_topk=self.topk,
            trellis_bits=4,
            trellis_codebook="mcg",
            trellis_scaled=True,
            trellis_identity_boundary=True,
            direct_routing=False,
            materialize_intermediate=materialized,
            share_input_across_experts=materialized,
            swiglu_limit=self.swiglu_limit,
        )
        launch = _DynamicMoEW4A8Launch(
            kernel,
            k=self.hidden,
            n=self.intermediate,
            w1_n=2 * self.intermediate,
            num_topk=self.topk,
        )

        def ptr(dtype, address: int, align: int = 16):
            return make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=align)

        def fake_ptr_u8():
            return ptr(cutlass.Uint8, 16)

        def fake_ptr_i32():
            return ptr(cutlass.Int32, 4, 4)

        def fake_ptr_u32():
            return ptr(cutlass.Uint32, 16)

        b_w13_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Float4E2M1FN,
            (2 * self.intermediate, self.hidden, self.experts),
            stride_order=(1, 0, 2),
            assumed_align=16,
        )
        b_w2_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Float4E2M1FN,
            (self.hidden, self.intermediate, self.experts),
            stride_order=(1, 0, 2),
            assumed_align=16,
        )
        compiled = b12x_compile(
            launch,
            ptr(cutlass.BFloat16, 16),
            fake_ptr_i32(),
            ptr(cutlass.Float32, 4, 4),
            ptr(cutlass.Float4E2M1FN, 16),
            ptr(cutlass.Float8E4M3FN, 16),
            fake_ptr_u8(),
            fake_ptr_u8(),
            fake_ptr_u32(),
            _fake_i32((1,)), _fake_i32((1,)), _fake_i32((1,)),
            _fake_i32((1,)), _fake_i32((1,)), _fake_i32((1,)), _fake_i32((1,)),
            fake_ptr_i32(), fake_ptr_i32(), fake_ptr_i32(),
            fake_ptr_i32(), fake_ptr_i32(), fake_ptr_i32(), fake_ptr_i32(),
            b_w13_fake,
            ptr(cutlass.Float8E4M3FN, 16),
            b_w2_fake,
            ptr(cutlass.Float8E4M3FN, 16),
            fake_ptr_u8(), fake_ptr_u8(), fake_ptr_u8(), fake_ptr_u8(),
            fake_ptr_u32(), fake_ptr_u32(), fake_ptr_u32(), fake_ptr_u32(),
            _fake_i32((self.experts,)),
            _fake_i32((self.experts,)),
            _fake_i32((self.experts + 1,)),
            _fake_f32((self.experts,)), _fake_f32((self.experts,)),
            _fake_f32((self.experts,)), _fake_f32((self.experts,)),
            ptr(cutlass.BFloat16, 16),
            fake_ptr_i32(),
            ptr(cutlass.Float32, 16),
            1, 1, 1, 1, 1, 1, 1,
            current_cuda_stream(),
            fake_ptr_u8(),
            ptr(cutlass.Float16, 16),
            compile_spec=KernelCompileSpec.from_fields(
                "glm53.p8.native.tp4",
                1,
                ("materialized", int(materialized)),
                ("experts", self.experts),
                ("hidden", self.hidden),
                ("intermediate", self.intermediate),
                ("topk", self.topk),
                ("rank", self.tp_rank),
                ("scaled", 1),
                ("identity", 1),
                ("codebook", "mcg"),
            ),
            dsl_compile_options=OptLevel(2),
        )
        arm = _CompiledArm(compiled=compiled, tile_m=tile_m, materialized=materialized, mac=mac)
        self._compiled[materialized] = arm
        return arm

    @torch.inference_mode()
    def __call__(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        if x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] != self.hidden:
            raise RuntimeError(f"P8 native input contract mismatch: {x.dtype} {tuple(x.shape)}")
        m = int(x.shape[0])
        if tuple(topk_ids.shape) != (m, self.topk) or tuple(topk_weights.shape) != (m, self.topk):
            raise RuntimeError("P8 native routing shape mismatch")
        # The E=288 small-M monolithic specialization is not numerically
        # closed yet (the first M3 diagnostic emitted non-finite values).
        # Use the independently closed materialized path for the KLD gate;
        # decode specialization remains a separate speed task.
        materialized = True
        arm = self._compile(materialized)
        tile_m = arm.tile_m
        x = x.contiguous()
        flat_ids = topk_ids.to(dtype=torch.int32).contiguous().reshape(-1)
        flat_weights = topk_weights.to(dtype=torch.float32).contiguous().reshape(-1)
        physical_tiles = self.experts + (m * self.topk + tile_m - 1) // tile_m
        rows_padded = physical_tiles * tile_m
        gate_tile_count = ((2 * self.intermediate) // 128) // 2
        max_tasks = physical_tiles * max(gate_tile_count, 1)
        packed_a = torch.zeros(rows_padded * self.hidden, dtype=torch.uint8, device=self.device)
        scale_flat = torch.zeros(
            (self.experts + m * self.topk + 1) * tile_m * (self.hidden // 8),
            dtype=torch.uint8,
            device=self.device,
        )
        intermediate_u32 = torch.zeros(
            rows_padded * (self.intermediate + self.intermediate // 32) // 4,
            dtype=torch.int32,
            device=self.device,
        )

        def z1():
            return torch.zeros(1, dtype=torch.int32, device=self.device)

        def ztask():
            return torch.zeros(max_tasks, dtype=torch.int32, device=self.device)

        barrier_count, barrier_epoch = z1(), z1()
        pair_head, producers_done, all_published = z1(), z1(), z1()
        task_head, task_tail = z1(), z1()
        task_ready, task_expert, task_m_tile = ztask(), ztask(), ztask()
        task_slice_begin, task_slice_count, task_valid_rows = ztask(), ztask(), ztask()
        tile_write_count = torch.zeros(physical_tiles, dtype=torch.int32, device=self.device)
        row_counts = torch.zeros(self.experts, dtype=torch.int32, device=self.device)
        expert_write_rows = torch.zeros(self.experts, dtype=torch.int32, device=self.device)
        expert_tile_base = torch.zeros(self.experts + 1, dtype=torch.int32, device=self.device)
        token_map = torch.zeros(rows_padded, dtype=torch.int32, device=self.device)
        token_weights = torch.zeros(rows_padded, dtype=torch.float32, device=self.device)
        output = torch.zeros(m, self.hidden, dtype=torch.bfloat16, device=self.device)
        arm.compiled(
            _gptr(cutlass.BFloat16, x),
            _gptr(cutlass.Int32, flat_ids, 4),
            _gptr(cutlass.Float32, flat_weights, 4),
            _gptr(cutlass.Float4E2M1FN, packed_a),
            _gptr(cutlass.Float8E4M3FN, scale_flat),
            _gptr(cutlass.Uint8, packed_a),
            _gptr(cutlass.Uint8, scale_flat),
            _gptr(cutlass.Uint32, intermediate_u32),
            barrier_count, barrier_epoch, pair_head, producers_done, all_published,
            task_head, task_tail,
            _gptr(cutlass.Int32, task_ready, 4),
            _gptr(cutlass.Int32, task_expert, 4),
            _gptr(cutlass.Int32, task_m_tile, 4),
            _gptr(cutlass.Int32, task_slice_begin, 4),
            _gptr(cutlass.Int32, task_slice_count, 4),
            _gptr(cutlass.Int32, task_valid_rows, 4),
            _gptr(cutlass.Int32, tile_write_count, 4),
            self.w13_dummy,
            _gptr(cutlass.Float8E4M3FN, self.sentinel),
            self.w2_dummy,
            _gptr(cutlass.Float8E4M3FN, self.sentinel),
            _gptr(cutlass.Uint8, self.sentinel),
            _gptr(cutlass.Uint8, self.sentinel),
            _gptr(cutlass.Uint8, self.sentinel),
            _gptr(cutlass.Uint8, self.sentinel),
            _gptr(cutlass.Uint32, self.w13_stream),
            _gptr(cutlass.Uint32, self.w13_sfb),
            _gptr(cutlass.Uint32, self.w2_stream),
            _gptr(cutlass.Uint32, self.w2_sfb),
            row_counts, expert_write_rows, expert_tile_base,
            self.ones, self.ones, self.ones, self.ones,
            _gptr(cutlass.BFloat16, output),
            _gptr(cutlass.Int32, token_map, 4),
            _gptr(cutlass.Float32, token_weights, 4),
            m,
            m * self.topk,
            m,
            rows_padded,
            max_tasks,
            physical_tiles,
            arm.mac,
            current_cuda_stream(),
            _gptr(cutlass.Uint8, self.zero_lut),
            _gptr(cutlass.Float16, self.zero_rotation),
        )
        return output
