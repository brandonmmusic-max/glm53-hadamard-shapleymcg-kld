"""Experimental TP-local P8 procedural-MCG MoE runtime for GLM-5.3.

This is a direct device path: a K3, K4 or K5 trellis stream is decoded to E4M3 inside
the MMA kernel and physical UE8M0/32 scales are consumed by the tensor core.
It implements the frozen TP4 identity-boundary P8 contract and an opt-in M1
H128 suh/svh scale component for a GLM routed layer whose sidecar carries the
matching immutable layer identity.
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
from p8_coupled_scales import (
    COUPLED_SCHEMA as P8_COUPLED_SCHEMA,
    SCALE_NAMES,
    SCHEMA as P8_SCALE_COMPONENT_SCHEMA,
    validate_coupled_component,
    validate_scale_component,
)
from p8_smallm_schedule import P8SmallMGeometry, p8_small_m_scratch_layout, use_small_m

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.utils import get_max_active_clusters
from b12x.moe._shared.kernels.dynamic import MoEDynamicKernelBackend
from b12x.moe.fused_moe._impl import (
    _DynamicMoEW4A8Launch,
    _e8m0_scale_to_w4a8_sfb_inplace,
    _launch_dynamic_topk_sum,
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
        layer: int = 3,
        expected_design_sha256: str | None = None,
        expected_transform_sha256: str | None = None,
        topk: int = 8,
        hidden: int = 4096,
        intermediate: int = 512,
        swiglu_limit: float = 10.0,
        force_materialized: bool | None = None,
        mac_override: int | None = None,
        deterministic_output: bool = True,
        small_m_scheduler: bool = False,
        fc1_tile_n: int = 128,
        debug_capture: bool = False,
        diagnostic_raw_fc1: bool = False,
        fuse_scratch_zero: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.tp_rank = int(tp_rank)
        self.layer = int(layer)
        if not 3 <= self.layer <= 44:
            raise ValueError("P8 native layer must be in GLM routed layers 3..44")
        self.topk = int(topk)
        self.hidden = int(hidden)
        self.intermediate = int(intermediate)
        self.swiglu_limit = float(swiglu_limit)
        self.force_materialized = force_materialized
        self.mac_override = None if mac_override is None else int(mac_override)
        self.deterministic_output = bool(deterministic_output)
        # Explicit developmental opt-in. M2/M3 and prefill retain baseline
        # selection; this is not enabled through a serving environment flag.
        self.small_m_scheduler = bool(small_m_scheduler)
        self.fc1_tile_n = int(fc1_tile_n)
        self.debug_capture = bool(debug_capture)
        self.diagnostic_raw_fc1 = bool(diagnostic_raw_fc1)
        self.fuse_scratch_zero = bool(fuse_scratch_zero)
        self._scratch_layout = (
            p8_small_m_scratch_layout() if self.fuse_scratch_zero else None
        )
        self.debug_tensors = {}
        if self.fc1_tile_n not in (32, 64, 128):
            raise ValueError("FC1 tile N must be 32, 64, or 128")
        if self.fc1_tile_n != 128 and (
            not self.small_m_scheduler or self.swiglu_limit != 10.0
        ):
            raise ValueError("Narrow FC1 requires small-M and SwiGLU limit 10")
        if self.small_m_scheduler and (
            not self.deterministic_output or force_materialized is not None
            or (topk, hidden, intermediate) != (8, 4096, 512)
        ):
            raise ValueError("P8 small-M requires deterministic GLM TP4 and automatic fallback")
        if self.diagnostic_raw_fc1 and (
            not self.debug_capture
            or not self.small_m_scheduler
            or self.fc1_tile_n != 128
            or self.fuse_scratch_zero
        ):
            raise ValueError(
                "raw FC1 diagnostic requires debug M1 small-M N128 capture"
            )
        if self.mac_override is not None and self.mac_override <= 0:
            raise ValueError("mac_override must be positive")
        with safe_open(sidecar, framework="pt", device="cpu") as src:
            metadata = src.metadata() or {}
            schema = metadata.get("schema")
            source_design_sha256 = metadata.get("source_design_sha256")
            bits_text = metadata.get("bits", "")
            if bits_text not in {"3", "4", "5"}:
                raise RuntimeError(f"invalid P8 trellis rate: {bits_text!r}")
            self.trellis_bits = int(bits_text)
            base_required = {
                "layer": str(self.layer),
                "rank": str(self.tp_rank),
                "world_size": "4",
                "bits": bits_text,
                "alphabet": "e4m3",
                "scale": "ue8m0-k32",
                "law": "procedural-mcg-alpha2",
                "ldlq": "false",
            }
            identity_schema = schema in {
                    "glm53-p8-identity-mcg-tp4-rank.v1",
                    "glm53-p8-mcg-tp4-rank.v2",
            }
            scale_component_schema = schema == P8_SCALE_COMPONENT_SCHEMA
            full_coupled_schema = schema == P8_COUPLED_SCHEMA
            if (
                not (identity_schema or scale_component_schema or full_coupled_schema)
                or any(metadata.get(key) != value for key, value in base_required.items())
                or (identity_schema and metadata.get("boundary") != "identity")
            ):
                raise RuntimeError(f"invalid P8 native sidecar metadata: {metadata}")
            if schema in {
                "glm53-p8-mcg-tp4-rank.v2",
                P8_SCALE_COMPONENT_SCHEMA,
                P8_COUPLED_SCHEMA,
            }:
                if (
                    not isinstance(source_design_sha256, str)
                    or len(source_design_sha256) != 64
                    or any(char not in "0123456789abcdef" for char in source_design_sha256)
                ):
                    raise RuntimeError("v2 P8 sidecar lacks a valid source design hash")
                if (
                    expected_design_sha256 is not None
                    and source_design_sha256 != expected_design_sha256
                ):
                    raise RuntimeError("P8 sidecar does not match the expected design")
            elif expected_design_sha256 is not None:
                raise RuntimeError("historical P8 sidecars cannot satisfy a v2 design pin")
            w13 = src.get_tensor("w13_trellis")
            w2 = src.get_tensor("w2_trellis")
            w13_scale = src.get_tensor("w13_scale_ue8m0")
            w2_scale = src.get_tensor("w2_scale_ue8m0")
            scale_tensors = (
                {name: src.get_tensor(name) for name in SCALE_NAMES}
                if scale_component_schema or full_coupled_schema
                else None
            )
        self.source_design_sha256 = source_design_sha256
        experts = int(w13.shape[1])
        stream_words = 16 * self.trellis_bits
        if tuple(w13.shape) != (
            2, experts, hidden // 16, intermediate // 16, stream_words
        ):
            raise RuntimeError(f"unexpected W13 trellis shape {tuple(w13.shape)}")
        if tuple(w2.shape) != (
            experts, intermediate // 16, hidden // 16, stream_words
        ):
            raise RuntimeError(f"unexpected W2 trellis shape {tuple(w2.shape)}")
        if tuple(w13_scale.shape) != (experts, 2 * intermediate, hidden // 32):
            raise RuntimeError(f"unexpected W13 scale shape {tuple(w13_scale.shape)}")
        if tuple(w2_scale.shape) != (experts, hidden, intermediate // 32):
            raise RuntimeError(f"unexpected W2 scale shape {tuple(w2_scale.shape)}")
        self.experts = experts
        self.scale_component = None
        self.full_coupled = bool(full_coupled_schema)
        if self.full_coupled and expected_transform_sha256 is None:
            raise RuntimeError(
                "full-coupled P8 requires an externally pinned encoder transform"
            )
        if scale_tensors is not None:
            validator = (
                validate_coupled_component
                if self.full_coupled
                else validate_scale_component
            )
            validator_kwargs = {}
            if self.full_coupled:
                validator_kwargs["expected_transform_sha256"] = (
                    expected_transform_sha256
                )
            self.scale_component = validator(
                metadata, scale_tensors, layer=self.layer, rank=self.tp_rank,
                experts=experts, hidden=hidden, intermediate=intermediate,
                **validator_kwargs,
            )
            if not self.small_m_scheduler or self.fc1_tile_n != 128:
                raise RuntimeError(
                    "P8 scale component requires the M1 N128 owner path"
                )
        # The trellis storage is byte-for-byte the same size as the packed
        # E2M1 descriptor carrier expected by the inherited W4A8 launch ABI.
        # Alias it for the descriptor-only arguments instead of allocating a
        # second ~0.9 GiB of unread dummy weights per layer and TP rank.  The
        # kernel reads the procedural stream through the uint32 pointers below;
        # it never dereferences the descriptor carrier values.
        w13_stream_storage = w13.to(device=self.device).contiguous()
        w2_stream_storage = w2.to(device=self.device).contiguous()
        self.w13_stream = w13_stream_storage.view(torch.int32).reshape(-1)
        self.w2_stream = w2_stream_storage.view(torch.int32).reshape(-1)
        w13_scale = w13_scale.to(device=self.device).contiguous()
        w2_scale = w2_scale.to(device=self.device).contiguous()
        # The monolithic kernel consumes the logical [E, N, K/32] UE8M0
        # plane through the sfb_*_mx ABI slots.  The split materialized
        # kernels consume a separately repacked copy through *_sfb_rp.
        # Keep both representations: a one-byte sentinel in the logical slots
        # is an out-of-bounds scale read, not an identity scale.
        self.w13_scale_mx = w13_scale.reshape(-1)
        self.w2_scale_mx = w2_scale.reshape(-1)
        self.w13_sfb = _e8m0_scale_to_w4a8_sfb_inplace(
            w13_scale.clone(),
            weight_E=experts,
            rows=2 * intermediate,
            k_dim=hidden,
            gated_half_rows=intermediate,
        ).reshape(-1)
        self.w2_sfb = _e8m0_scale_to_w4a8_sfb_inplace(
            w2_scale.clone(),
            weight_E=experts,
            rows=hidden,
            k_dim=intermediate,
        ).reshape(-1)
        # These are descriptor carriers only; they alias the trellis storage
        # above and therefore add zero payload bytes.  Their trailing extent is
        # the PACKED row length, which is bits/8 bytes per weight: hidden // 2
        # only at K4.  Deriving it from the stored rate keeps K4 byte-identical
        # while letting K3 and K5 describe their own shorter or longer rows.
        w13_row_bytes = hidden * self.trellis_bits // 8
        w2_row_bytes = intermediate * self.trellis_bits // 8
        w13_dummy_bytes = experts * 2 * intermediate * w13_row_bytes
        w2_dummy_bytes = experts * hidden * w2_row_bytes
        self.w13_dummy = w13_stream_storage.view(torch.uint8).reshape(-1)[
            :w13_dummy_bytes
        ].reshape(
            experts, 2 * intermediate, w13_row_bytes
        )
        self.w2_dummy = w2_stream_storage.view(torch.uint8).reshape(-1)[
            :w2_dummy_bytes
        ].reshape(
            experts, hidden, w2_row_bytes
        )
        self.sentinel = torch.zeros(1, dtype=torch.uint8, device=self.device)
        self.zero_lut = torch.zeros(1, dtype=torch.uint8, device=self.device)
        # MCG never dereferences the LUT pointer. The diagnostic-only arm
        # reuses that dead ABI slot for exactly 128 FP32 trace values (512 B),
        # initialized to an all-ones NaN sentinel so partial writes fail closed.
        self.input_prequant_trace = (
            torch.full((512,), 0xFF, dtype=torch.uint8, device=self.device)
            if self.diagnostic_raw_fc1
            else self.zero_lut
        )
        self.zero_rotation = torch.zeros(1, dtype=torch.float16, device=self.device)
        self.scale_component_packed = (
            self.scale_component.packed.to(device=self.device)
            if self.scale_component is not None
            else self.zero_rotation
        )
        self.ones = torch.ones(experts, dtype=torch.float32, device=self.device)
        if self.small_m_scheduler and self.experts != 288:
            raise ValueError("P8 small-M requires 288 experts")
        # v11: the small-M owner path is compiled per stored rate (K3/K4/K5);
        # M>1 on a non-K4 layer is served row by row through that same exact
        # kernel because the grouped M64 prefill kernels remain K4-only.
        self._compiled: dict[tuple[bool, bool], _CompiledArm] = {}
        self._coupled_reducer = None

    def _compile(self, materialized: bool, small_m: bool = False) -> _CompiledArm:
        cache_key = (materialized, small_m)
        cached = self._compiled.get(cache_key)
        if cached is not None:
            return cached
        tile_m = 64 if materialized and not small_m else 16
        mac = (
            self.mac_override
            if self.mac_override is not None
            else (64 if materialized else int(get_max_active_clusters(1)))
        )
        kernel = MoEDynamicKernelBackend(
            16,
            (tile_m, 128),
            activation="silu",
            quant_recipe="w4a8_trellis",
            w4a8_repacked=True,
            num_topk=self.topk,
            trellis_bits=self.trellis_bits,
            trellis_codebook="mcg",
            trellis_scaled=True,
            trellis_identity_boundary=not self.full_coupled,
            direct_routing=small_m,
            materialize_intermediate=materialized,
            p8_small_m=small_m,
            p8_fc1_tile_n=self.fc1_tile_n if small_m else 128,
            p8_scale_sandwich=self.scale_component is not None,
            p8_full_coupled=self.full_coupled,
            share_input_across_experts=materialized,
            deterministic_output=self.deterministic_output,
            swiglu_limit=self.swiglu_limit,
        )
        if self.diagnostic_raw_fc1:
            if not (small_m and self.full_coupled):
                raise RuntimeError(
                    "raw FC1 diagnostic dispatched outside full-coupled M1"
                )
            from b12x.moe._shared.kernels.p8_h128_fc1 import (
                P8H128FC1RawCaptureKernel,
            )

            kernel.materialized_phase1_kernel = P8H128FC1RawCaptureKernel()
            kernel.p8_input_prequant_diagnostic = True
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
            ptr(
                cutlass.Float32 if self.full_coupled else cutlass.BFloat16,
                16,
            ),
            fake_ptr_i32(),
            ptr(cutlass.Float32, 16),
            1, 1, 1, 1, 1, 1, 1,
            current_cuda_stream(),
            fake_ptr_u8(),
            ptr(cutlass.Float16, 16),
            # The explicit spec IS the JIT cache key, in memory and on disk. Every
            # field the kernel is specialised on must appear here: the stored
            # trellis rate was missing, so in a mixed-rate model the first rate
            # compiled per rank was served for every layer (K3, K4 and K5 alike),
            # while single-rate closures could never see it. Version 2 retires
            # any rate-less cache entries.
            compile_spec=KernelCompileSpec.from_fields(
                "glm53.p8.native.tp4",
                2,
                ("trellis_bits", self.trellis_bits),
                ("materialized", int(materialized)),
                ("small_m_scheduler", int(small_m)),
                ("fc1_tile_n", self.fc1_tile_n if small_m else 128),
                ("experts", self.experts),
                ("hidden", self.hidden),
                ("intermediate", self.intermediate),
                ("topk", self.topk),
                ("rank", self.tp_rank),
                ("scaled", 1),
                ("identity", int(not self.full_coupled)),
                ("scale_sandwich", int(self.scale_component is not None)),
                ("full_coupled", int(self.full_coupled)),
                ("raw_fc1_diagnostic", int(self.diagnostic_raw_fc1)),
                ("input_prequant_diagnostic", int(self.diagnostic_raw_fc1)),
                ("codebook", "mcg"),
                ("deterministic_output", int(self.deterministic_output)),
            ),
            dsl_compile_options=OptLevel(2),
        )
        arm = _CompiledArm(compiled=compiled, tile_m=tile_m, materialized=materialized, mac=mac)
        self._compiled[cache_key] = arm
        return arm

    def _compile_full_coupled_reducer(self):
        if not self.full_coupled:
            raise RuntimeError("coupled reducer requested for non-coupled P8")
        if self._coupled_reducer is not None:
            return self._coupled_reducer
        from b12x.moe._shared.kernels.p8_coupled_topk import (
            P8CoupledTopKSumKernel,
        )

        reducer = P8CoupledTopKSumKernel(topk=self.topk, hidden=self.hidden)
        self._coupled_reducer = b12x_compile(
            reducer,
            make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Float32, 4, cute.AddressSpace.gmem, assumed_align=4),
            make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
            1,
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_fields(
                "glm53.p8.coupled_topk_h512",
                1,
                ("topk", self.topk),
                ("hidden", self.hidden),
                ("rank", self.tp_rank),
                ("route_dtype", "fp32"),
                ("output_dtype", "bf16"),
            ),
            dsl_compile_options=OptLevel(2),
        )
        return self._coupled_reducer

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
        # Every stored rate now has a fused grouped M64/N128 owner, so a non-K4 layer at M>1
        # runs the same native path as K4 rather than looping the M1 kernel row by row. The
        # row-by-row fallback is deliberately gone: a rate without a grouped specialization
        # must fail closed instead of silently serving at a fraction of the speed.
        if self.scale_component is not None and not self.full_coupled and m != 1:
            raise RuntimeError("P8 scale sandwich currently supports M=1 only")
        # Match the W4A8 planner's measured M16-to-M64 transition: sparse
        # decode and ordinary prefill stay monolithic; only dense routed
        # batches pay for the split materialized phase kernels.
        materialized = (
            m * self.topk >= 36 * self.experts
            if self.force_materialized is None
            else self.force_materialized
        )
        small_m = use_small_m(self.small_m_scheduler, m)
        if self.full_coupled:
            # Decode keeps the exact direct-route M1 owner. Every M>1 call is
            # forced through the one exact grouped M64/N128 implementation;
            # there is no second monolithic coupled arithmetic path.
            small_m = m == 1
            materialized = True
        materialized = materialized or small_m
        arm = self._compile(materialized, small_m=small_m)
        tile_m = arm.tile_m
        x = x.contiguous()
        flat_ids = topk_ids.to(dtype=torch.int32).contiguous().reshape(-1)
        flat_weights = topk_weights.to(dtype=torch.float32).contiguous().reshape(-1)
        physical_tiles = (
            P8SmallMGeometry().physical_tiles if small_m
            else self.experts + (m * self.topk + tile_m - 1) // tile_m
        )
        rows_padded = physical_tiles * tile_m
        gate_tile_count = ((2 * self.intermediate) // 128) // 2
        max_tasks = physical_tiles * max(gate_tile_count, 1)
        fused_scratch_zero = self.fuse_scratch_zero and small_m
        if fused_scratch_zero:
            layout = self._scratch_layout
            assert layout is not None
            # A single GPU fill initializes all original bytes plus alignment
            # padding. The views add no casts, copies, or device kernels.
            arena = torch.zeros(layout.nbytes, dtype=torch.uint8, device=self.device)
            buffers = {
                region.name: arena.narrow(0, region.offset, region.nbytes)
                .view(getattr(torch, region.dtype)).reshape(region.shape)
                for region in layout.regions
            }
            packed_a = buffers["packed_a"]
            scale_flat = buffers["scale_flat"]
            intermediate_u32 = buffers["intermediate_u32"]
            barrier_count = buffers["barrier_count"]
            barrier_epoch = buffers["barrier_epoch"]
            pair_head = buffers["pair_head"]
            producers_done = buffers["producers_done"]
            all_published = buffers["all_published"]
            task_head = buffers["task_head"]
            task_tail = buffers["task_tail"]
            task_ready = buffers["task_ready"]
            task_expert = buffers["task_expert"]
            task_m_tile = buffers["task_m_tile"]
            task_slice_begin = buffers["task_slice_begin"]
            task_slice_count = buffers["task_slice_count"]
            task_valid_rows = buffers["task_valid_rows"]
            tile_write_count = buffers["tile_write_count"]
            row_counts = buffers["row_counts"]
            expert_write_rows = buffers["expert_write_rows"]
            expert_tile_base = buffers["expert_tile_base"]
            token_map = buffers["token_map"]
            token_weights = buffers["token_weights"]
            output = buffers["output"]
        else:
            packed_a = torch.zeros(rows_padded * self.hidden, dtype=torch.uint8, device=self.device)
            scale_elements = (
                m * (self.hidden // 32)
                if self.full_coupled and materialized and not small_m
                else (self.experts + m * self.topk + 1)
                * tile_m
                * (self.hidden // 8)
            )
            scale_flat = torch.zeros(
                scale_elements, dtype=torch.uint8, device=self.device
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
        if self.diagnostic_raw_fc1:
            # Keep the ordinary allocation block exactly unchanged. Only the
            # diagnostic arm fills NaN payload sentinels and zeroes counters.
            intermediate_u32.fill_(-1)
            trace_base = rows_padded * (self.intermediate // 4)
            intermediate_u32[trace_base + 32 : trace_base + 64].zero_()
        kernel_output = (
            torch.empty(
                m * self.topk,
                self.hidden,
                dtype=torch.float32 if self.full_coupled else torch.bfloat16,
                device=self.device,
            )
            if self.deterministic_output
            else output
        )
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
            _gptr(cutlass.Uint8, self.w13_scale_mx),
            _gptr(cutlass.Uint8, self.w2_scale_mx),
            _gptr(cutlass.Uint8, self.sentinel),
            _gptr(cutlass.Uint8, self.sentinel),
            _gptr(cutlass.Uint32, self.w13_stream),
            _gptr(cutlass.Uint32, self.w13_sfb),
            _gptr(cutlass.Uint32, self.w2_stream),
            _gptr(cutlass.Uint32, self.w2_sfb),
            row_counts, expert_write_rows, expert_tile_base,
            self.ones, self.ones, self.ones, self.ones,
            _gptr(
                cutlass.Float32 if self.full_coupled else cutlass.BFloat16,
                kernel_output,
            ),
            _gptr(cutlass.Int32, token_map, 4),
            _gptr(cutlass.Float32, token_weights, 4),
            m,
            m * self.topk,
            m * self.topk if self.deterministic_output else m,
            rows_padded,
            max_tasks,
            physical_tiles,
            arm.mac,
            current_cuda_stream(),
            _gptr(cutlass.Uint8, self.input_prequant_trace),
            _gptr(cutlass.Float16, self.scale_component_packed),
        )
        if self.deterministic_output and not self.diagnostic_raw_fc1:
            if self.full_coupled:
                reducer = self._compile_full_coupled_reducer()
                reducer(
                    _gptr(cutlass.Float32, kernel_output),
                    _gptr(cutlass.Float32, flat_weights, 4),
                    _gptr(cutlass.BFloat16, output),
                    m,
                    current_cuda_stream(),
                )
            else:
                _launch_dynamic_topk_sum(
                    route_output=kernel_output,
                    output=output,
                    m=m,
                    num_topk=self.topk,
                    k=self.hidden,
                    stream=current_cuda_stream(),
                )
        if self.debug_capture:
            self.debug_tensors = {
                "packed_a": packed_a, "scale_flat": scale_flat,
                "intermediate_u32": intermediate_u32,
                "route_output": kernel_output,
                "token_map": token_map, "row_counts": row_counts,
                "expert_tile_base": expert_tile_base,
            }
            self.debug_dispatch = {"small_m": small_m, "materialized": materialized,
                                   "fused_scratch_zero": fused_scratch_zero,
                                   "fc1_tile_n": self.fc1_tile_n if small_m else 128,
                                   "tile_m": tile_m}
            if self.diagnostic_raw_fc1:
                self.debug_dispatch["diagnostic_raw_fc1"] = True
                self.debug_tensors["input_prequant_trace"] = (
                    self.input_prequant_trace
                )
        return output
