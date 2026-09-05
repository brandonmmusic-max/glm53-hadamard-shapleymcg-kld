"""Developmental P8 M1 FC2: route x N256 ownership, ordered K128 sums.

Adapted from the pinned B12X P8 w4a8_phase2.py
sha256 be317f7f76153ff5f60d2d1cb76f15dae14e6252a109b2d448e7dd8852f66194.
Only compressed stream/scales and quantized activations are stored globally.
This source has not passed device closure; it is opt-in through p8_small_m.
"""
from __future__ import annotations

import cutlass
import cutlass.cute as cute

from cutlass.cutlass_dsl import Int32, Int64, T, Uint32, dsl_user_op
from cutlass._mlir.dialects import llvm

from b12x._lib.intrinsics import (
    ld_shared_u32,
    ld_shared_v2_u32,
    shared_ptr_to_u32,
    st_shared_u32,
)
from b12x._lib.intrinsics import (
    mxfp8_mma_m16n8k32_f32_e4m3,
)
from b12x.moe._shared.kernels.w4a8_trellis_decode import (
    _w4a8_had128_quad,
    _w4a8_trellis_lane_geom,
)
from b12x.moe._shared.kernels.w4a8_mcg_decode import (
    w4a8_trellis_pair_words_dispatch,
)

from b12x.moe._shared.kernels.w4a8_phase2 import W4A8MaterializedPhase2Kernel


@dsl_user_op
def _p8_pack_f32x2_to_half2(x0, x1, *, loc=None, ip=None):
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Float32(x0).ir_value(loc=loc, ip=ip),
                cutlass.Float32(x1).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.f16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _p8_ld_shared_f16_to_f32(addr, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "{.reg .b16 tmp; ld.shared.b16 tmp, [$1]; cvt.f32.f16 $0, tmp;}",
            "=f,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class P8SmallMPhase2Kernel(W4A8MaterializedPhase2Kernel):
    """M16 arithmetic inherited from P8, with one real row per route."""

    tile_m = 16
    source_tile_m = 16
    a_payload_bytes = 16 * 128
    a_scale_bytes = 16 * 4
    a_stage_bytes = a_payload_bytes + a_scale_bytes
    a_storage_bytes = 2 * a_stage_bytes
    b_storage_offset = ((a_storage_bytes + 1023) // 1024) * 1024
    b_stage_bytes = 128 * 128 // 2
    b_storage_bytes = 2 * b_stage_bytes
    sfb_storage_offset = b_storage_offset + b_storage_bytes
    sfb_stage_bytes = 16 * 8 * 4
    shared_bytes = sfb_storage_offset + 2 * sfb_stage_bytes
    shared_words = (shared_bytes + 3) // 4

    def __init__(self, *, scale_sandwich: bool = False):
        # Deliberately no codec/arithmetic toggles in this P8-only arm.
        self.source_halves = 1
        self.deterministic_output = True
        self.w4a8_trellis = True
        self.trellis_bits = 4
        self.trellis_direct_lut = False
        self.trellis_codebook = "mcg"
        self.trellis_scaled = True
        self.trellis_identity_boundary = True
        self.scale_sandwich = bool(scale_sandwich)
        self.trellis_lut_offset = self.shared_bytes

    @cute.jit
    def _scale_down_after_h128(
        self,
        value: cutlass.Float32,
        scale_component: cute.Tensor,
        output_col: Int32,
    ) -> cutlass.Float32:
        """Apply shared down svh after output H128 and before route sum."""

        down_svh_base = Int32(4096 + 288 * 3 * 512)
        return value * scale_component[down_svh_base + output_col].to(
            cutlass.Float32
        )

    @cute.jit
    def _run_task(
        self,
        intermediate_u32: cute.Tensor,
        down_rp: cute.Tensor,
        down_sfb_rp: cute.Tensor,
        scatter_output: cute.Tensor,
        token_map: cute.Tensor,
        token_weights: cute.Tensor,
        down_alpha: cute.Tensor,
        global_scale: cute.Tensor,
        trellis_lut: cute.Tensor,
        scale_component: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        warp_idx: Int32,
        source_m_tile: Int32,
        m_half: Int32,
        expert_idx: Int32,
        output_tile: Int32,
        valid_rows: Int32,
        rows_capacity: Int32,
        intermediate_tiles: Int32,
        packed_output_tiles: Int32,
    ):
        lane = tid & Int32(31)
        q = lane >> Int32(2)
        c = lane & Int32(3)
        if cutlass.const_expr(self.w4a8_trellis):
            tr_ia, tr_ib, tr_s2 = _w4a8_trellis_lane_geom(
                lane, self.trellis_bits
            )
            trellis_lut_addr = Int64(
                smem_base + Int32(self.trellis_lut_offset)
            )
            if cutlass.const_expr(self.trellis_direct_lut):
                trellis_lut_addr = trellis_lut.iterator.toint()

        self._stage_slice(
            intermediate_u32,
            down_rp,
            down_sfb_rp,
            smem_base,
            tid,
            source_m_tile,
            m_half,
            expert_idx,
            output_tile,
            Int32(0),
            rows_capacity,
            intermediate_tiles,
            packed_output_tiles,
        )
        cute.arch.cp_async_commit_group()

        # Each warp owns M16xN32: one M16 block and four N8 fragments.
        facc = tuple(
            tuple(cute.make_rmem_tensor((4,), cutlass.Float32) for _nt in range(4))
            for _blk in range(1)
        )
        for blk in cutlass.range_constexpr(1):
            for nt in cutlass.range_constexpr(4):
                facc[blk][nt].fill(0.0)

        intermediate_slice = Int32(0)
        while intermediate_slice < intermediate_tiles:
            # The serving M1 monolithic path restarts its FC2 accumulator for
            # every K128 slice and rounds the ordered running output to BF16.
            if cutlass.const_expr(not self.scale_sandwich):
                for nt in cutlass.range_constexpr(4):
                    facc[0][nt].fill(0.0)
            stage = intermediate_slice & Int32(1)
            a_base = smem_base + stage * Int32(self.a_stage_bytes)
            sfa_base = a_base + Int32(self.a_payload_bytes)
            b_base = (
                smem_base
                + Int32(self.b_storage_offset)
                + stage * Int32(self.b_stage_bytes)
            )
            sfb_base = (
                smem_base
                + Int32(self.sfb_storage_offset)
                + stage * Int32(self.sfb_stage_bytes)
            )

            next_slice = intermediate_slice + Int32(1)
            if next_slice < intermediate_tiles:
                self._stage_slice(
                    intermediate_u32,
                    down_rp,
                    down_sfb_rp,
                    smem_base,
                    tid,
                    source_m_tile,
                    m_half,
                    expert_idx,
                    output_tile,
                    next_slice,
                    rows_capacity,
                    intermediate_tiles,
                    packed_output_tiles,
                )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(1)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()

            asc = cute.make_rmem_tensor((1,), Uint32)
            for blk in cutlass.range_constexpr(1):
                sf_row = Int32(blk * 16) + q + ((lane & Int32(1)) << Int32(3))
                asc[blk] = ld_shared_u32(sfa_base + (sf_row << Int32(2)))

            for kb in cutlass.range_constexpr(4):
                u_phys = (Int32(kb * 2) + (c >> Int32(1))) ^ q
                a_frag = cute.make_rmem_tensor((1, 4), Uint32)
                for blk in cutlass.range_constexpr(1):
                    a_lo = (
                        a_base
                        + Int32(blk * 16 * self.tile_k)
                        + (q << Int32(7))
                        + (u_phys << Int32(4))
                        + ((c & Int32(1)) << Int32(3))
                    )
                    a0, a2 = ld_shared_v2_u32(a_lo)
                    a1, a3 = ld_shared_v2_u32(a_lo + Int32(8 * self.tile_k))
                    a_frag[blk, 0] = a0
                    a_frag[blk, 1] = a1
                    a_frag[blk, 2] = a2
                    a_frag[blk, 3] = a3

                dn_b0 = cute.make_rmem_tensor((4,), Uint32)
                dn_b1 = cute.make_rmem_tensor((4,), Uint32)
                if cutlass.const_expr(self.w4a8_trellis):
                    for th in cutlass.range_constexpr(2):
                        tr_n16 = warp_idx * Int32(2) + Int32(th)
                        tr_b0 = (Int32(kb * 16) + tr_n16) * Int32(
                            8 * self.trellis_bits
                        )
                        d_lo0, d_lo1, d_hi0, d_hi1 = (
                            w4a8_trellis_pair_words_dispatch(
                                b_base,
                                lane,
                                tr_b0,
                                tr_b0 + Int32(64 * self.trellis_bits),
                                tr_ia,
                                tr_ib,
                                tr_s2,
                                self.trellis_bits,
                                trellis_lut_addr,
                                not self.trellis_direct_lut
                                and self.trellis_codebook != "mcg",
                                self.trellis_direct_lut,
                            )
                        )
                        dn_b0[th * 2] = d_lo0
                        dn_b1[th * 2] = d_lo1
                        dn_b0[th * 2 + 1] = d_hi0
                        dn_b1[th * 2 + 1] = d_hi1
                for nt in cutlass.range_constexpr(4):
                    n8 = warp_idx * Int32(4) + Int32(nt)
                    b0 = dn_b0[nt]
                    b1 = dn_b1[nt]
                    sfb_word = Uint32(0x7F7F7F7F)
                    if cutlass.const_expr(
                        not self.w4a8_trellis or self.trellis_scaled
                    ):
                        sfb_word = ld_shared_u32(
                            sfb_base + ((n8 * Int32(8) + q) << Int32(2))
                        )
                    for blk in cutlass.range_constexpr(1):
                        fragment = facc[blk][nt]
                        if cutlass.const_expr(self.w4a8_trellis):
                            d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                                fragment[0],
                                fragment[1],
                                fragment[2],
                                fragment[3],
                                a_frag[blk, 0],
                                a_frag[blk, 1],
                                a_frag[blk, 2],
                                a_frag[blk, 3],
                                b0,
                                b1,
                                asc[blk],
                                sfb_word,
                                bid_a=kb,
                                bid_b=kb,
                            )
                        fragment[0] = d0
                        fragment[1] = d1
                        fragment[2] = d2
                        fragment[3] = d3

            if cutlass.const_expr(not self.scale_sandwich):
                # Preserve the identity serving boundary exactly:
                # BF16(down_scale * slice_dot), route weight, ordered BF16 sum.
                down_scale = down_alpha[expert_idx].to(cutlass.Float32) * global_scale[
                    expert_idx
                ].to(cutlass.Float32)
                weight = token_weights[source_m_tile * Int32(16)].to(cutlass.Float32)
                col_base = output_tile * Int32(128) + warp_idx * Int32(32) + c * Int32(2)
                if q == Int32(0):
                    for nt in cutlass.range_constexpr(4):
                        col = col_base + Int32(nt * 8)
                        for element in cutlass.range_constexpr(2):
                            partial = cutlass.Float32(cutlass.BFloat16(
                                down_scale * facc[0][nt][element]
                            ))
                            previous = cutlass.Float32(0.0)
                            if intermediate_slice > Int32(0):
                                previous = cutlass.Float32(scatter_output[
                                    source_m_tile, col + Int32(element)
                                ])
                            weighted = weight * partial
                            scatter_output[source_m_tile, col + Int32(element)] = (
                                cutlass.BFloat16(previous + weighted)
                            )
            cute.arch.sync_threads()
            intermediate_slice += Int32(1)

        if cutlass.const_expr(self.scale_sandwich):
            # The complete K=512 result remains FP32 until one N128 CTA owns
            # its exact output block.  Store the physical projection as FP16,
            # run H128, then apply shared down_svh before route weighting.
            cute.arch.cp_async_wait_group(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()
            down_scale = down_alpha[expert_idx].to(cutlass.Float32) * global_scale[
                expert_idx
            ].to(cutlass.Float32)
            physical_base = smem_base
            if q == Int32(0):
                col_in_tile = warp_idx * Int32(32) + c * Int32(2)
                for nt in cutlass.range_constexpr(4):
                    col = col_in_tile + Int32(nt * 8)
                    st_shared_u32(
                        physical_base + col * Int32(2),
                        _p8_pack_f32x2_to_half2(
                            down_scale * facc[0][nt][0],
                            down_scale * facc[0][nt][1],
                        ),
                    )
            cute.arch.sync_threads()
            if warp_idx == Int32(0):
                hcol = lane * Int32(4)
                addr = physical_base + hcol * Int32(2)
                h0 = _p8_ld_shared_f16_to_f32(addr)
                h1 = _p8_ld_shared_f16_to_f32(addr + Int32(2))
                h2 = _p8_ld_shared_f16_to_f32(addr + Int32(4))
                h3 = _p8_ld_shared_f16_to_f32(addr + Int32(6))
                h0, h1, h2, h3 = _w4a8_had128_quad(
                    h0, h1, h2, h3, lane
                )
                output_col = output_tile * Int32(128) + hcol
                h0 = self._scale_down_after_h128(
                    h0, scale_component, output_col
                )
                h1 = self._scale_down_after_h128(
                    h1, scale_component, output_col + Int32(1)
                )
                h2 = self._scale_down_after_h128(
                    h2, scale_component, output_col + Int32(2)
                )
                h3 = self._scale_down_after_h128(
                    h3, scale_component, output_col + Int32(3)
                )
                weight = token_weights[source_m_tile * Int32(16)].to(
                    cutlass.Float32
                )
                scatter_output[source_m_tile, output_col] = cutlass.BFloat16(
                    weight * h0
                )
                scatter_output[
                    source_m_tile, output_col + Int32(1)
                ] = cutlass.BFloat16(weight * h1)
                scatter_output[
                    source_m_tile, output_col + Int32(2)
                ] = cutlass.BFloat16(weight * h2)
                scatter_output[
                    source_m_tile, output_col + Int32(3)
                ] = cutlass.BFloat16(weight * h3)
            cute.arch.sync_threads()

    @cute.kernel
    def kernel(
        self, intermediate_u32: cute.Tensor, down_rp: cute.Tensor,
        down_sfb_rp: cute.Tensor, scatter_output: cute.Tensor,
        token_map: cute.Tensor, token_weights: cute.Tensor,
        task_expert: cute.Tensor, task_valid_rows: cute.Tensor,
        expert_tile_base: cute.Tensor, down_alpha: cute.Tensor,
        global_scale: cute.Tensor, trellis_lut: cute.Tensor,
        scale_component: cute.Tensor,
        intermediate_tiles: cutlass.Int32, packed_output_tiles: cutlass.Int32,
    ):
        # task_expert is the original topk_ids, not grouped metadata.
        tidx, _, _ = cute.arch.thread_idx()
        _, _, bidz = cute.arch.block_idx()
        _, _, gdimz = cute.arch.grid_dim()
        tid = Int32(tidx)
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, self.shared_words], 1024
            ]

        storage = smem.allocate(Storage)
        smem_base = shared_ptr_to_u32(storage.words.data_ptr())
        rows_capacity = Int32(token_map.shape[0])
        routes = Int32(scatter_output.shape[0])
        experts = Int32(down_alpha.shape[0])
        task_slot = Int32(bidz)
        while task_slot < routes * packed_output_tiles:
            route = task_slot // packed_output_tiles
            output_pair = task_slot % packed_output_tiles
            expert = task_expert[route].to(Int32)
            if expert >= Int32(0) and expert < experts:
                # One arithmetic route x N256 task, two N128 MMA halves.
                for half in cutlass.range_constexpr(2):
                    self._run_task(
                        intermediate_u32, down_rp, down_sfb_rp, scatter_output,
                        token_map, token_weights, down_alpha, global_scale,
                        trellis_lut, scale_component, smem_base, tid, warp_idx,
                        route, Int32(0),
                        expert, output_pair * Int32(2) + Int32(half), Int32(1),
                        rows_capacity, intermediate_tiles, packed_output_tiles,
                    )
            else:
                # Invalid routes also overwrite their entire output row.
                for part in cutlass.range_constexpr(2):
                    scatter_output[route, output_pair * Int32(256) + tid
                                   + Int32(part * 128)] = cutlass.BFloat16(0.0)
            cute.arch.sync_threads()
            task_slot += Int32(gdimz)
