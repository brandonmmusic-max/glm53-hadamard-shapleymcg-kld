"""Developmental P8 M1 FC2: route x N256 ownership, ordered K128 sums.

Adapted from the pinned B12X P8 w4a8_phase2.py
sha256 be317f7f76153ff5f60d2d1cb76f15dae14e6252a109b2d448e7dd8852f66194.
Only compressed stream/scales and quantized activations are stored globally.
This source has not passed device closure; it is opt-in through p8_small_m.
"""
from __future__ import annotations

import cutlass
import cutlass.cute as cute

from cutlass.cutlass_dsl import Int32, Int64, Uint32

from b12x._lib.intrinsics import (
    cp_async4_shared_global,
    cp_async_u32_shared_global,
    get_ptr_as_int64,
    ld_shared_u32,
    ld_shared_v2_u32,
    shared_ptr_to_u32,
)
from b12x._lib.intrinsics import (
    mxfp8_mma_m16n8k32_f32_e4m3,
)
from b12x.moe._shared.kernels.w4a8_trellis_decode import (
    _w4a8_stage_trellis_b_tile,
    _w4a8_trellis_lane_geom,
)
from b12x.moe._shared.kernels.w4a8_mcg_decode import (
    w4a8_trellis_pair_words_dispatch,
)

from b12x.moe._shared.kernels.w4a8_phase2 import W4A8MaterializedPhase2Kernel


class P8SmallMPhase2Kernel(W4A8MaterializedPhase2Kernel):
    """M16 arithmetic inherited from P8, with one real row per route."""

    tile_m = 16
    source_tile_m = 16
    a_payload_bytes = 16 * 128
    a_scale_bytes = 16 * 4
    a_stage_bytes = a_payload_bytes + a_scale_bytes
    a_storage_bytes = 2 * a_stage_bytes
    b_storage_offset = ((a_storage_bytes + 1023) // 1024) * 1024
    # Keep both N128 halves of one physical N256 weight tile resident.  The
    # previous arm called the inherited N128 task twice and therefore loaded
    # the same activation tile twice for every K128 slice.
    b_half_bytes = 128 * 128 // 2
    b_stage_bytes = 2 * b_half_bytes
    b_storage_bytes = 2 * b_stage_bytes
    sfb_storage_offset = b_storage_offset + b_storage_bytes
    sfb_half_bytes = 16 * 8 * 4
    sfb_stage_bytes = 2 * sfb_half_bytes
    shared_bytes = sfb_storage_offset + 2 * sfb_stage_bytes
    shared_words = (shared_bytes + 3) // 4

    def __init__(self):
        # Deliberately no codec/arithmetic toggles in this P8-only arm.
        self.source_halves = 1
        self.deterministic_output = True
        self.w4a8_trellis = True
        self.trellis_bits = 4
        self.trellis_direct_lut = False
        self.trellis_codebook = "mcg"
        self.trellis_scaled = True
        self.trellis_identity_boundary = True
        self.trellis_lut_offset = self.shared_bytes

    @cute.jit
    def _stage_pair_slice(
        self,
        intermediate_u32: cute.Tensor,
        down_rp: cute.Tensor,
        down_sfb_rp: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        source_m_tile: Int32,
        expert_idx: Int32,
        output_pair: Int32,
        intermediate_slice: Int32,
        rows_capacity: Int32,
        intermediate_tiles: Int32,
        packed_output_tiles: Int32,
    ):
        """Stage one route's A once and both N128 halves of an N256 tile."""

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

        words_per_row = intermediate_tiles * Int32(32)
        physical_row_base = source_m_tile * Int32(self.source_tile_m)

        # M16xK128 E4M3 activation payload.  Only row zero is live for M1;
        # the remaining rows are the caller-owned zero padding used by QMMA.
        for i in cutlass.range_constexpr(
            (self.tile_m * 8 + self.threads_per_cta - 1)
            // self.threads_per_cta
        ):
            idx = tid + Int32(i * self.threads_per_cta)
            if idx < Int32(self.tile_m * 8):
                row = idx >> Int32(3)
                vec = idx & Int32(7)
                physical_vec = vec ^ (row & Int32(7))
                src_word = (
                    (physical_row_base + row) * words_per_row
                    + intermediate_slice * Int32(32)
                    + (vec << Int32(2))
                )
                cp_async4_shared_global(
                    a_base
                    + row * Int32(self.tile_k)
                    + (physical_vec << Int32(4)),
                    get_ptr_as_int64(intermediate_u32, src_word),
                )

        # One packed UE8M0 K/32 scale word per activation row.
        if tid < Int32(self.tile_m):
            sf_src = (
                rows_capacity * words_per_row
                + intermediate_slice * rows_capacity
                + physical_row_base
                + tid
            )
            cp_async_u32_shared_global(
                sfa_base + (tid << Int32(2)),
                get_ptr_as_int64(intermediate_u32, sf_src),
            )

        # The physical stream is expert-major [K16][N16].  Preserve the
        # inherited K128xN128 staging layout independently for each half so the
        # procedural sliding-window decoder remains byte-for-byte unchanged.
        tr_n16_cnt = packed_output_tiles * Int32(16)
        tr_k16_stride = tr_n16_cnt * Int32(8 * self.trellis_bits)
        tr_eu = Int64(intermediate_tiles * Int32(8)) * Int64(tr_k16_stride)
        tr_base = (
            Int64(expert_idx) * tr_eu
            + Int64(intermediate_slice * Int32(8)) * Int64(tr_k16_stride)
            + Int64(output_pair * Int32(16))
            * Int64(8 * self.trellis_bits)
        )
        _w4a8_stage_trellis_b_tile(
            down_rp,
            b_base,
            tr_base,
            tr_k16_stride,
            self.trellis_bits,
            tid,
            self.threads_per_cta,
            8,
        )
        _w4a8_stage_trellis_b_tile(
            down_rp,
            b_base + Int32(self.b_half_bytes),
            tr_base + Int64(8 * 8 * self.trellis_bits),
            tr_k16_stride,
            self.trellis_bits,
            tid,
            self.threads_per_cta,
            8,
        )

        # Repacked physical UE8M0 scales are already N256-tile-major.  Copy the
        # complete tile rather than compacting and then reloading each N128 half.
        b_tile = (
            expert_idx * packed_output_tiles + output_pair
        ) * intermediate_tiles + intermediate_slice
        sfb_word_base = Int64(b_tile) * Int64(256)
        for i in cutlass.range_constexpr(
            ((32 * 8) // 4 + self.threads_per_cta - 1)
            // self.threads_per_cta
        ):
            idx = tid + Int32(i * self.threads_per_cta)
            if idx < Int32((32 * 8) // 4):
                cp_async4_shared_global(
                    sfb_base + (idx << Int32(4)),
                    get_ptr_as_int64(
                        down_sfb_rp,
                        sfb_word_base + Int64(idx * 4),
                    ),
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
        smem_base: Int32,
        tid: Int32,
        warp_idx: Int32,
        source_m_tile: Int32,
        expert_idx: Int32,
        output_pair: Int32,
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

        self._stage_pair_slice(
            intermediate_u32,
            down_rp,
            down_sfb_rp,
            smem_base,
            tid,
            source_m_tile,
            expert_idx,
            output_pair,
            Int32(0),
            rows_capacity,
            intermediate_tiles,
            packed_output_tiles,
        )
        cute.arch.cp_async_commit_group()

        # Each warp owns M16xN64: one M16 block and eight N8 fragments.  This
        # is the B12X N256 task structure specialized to one live M row.
        facc = tuple(
            tuple(cute.make_rmem_tensor((4,), cutlass.Float32) for _nt in range(8))
            for _blk in range(1)
        )
        for blk in cutlass.range_constexpr(1):
            for nt in cutlass.range_constexpr(8):
                facc[blk][nt].fill(0.0)

        intermediate_slice = Int32(0)
        while intermediate_slice < intermediate_tiles:
            # The serving M1 monolithic path restarts its FC2 accumulator for
            # every K128 slice and rounds the ordered running output to BF16.
            for nt in cutlass.range_constexpr(8):
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
                self._stage_pair_slice(
                    intermediate_u32,
                    down_rp,
                    down_sfb_rp,
                    smem_base,
                    tid,
                    source_m_tile,
                    expert_idx,
                    output_pair,
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

                dn_b0 = cute.make_rmem_tensor((8,), Uint32)
                dn_b1 = cute.make_rmem_tensor((8,), Uint32)
                if cutlass.const_expr(self.w4a8_trellis):
                    for th in cutlass.range_constexpr(4):
                        tr_n16 = warp_idx * Int32(4) + Int32(th)
                        tr_half = tr_n16 >> Int32(3)
                        tr_n16_local = tr_n16 & Int32(7)
                        tr_b_base = b_base + tr_half * Int32(self.b_half_bytes)
                        tr_b0 = (Int32(kb * 16) + tr_n16_local) * Int32(
                            8 * self.trellis_bits
                        )
                        d_lo0, d_lo1, d_hi0, d_hi1 = (
                            w4a8_trellis_pair_words_dispatch(
                                tr_b_base,
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
                for nt in cutlass.range_constexpr(8):
                    n8 = warp_idx * Int32(8) + Int32(nt)
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

            # Preserve the monolithic serving boundary exactly:
            # BF16(down_scale * slice_dot), then route weighting, then the
            # ordered BF16 running sum.
            down_scale = down_alpha[expert_idx].to(cutlass.Float32) * global_scale[
                expert_idx
            ].to(cutlass.Float32)
            weight = token_weights[source_m_tile * Int32(16)].to(cutlass.Float32)
            col_base = (
                output_pair * Int32(256)
                + warp_idx * Int32(64)
                + c * Int32(2)
            )
            if q == Int32(0):
                for nt in cutlass.range_constexpr(8):
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

    @cute.kernel
    def kernel(
        self, intermediate_u32: cute.Tensor, down_rp: cute.Tensor,
        down_sfb_rp: cute.Tensor, scatter_output: cute.Tensor,
        token_map: cute.Tensor, token_weights: cute.Tensor,
        task_expert: cute.Tensor, task_valid_rows: cute.Tensor,
        expert_tile_base: cute.Tensor, down_alpha: cute.Tensor,
        global_scale: cute.Tensor, trellis_lut: cute.Tensor,
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
                # One arithmetic route x N256 task. A is staged once per K128
                # slice and shared by both logical N128 halves.
                self._run_task(
                    intermediate_u32, down_rp, down_sfb_rp, scatter_output,
                    token_map, token_weights, down_alpha, global_scale,
                    trellis_lut, smem_base, tid, warp_idx, route, expert,
                    output_pair, Int32(1), rows_capacity,
                    intermediate_tiles, packed_output_tiles,
                )
            else:
                # Invalid routes also overwrite their entire output row.
                for part in cutlass.range_constexpr(2):
                    scatter_output[route, output_pair * Int32(256) + tid
                                   + Int32(part * 128)] = cutlass.BFloat16(0.0)
            cute.arch.sync_threads()
            task_slot += Int32(gdimz)
