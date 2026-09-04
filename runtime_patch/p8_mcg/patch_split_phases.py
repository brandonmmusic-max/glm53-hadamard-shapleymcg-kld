"""Fail-closed source transformer for the pinned B12X split P8 kernels."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXPECTED = {
    "w4a8_phase1.py": "cb72c50dab933ee866103caf7c32f1a6cfb15df991fcdd8b753ac765551946b1",
    "w4a8_phase2.py": "ce31085628a8423468a453275f18c3322f8029d35afccb35f5e7060cbab11a7e",
}


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"{label}: expected exactly one source match, got {text.count(old)}")
    return text.replace(old, new, 1)


def patch(path: Path) -> None:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED[path.name]:
        raise RuntimeError(f"{path}: pinned SHA mismatch: {digest}")
    text = raw.decode()
    text = replace_once(
        text,
        "    _w4a8_trellis_pair_words_both,\n",
        "",
        f"{path.name} remove SQG-only import",
    )
    anchor = ")\n\n\nclass W4A8Materialized"
    text = replace_once(
        text,
        anchor,
        ")\nfrom b12x.moe._shared.kernels.w4a8_mcg_decode import (\n"
        "    w4a8_trellis_pair_words_dispatch,\n"
        ")\n\n\nclass W4A8Materialized",
        f"{path.name} add dispatch import",
    )
    text = text.replace(
        "_w4a8_trellis_pair_words_both(",
        "w4a8_trellis_pair_words_dispatch(",
    )
    expected_calls = 2 if path.name == "w4a8_phase1.py" else 1
    if text.count("w4a8_trellis_pair_words_dispatch(") != expected_calls:
        raise RuntimeError(f"{path.name}: dispatch call-count mismatch")
    text = replace_once(
        text,
        "        trellis_direct_lut: bool = False,\n    ):\n",
        "        trellis_direct_lut: bool = False,\n"
        "        trellis_codebook: str = \"none\",\n"
        "        trellis_scaled: bool = False,\n"
        "        trellis_identity_boundary: bool = False,\n"
        "    ):\n",
        f"{path.name} constructor argument",
    )
    old = """        self.trellis_direct_lut = bool(trellis_direct_lut) and self.w4a8_trellis
        if self.w4a8_trellis:
            self.trellis_lut_offset = self.shared_bytes
            if not self.trellis_direct_lut:
                self.shared_words = (self.shared_bytes + 4096 + 3) // 4
"""
    new = """        self.trellis_direct_lut = bool(trellis_direct_lut) and self.w4a8_trellis
        if trellis_codebook not in {"none", "sqg-xor-cheb-t12", "mcg"}:
            raise ValueError(f"unsupported split trellis codebook {trellis_codebook!r}")
        self.trellis_codebook = str(trellis_codebook)
        if trellis_scaled and not self.w4a8_trellis:
            raise ValueError("trellis_scaled requires a trellis payload")
        self.trellis_scaled = bool(trellis_scaled)
        if trellis_identity_boundary and not self.w4a8_trellis:
            raise ValueError("trellis_identity_boundary requires a trellis payload")
        if trellis_identity_boundary and getattr(self, "trellis_coupled", False):
            raise ValueError("identity and coupled trellis boundaries are exclusive")
        self.trellis_identity_boundary = bool(trellis_identity_boundary)
        if self.trellis_codebook == "mcg" and self.trellis_bits == 2:
            raise ValueError("P8 MCG split kernels support K3/K4, not K2")
        if self.trellis_codebook == "mcg" and self.trellis_direct_lut:
            raise ValueError("procedural MCG does not use a direct LUT")
        if self.w4a8_trellis:
            self.trellis_lut_offset = self.shared_bytes
            if self.trellis_codebook != "mcg" and not self.trellis_direct_lut:
                self.shared_words = (self.shared_bytes + 4096 + 3) // 4
"""
    text = replace_once(text, old, new, f"{path.name} codebook policy")
    if path.name == "w4a8_phase1.py":
        text = replace_once(
            text,
            "        if cutlass.const_expr(self.w4a8_trellis):\n"
            "            # Trellis activation boundary:",
            "        if cutlass.const_expr(\n"
            "            self.w4a8_trellis and not self.trellis_identity_boundary\n"
            "        ):\n"
            "            # Trellis activation boundary:",
            "phase1 identity activation boundary",
        )
        identity_anchor = """                    st_shared_u32(row_addr, pack_f32x2_to_bfloat2(o0, o1))
                    st_shared_u32(row_addr + Int32(4), pack_f32x2_to_bfloat2(o2, o3))
        else:
            for nt in cutlass.range_constexpr(4):
"""
        identity_epilogue = """                    st_shared_u32(row_addr, pack_f32x2_to_bfloat2(o0, o1))
                    st_shared_u32(row_addr + Int32(4), pack_f32x2_to_bfloat2(o2, o3))
        elif cutlass.const_expr(
            self.w4a8_trellis and self.trellis_identity_boundary
        ):
            # The trellis accumulator fragments use the same shared-memory
            # coordinate contract as the H128 boundary. Materialize both raw
            # projections first, then pair gate/up by their natural feature
            # coordinate before applying the ordinary activation.
            up_tile_base = smem_base + Int32(self.tile_m * self.tile_n * 2)
            for nt in cutlass.range_constexpr(4):
                col = col_base + Int32(nt * 8)
                for blk in cutlass.range_constexpr(4):
                    gate_fragment = gate_acc[blk][nt]
                    up_fragment = up_acc[blk][nt]
                    row_lo = Int32(blk * 16) + q
                    row_hi = row_lo + Int32(8)
                    st_shared_u32(
                        epilogue_base
                        + (row_lo * Int32(self.tile_n) + col) * Int32(2),
                        pack_f32x2_to_bfloat2(
                            alpha_value * gate_fragment[0],
                            alpha_value * gate_fragment[1],
                        ),
                    )
                    st_shared_u32(
                        epilogue_base
                        + (row_hi * Int32(self.tile_n) + col) * Int32(2),
                        pack_f32x2_to_bfloat2(
                            alpha_value * gate_fragment[2],
                            alpha_value * gate_fragment[3],
                        ),
                    )
                    st_shared_u32(
                        up_tile_base
                        + (row_lo * Int32(self.tile_n) + col) * Int32(2),
                        pack_f32x2_to_bfloat2(
                            alpha_value * up_fragment[0],
                            alpha_value * up_fragment[1],
                        ),
                    )
                    st_shared_u32(
                        up_tile_base
                        + (row_hi * Int32(self.tile_n) + col) * Int32(2),
                        pack_f32x2_to_bfloat2(
                            alpha_value * up_fragment[2],
                            alpha_value * up_fragment[3],
                        ),
                    )
            cute.arch.sync_threads()
            unit_alpha = cutlass.Float32(1.0)
            for row_it in cutlass.range_constexpr(self.tile_m // self.num_warps):
                row = warp_idx * Int32(self.tile_m // self.num_warps) + Int32(row_it)
                row_addr = (
                    epilogue_base
                    + (row * Int32(self.tile_n) + lane * Int32(4)) * Int32(2)
                )
                up_addr = (
                    up_tile_base
                    + (row * Int32(self.tile_n) + lane * Int32(4)) * Int32(2)
                )
                g0 = ld_shared_bf16_to_f32(row_addr)
                g1 = ld_shared_bf16_to_f32(row_addr + Int32(2))
                g2 = ld_shared_bf16_to_f32(row_addr + Int32(4))
                g3 = ld_shared_bf16_to_f32(row_addr + Int32(6))
                u0 = ld_shared_bf16_to_f32(up_addr)
                u1 = ld_shared_bf16_to_f32(up_addr + Int32(2))
                u2 = ld_shared_bf16_to_f32(up_addr + Int32(4))
                u3 = ld_shared_bf16_to_f32(up_addr + Int32(6))
                a0 = self._activated_value(g0, u0, unit_alpha)
                a1 = self._activated_value(g1, u1, unit_alpha)
                a2 = self._activated_value(g2, u2, unit_alpha)
                a3 = self._activated_value(g3, u3, unit_alpha)
                st_shared_u32(row_addr, pack_f32x2_to_bfloat2(a0, a1))
                st_shared_u32(
                    row_addr + Int32(4), pack_f32x2_to_bfloat2(a2, a3)
                )
        else:
            for nt in cutlass.range_constexpr(4):
"""
        text = replace_once(
            text,
            identity_anchor,
            identity_epilogue,
            "phase1 identity coordinate epilogue",
        )
    if path.name == "w4a8_phase1.py":
        old_stage = """            _w4a8_stage_trellis_b_tile(
                w13_rp,
                up_b_base,
                tr_common + tr_w13_half,
                tr_k16_stride,
                self.trellis_bits,
                tid,
                self.threads_per_cta,
                4,
            )
        else:
"""
        new_stage = """            _w4a8_stage_trellis_b_tile(
                w13_rp,
                up_b_base,
                tr_common + tr_w13_half,
                tr_k16_stride,
                self.trellis_bits,
                tid,
                self.threads_per_cta,
                4,
            )
            if cutlass.const_expr(self.trellis_scaled):
                self._stage_sfb_half(
                    w13_sfb_rp,
                    gate_sfb_base,
                    Int64(gate_tile_idx) * Int64(256),
                    gate_packed_half,
                    tid,
                )
                self._stage_sfb_half(
                    w13_sfb_rp,
                    up_sfb_base,
                    Int64(up_tile) * Int64(256),
                    up_packed_half,
                    tid,
                )
        else:
"""
        text = replace_once(text, old_stage, new_stage, "phase1 scaled SFB staging")
    else:
        old_stage = """            _w4a8_stage_trellis_b_tile(
                down_rp,
                b_base,
                tr_base,
                tr_k16_stride,
                self.trellis_bits,
                tid,
                self.threads_per_cta,
                8,
            )
        else:
"""
        new_stage = """            _w4a8_stage_trellis_b_tile(
                down_rp,
                b_base,
                tr_base,
                tr_k16_stride,
                self.trellis_bits,
                tid,
                self.threads_per_cta,
                8,
            )
            if cutlass.const_expr(self.trellis_scaled):
                packed_tile = output_tile >> Int32(1)
                packed_half = output_tile & Int32(1)
                b_tile = (
                    expert_idx * packed_output_tiles + packed_tile
                ) * intermediate_tiles + intermediate_slice
                sfb_word_base = Int64(b_tile) * Int64(256)
                for i in cutlass.range_constexpr(
                    ((16 * 8) // 4 + self.threads_per_cta - 1)
                    // self.threads_per_cta
                ):
                    idx = tid + Int32(i * self.threads_per_cta)
                    if idx < Int32((16 * 8) // 4):
                        src_word = sfb_word_base + Int64(
                            packed_half * Int32(16 * 8) + idx * 4
                        )
                        cp_async4_shared_global(
                            sfb_base + (idx << Int32(4)),
                            get_ptr_as_int64(down_sfb_rp, src_word),
                        )
        else:
"""
        text = replace_once(text, old_stage, new_stage, "phase2 scaled SFB staging")
    text = replace_once(
        text,
        "                    if cutlass.const_expr(not self.w4a8_trellis):\n",
        "                    if cutlass.const_expr(\n"
        "                        not self.w4a8_trellis or self.trellis_scaled\n"
        "                    ):\n",
        f"{path.name} scaled SFB consumption",
    )
    text = text.replace(
        "                                not self.trellis_direct_lut,\n",
        "                                not self.trellis_direct_lut\n"
        "                                and self.trellis_codebook != \"mcg\",\n",
    )
    if text.count("and self.trellis_codebook != \"mcg\",") != expected_calls:
        raise RuntimeError(f"{path.name}: dispatch mode call-count mismatch")
    text = replace_once(
        text,
        "            self.w4a8_trellis and not self.trellis_direct_lut\n",
        "            self.w4a8_trellis\n"
        "            and self.trellis_codebook != \"mcg\"\n"
        "            and not self.trellis_direct_lut\n",
        f"{path.name} LUT copy guard",
    )
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args()
    for root in args.roots:
        for name in EXPECTED:
            patch(root / "moe/_shared/kernels" / name)


if __name__ == "__main__":
    main()
