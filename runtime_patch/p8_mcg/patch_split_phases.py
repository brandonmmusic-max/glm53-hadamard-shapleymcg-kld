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
