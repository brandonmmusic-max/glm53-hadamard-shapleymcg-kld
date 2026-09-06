#!/usr/bin/env python3
"""Localize the bytes that differ between the device kernel and the CPU reference.

The grouped M64/N128 prefill device closure asserts byte-exact equality on four
observable carriers. When it fails it saves the actual/expected pairs; this
script decodes the differing bytes so the failure can be classified.

Discriminator the closure is really trying to make:

  * A decode defect (wrong rate, wrong lane, wrong descriptor stride) corrupts
    many bytes with arbitrary deltas and normally disturbs the UE8M0 scale
    bytes too, because the block maximum moves.
  * A floating-point ordering difference between a CPU fp32 reference GEMM and
    a tensor-core mxf8f6f4 MMA moves a handful of elements that sit on an E4M3
    rounding boundary by exactly one code point, and leaves every scale byte
    identical.

Usage:
    python3 scripts/closure-repro/localize_carrier_mismatch.py <first-carrier-failure.npz>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

CARRIERS = ("input_payload", "input_scale", "middle_payload", "middle_scale")


def decode_e4m3(byte_value: int) -> float:
    raw = np.array([byte_value], dtype=np.uint8)
    return torch.from_numpy(raw).view(torch.float8_e4m3fn).float().item()


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    path = Path(argv[1])
    bundle = np.load(path)

    verdict_one_ulp = True
    for carrier in CARRIERS:
        actual = bundle[f"{carrier}_actual"]
        expected = bundle[f"{carrier}_expected"]
        rows, cols = np.nonzero(actual != expected)
        print(f"{carrier}: shape {tuple(actual.shape)}, "
              f"{len(rows)} of {actual.size} bytes differ")
        if len(rows) == 0:
            continue
        if carrier.endswith("_scale"):
            verdict_one_ulp = False
        for row, col in zip(rows[:32], cols[:32]):
            a = int(actual[row, col])
            e = int(expected[row, col])
            delta = a - e
            if abs(delta) != 1:
                verdict_one_ulp = False
            detail = ""
            if carrier.endswith("_payload"):
                detail = f"  device {decode_e4m3(a):+.6g}  cpu {decode_e4m3(e):+.6g}"
            print(f"    [{row},{col}] device {a:3d} cpu {e:3d} delta {delta:+d}{detail}")
        if len(rows) > 32:
            print(f"    ... {len(rows) - 32} more")

    print()
    if verdict_one_ulp:
        print("VERDICT: every difference is a single E4M3 code point and no scale "
              "byte moved. This is consistent with accumulate-order rounding, not "
              "with a decode defect.")
    else:
        print("VERDICT: differences exceed one code point or disturb the scale "
              "bytes. Treat this as a decode defect until proven otherwise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
