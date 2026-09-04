"""Offline audit diagnostics, not a P4 codec or kernel implementation.

Uses only the standard library, explicit source paths, and synthetic integers.
It does not import repository modules, discover datasets, or initialize CUDA.
The census interprets inspected source semantics; it does not execute Torch.
ExLlamaV3 MCG constants: turboderp and contributors; see LICENSE.exllamav3.
"""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import struct
import unittest


ROOT = Path(__file__).resolve().parents[4]
SOURCES = (
    "experiments/p4-astra-agent-plan-v1.json",
    "glm53_nvfp4/modelopt.py",
    "glm53_nvfp4/trellis_nvfp4.py",
    "glm53_nvfp4/trellis_mxf.py",
    "glm53_nvfp4/endpoint_codec.py",
    "glm53_nvfp4/export_native_endpoint.py",
    "runtime_patch/p8_native_kernel.py",
    "runtime_patch/sitecustomize.py",
    "runtime_patch/p8_mcg/w4a8_mcg_decode.py",
    "runtime_patch/p8_mcg/patch_split_phases.py",
    "runtime_patch/p8_mcg/Dockerfile",
    "runtime_patch/b12x_h16/b12x/moe/_shared/kernels/dynamic.py",
    "tests/test_modelopt.py",
    "tests/test_trellis_nvfp4.py",
    "tests/test_trellis_mxf.py",
    "tests/test_endpoint_codec.py",
    "tests/test_export_native_endpoint.py",
    "THIRD_PARTY_NOTICES.md",
    "LICENSE",
    "LICENSE.exllamav3",
    "runtime_patch/b12x_h16/LICENSE.b12x",
)
LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def half_value(word: int) -> float:
    return struct.unpack("<e", struct.pack("<H", word))[0]


def mcg(state: int) -> float:
    word = (((state * 0xCBAC1FED) & 0xFFFFFFFF) & 0x8FFF8FFF) ^ 0x3B603B60
    summed = half_value(word & 65535) + half_value(word >> 16)
    return struct.unpack("<e", struct.pack("<e", summed))[0]


def half_units(word: int) -> int:
    """Independent finite binary16 decode in integer units of 2**-24."""
    exp, frac = (word >> 10) & 31, word & 1023
    assert exp != 31
    units = frac if exp == 0 else (1024 + frac) << (exp - 1)
    return -units if word & 32768 else units


def integer_half_sum(left: int, right: int) -> float:
    units = half_units(left) + half_units(right)
    sign = -1 if units < 0 else 1
    value = abs(units)
    shift = max(0, value.bit_length() - 11)
    quotient, remainder = divmod(value, 1 << shift)
    if shift and (remainder * 2 > (1 << shift) or
                  (remainder * 2 == (1 << shift) and quotient & 1)):
        quotient += 1
    return sign * (quotient << shift) * 2.0**-24


def nibble(value: float, *, ties_even: bool) -> int:
    magnitude = min(range(8), key=lambda i: (
        abs(abs(value) - LEVELS[i]), i % 2 if ties_even else i
    ))
    # Both diagnostic laws canonicalize zero, isolating rounding differences.
    return magnitude | (8 if value < 0 and magnitude else 0)


def census(alpha: int) -> dict:
    source, nearest_even = bytearray(), bytearray()
    magnitudes: Counter = Counter()
    examples = []
    for state in range(65536):
        value = mcg(state) * alpha
        low = nibble(value, ties_even=False)
        even = nibble(value, ties_even=True)
        source.append(low)
        nearest_even.append(even)
        if low != even:
            magnitudes[str(abs(value))] += 1
            if len(examples) < 2:
                examples.append(dict(state=state, value=value,
                                     source_nibble=low, rne_nibble=even))
    return dict(states=65536, mismatches=sum(magnitudes.values()),
                magnitude_counts=dict(sorted(magnitudes.items())), examples=examples,
                canonical_nibble_sha256=hashlib.sha256(source).hexdigest(),
                rne_nibble_sha256=hashlib.sha256(nearest_even).hexdigest())


class AuditDiagnostics(unittest.TestCase):
    def test_two_independent_half_add_models(self):
        for state in range(65536):
            word = (((state * 0xCBAC1FED) & 0xFFFFFFFF) & 0x8FFF8FFF) ^ 0x3B603B60
            self.assertEqual(mcg(state), integer_half_sum(word & 65535, word >> 16))

    def test_rounding_counterexample(self):
        self.assertEqual(mcg(110), -1.75)
        self.assertEqual(nibble(mcg(110), ties_even=False), 0xB)
        self.assertEqual(nibble(mcg(110), ties_even=True), 0xC)

    def test_dense_mma_coordinate_coverage(self):
        # Algebraic restatement of the PTX dense m16n8k64 fragment map.
        # This checks the audit's map, not a compiled kernel's lane mapping.
        a, b, d = [], [], []
        for lane in range(32):
            g, c = lane // 4, lane % 4
            a.extend((g + 8 * ((i // 8) % 2),
                      8 * c + i % 8 + 32 * (i // 16)) for i in range(32))
            b.extend((8 * c + i % 8 + 32 * (i // 8), g) for i in range(16))
            d.extend((g + 8 * (i // 2), 2 * c + i % 2) for i in range(4))
        for coordinates, rows, cols in ((a, 16, 64), (b, 64, 8), (d, 16, 8)):
            self.assertEqual(len(coordinates), rows * cols)
            self.assertEqual(set(coordinates),
                             {(r, c) for r in range(rows) for c in range(cols)})


def main() -> None:
    sources = {}
    parsed = 0
    for name in SOURCES:
        path = ROOT / name
        assert not path.is_symlink(), f"unexpected source symlink: {name}"
        raw = path.read_bytes()
        sources[name] = hashlib.sha256(raw).hexdigest()
        if path.suffix == ".py":
            compile(ast.parse(raw, filename=name), name, "exec")
            parsed += 1
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AuditDiagnostics)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    decoder = ast.parse((ROOT / "runtime_patch/p8_mcg/w4a8_mcg_decode.py").read_text())
    asm = next(node.value for node in ast.walk(decoder)
               if isinstance(node, ast.Constant) and isinstance(node.value, str)
               and "mov.b32 M, 0xCBAC1FED;" in node.value)
    ops = Counter(re.findall(r"\b(mov|and|shr|mul|lop3|prmt|add|cvt)\.[^;\s]+", asm))
    print(json.dumps(dict(
        schema="glm53-p4-audit-static-diagnostics.v1",
        python_files_syntax_checked=parsed,
        diagnostic_tests_passed=result.testsRun,
        repository_modules_executed=False,
        gpu_initialized=False,
        source_sha256=sources,
        rounding_census={str(alpha): census(alpha) for alpha in (1, 2)},
        p8_template_ptx_ops_per_eight_states=dict(sorted(ops.items())),
        p8_template_ptx_ops_total=sum(ops.values()),
        limits="Analytical CPU diagnostics and syntax only; no product bit closure or SASS cost measurement."
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
