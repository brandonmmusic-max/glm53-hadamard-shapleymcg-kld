"""Synthetic P4 fixture and a scalar byte oracle independent of the vector decoder.

Run ``python -m glm53_nvfp4.p4_fixture --help``. No Torch or GPU imports.
The oracle uses scalar struct binary16 rounding and inverse logical-coordinate
addressing, rather than the production decoder's NumPy lane scatter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
from pathlib import Path

import numpy as np

from .p4_codec import (
    P4Operands, P4Payload, decode_p4, e2m1_codes, e4m3_scale_values,
    fit_e4m3_scales, mcg_half_values, pack_k4_edges, read_p4, serialize_p4,
    storage_accounting, swizzle_scales, verify_operands,
)


def scalar_mcg_code(state: int) -> int:
    product = (state * 0xCBAC1FED) % (1 << 32)
    word = (product & 0x8FFF8FFF) ^ 0x3B603B60
    lo = struct.unpack("<e", struct.pack("<H", word & 65535))[0]
    hi = struct.unpack("<e", struct.pack("<H", word >> 16))[0]
    value = struct.unpack("<e", struct.pack("<e", lo + hi))[0]
    magnitudes = (0., .5, 1., 1.5, 2., 3., 4., 6.)
    code = min(range(8), key=lambda c: (abs(abs(value) - magnitudes[c]), c % 2))
    return code | (8 if math.copysign(1., value) < 0 else 0)


def scalar_state(tile: bytes, position: int) -> int:
    result = 0
    for lag in range(4):
        index = (position - lag) % 256
        word = struct.unpack_from("<I", tile, (index // 8) * 4)[0]
        edge = (word >> (28 - (index % 8) * 4)) & 15
        result += edge * 16**lag
    return result


def scalar_operands(payload: P4Payload) -> P4Operands:
    weight = np.zeros((payload.rows, payload.width // 2), dtype=np.uint8)
    for row in range(payload.rows):
        for column in range(payload.width):
            n, k = row % 16, column % 16
            lane = (n % 8) * 4 + (k % 8) // 2
            slot = (n // 8) * 4 + (k // 8) * 2 + k % 2
            tile = payload.trellis[column // 16, row // 16].tobytes()
            code = scalar_mcg_code(scalar_state(tile, lane * 8 + slot))
            weight[row, column // 2] |= code << (4 * (column % 2))
    return P4Operands(weight, payload.scale_e4m3.copy(), payload.global_scale.copy())


def scalar_swizzle(raw: np.ndarray) -> np.ndarray:
    rows, groups = raw.shape
    rp, gp = (rows + 127) // 128 * 128, (groups + 3) // 4 * 4
    result = np.zeros(rp * gp, dtype=np.uint8)
    for row in range(rows):
        for group in range(groups):
            tile_base = ((row // 128) * (gp // 4) + group // 4) * 512
            index = tile_base + (row % 32) * 16 + (row % 128 // 32) * 4 + group % 4
            result[index] = raw[row, group]
    return result.reshape(rp, gp)


def make_fixture() -> P4Payload:
    """Fixed [32,128] matrix, sixteen independently wrapped 256-symbol tiles."""
    edges = np.empty((8, 2, 256), dtype=np.uint8)
    for kt in range(8):
        for nt in range(2):
            state = (0x5034A57A ^ (kt * 0x9E3779B9) ^ (nt * 0x85EBCA6B)) & 0xFFFFFFFF
            for position in range(256):
                state ^= (state << 13) & 0xFFFFFFFF
                state ^= state >> 17
                state ^= (state << 5) & 0xFFFFFFFF
                edges[kt, nt, position] = state & 15
    scales = np.array([1 + (index * 37) % 126 for index in range(256)], dtype=np.uint8).reshape(32, 8)
    return P4Payload(32, 128, pack_k4_edges(edges), scales, np.array(.125, dtype="<f4"))


def fixture_expectation(payload: P4Payload) -> dict:
    expected = scalar_operands(payload)
    return {
        "schema": "glm53-p4-cpu-fixture.v1",
        "evidence_level": "structural",
        "input_role": "synthetic-codec-test-only",
        "payload_sha256": hashlib.sha256(serialize_p4(payload)).hexdigest(),
        "shape": [payload.rows, payload.width],
        "weight_shape": list(expected.weight.shape),
        "weight_e2m1_hex": expected.weight.tobytes().hex(),
        "scale_e4m3_shape": list(expected.scale_e4m3.shape),
        "scale_e4m3_hex": expected.scale_e4m3.tobytes().hex(),
        "scale_modelopt_128x4_shape": list(scalar_swizzle(expected.scale_e4m3).shape),
        "scale_modelopt_128x4_hex": scalar_swizzle(expected.scale_e4m3).tobytes().hex(),
        "global_scale_f32_le_hex": expected.global_scale.tobytes().hex(),
        "state_probes": [
            {"k_tile": kt, "n_tile": nt, "position": p,
             "state": scalar_state(payload.trellis[kt, nt].tobytes(), p)}
            for kt, nt in ((0, 0), (7, 1)) for p in (0, 1, 2, 3, 7, 8, 15, 16, 253, 254, 255)
        ],
        "not_tested": ["GPU execution", "MMA register layout", "GPU scale repacking", "MoE integration", "speed", "model quality"],
    }


def verify_fixture(directory: Path) -> dict:
    record = json.loads((directory / "fixture.expected.json").read_text())
    payload = read_p4(directory / "fixture.p4.safetensors", expected_shape=(32, 128),
                      expected_sha256=record["payload_sha256"])
    # Regenerate from the published recipe; do not trust self-reported hashes alone.
    if serialize_p4(make_fixture()) != serialize_p4(payload):
        raise ValueError("fixture input differs from the deterministic recipe")
    if fixture_expectation(payload) != record:
        raise ValueError("fixture expectation differs from the independent scalar oracle")
    mismatches = verify_operands(payload, scalar_operands(payload))
    if not np.array_equal(swizzle_scales(payload.scale_e4m3), scalar_swizzle(payload.scale_e4m3)):
        raise ValueError("scale swizzle differs from scalar address oracle")
    # Known representable sources recover every nondegenerate fixture scale.
    codes = decode_p4(payload).weight
    nibble_codes = np.stack((codes & 15, codes >> 4), axis=-1).reshape(32, 128)
    values = np.array([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=np.float64)[nibble_codes & 7]
    values *= np.where(nibble_codes & 8, -1, 1)
    source = (values.reshape(32, 8, 16) * e4m3_scale_values(payload.scale_e4m3)[..., None] * .125).reshape(32, 128).astype("<f4")
    fitted = fit_e4m3_scales(source, nibble_codes, payload.global_scale)
    expected_fit = np.where((values.reshape(32, 8, 16) ** 2).sum(-1) == 0, 0x38, payload.scale_e4m3)
    if not np.array_equal(fitted, expected_fit):
        raise ValueError("representable fixed-code scale fit did not round-trip")
    return {"status": "passed", "evidence_level": "structural", "operand_byte_mismatches": mismatches,
            "fixture_sha256": record["payload_sha256"], "storage": storage_accounting(payload),
            "scale_roundtrip": "passed", "independent_scalar_oracle": "passed",
            "not_tested": record["not_tested"]}


def verify_cuda_host_oracle(executable: Path) -> dict:
    """Run a g++-built host-only oracle; no CUDA runtime, driver, or device call."""
    digest = hashlib.sha256()
    process = subprocess.Popen([str(executable.resolve()), "mcg"], stdout=subprocess.PIPE)
    assert process.stdout is not None
    try:
        for start in range(0, 65536, 256):
            observed = process.stdout.read(256)
            expected = e2m1_codes(mcg_half_values(np.arange(start, start + 256, dtype="<u2"))).tobytes()
            if observed != expected:
                raise ValueError(f"CUDA host MCG conversion mismatch in states {start}..{start+255}")
            digest.update(observed)
        if process.stdout.read(1) or process.wait() != 0:
            raise ValueError("CUDA host oracle output length or exit status mismatch")
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
            process.wait()
    observed_scales = subprocess.check_output([str(executable.resolve()), "scales"])
    expected_scales = e4m3_scale_values(np.arange(1, 127, dtype=np.uint8)).astype("<f4").tobytes()
    if observed_scales != expected_scales:
        raise ValueError("CUDA host positive E4M3 decode mismatch")
    return {"status": "passed", "evidence_level": "structural", "mcg_states": 65536,
            "mcg_e2m1_sha256": digest.hexdigest(), "positive_e4m3_codes": 126,
            "gpu_executed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--write", action="store_true", help="create deterministic synthetic fixture; refuses overwrite")
    parser.add_argument("--cuda-host-oracle", type=Path, help="optional g++-built scripts/p4_cuda_host_oracle.cpp executable")
    args = parser.parse_args()
    if args.write:
        args.directory.mkdir(parents=True, exist_ok=True)
        paths = [args.directory / "fixture.p4.safetensors", args.directory / "fixture.expected.json"]
        if any(path.exists() for path in paths):
            raise FileExistsError("refusing to overwrite fixture files")
        payload = make_fixture()
        paths[0].write_bytes(serialize_p4(payload))
        paths[1].write_text(json.dumps(fixture_expectation(payload), sort_keys=True, indent=2) + "\n")
    result = verify_fixture(args.directory)
    if args.cuda_host_oracle:
        result["cuda_host_oracle"] = verify_cuda_host_oracle(args.cuda_host_oracle)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
