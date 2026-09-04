"""Bit-closure probe for the procedural P8 MCG-to-E4M3 device decoder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .shard_index import sha256_file
from .trellis_nvfp4 import procedural_state_values


def _expected_words(windows: torch.Tensor, bits: int) -> torch.Tensor:
    table = (
        procedural_state_values("mcg").float().mul_(2.0).to(torch.float8_e4m3fn)
    ).view(torch.uint8)
    words: list[int] = []
    mask = (1 << 32) - 1
    for raw_a, raw_b in windows.cpu().reshape(-1, 2).tolist():
        a = int(raw_a) & mask
        b = int(raw_b) & mask
        states = [
            (b >> (3 * bits)) & 0xFFFF,
            (b >> (2 * bits)) & 0xFFFF,
            (b >> bits) & 0xFFFF,
            b & 0xFFFF,
            (a >> (3 * bits)) & 0xFFFF,
            (a >> (2 * bits)) & 0xFFFF,
            (a >> bits) & 0xFFFF,
            a & 0xFFFF,
        ]
        values = table[torch.tensor(states)].tolist()
        words.append(sum(int(value) << (8 * index) for index, value in enumerate(values[:4])))
        words.append(sum(int(value) << (8 * index) for index, value in enumerate(values[4:])))
    return torch.tensor(words, dtype=torch.int64)


def _run_device(bits: int, pairs: int, seed: int) -> dict[str, object]:
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Uint32
    from cutlass.cute.runtime import from_dlpack

    from b12x._lib.compiler import compile as b12x_compile
    from b12x._lib.utils import current_cuda_stream
    from b12x.moe._shared.kernels.dynamic import (
        _packed_decode_trellis_mcg2_to_e4m3x8,
    )

    class DecodeProbe:
        def __init__(self, stream_bits: int, pair_count: int):
            self.stream_bits = int(stream_bits)
            self.pair_count = int(pair_count)

        @cute.jit
        def __call__(self, windows, output, stream):
            self.kernel(windows, output).launch(
                grid=((self.pair_count + 31) // 32, 1, 1),
                block=[32, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(self, windows, output):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            index = Int32(bidx) * Int32(32) + Int32(tidx)
            if index < Int32(self.pair_count):
                lo, hi = _packed_decode_trellis_mcg2_to_e4m3x8(
                    Uint32(windows[2 * index]),
                    Uint32(windows[2 * index + 1]),
                    self.stream_bits,
                )
                output[2 * index] = Int32(lo)
                output[2 * index + 1] = Int32(hi)

    generator = torch.Generator(device="cuda").manual_seed(seed + bits)
    windows = torch.randint(
        -(2**31),
        2**31 - 1,
        (pairs * 2,),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )
    output = torch.zeros(pairs * 2, dtype=torch.int32, device="cuda")

    def arguments():
        return (
            from_dlpack(windows, assumed_align=16),
            from_dlpack(output, assumed_align=16),
            current_cuda_stream(),
        )

    compiled = b12x_compile(DecodeProbe(bits, pairs), *arguments())
    compiled(*arguments())
    torch.cuda.synchronize()
    actual = output.cpu().to(torch.int64) & 0xFFFFFFFF
    expected = _expected_words(windows, bits)
    mismatches = actual != expected
    return {
        "bits": bits,
        "window_pairs": pairs,
        "decoded_values": pairs * 8,
        "mismatched_words": int(mismatches.sum().item()),
        "bit_exact": not bool(mismatches.any()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.pairs <= 0:
        raise ValueError("pairs must be positive")
    results = [_run_device(bits, args.pairs, args.seed) for bits in (3, 4)]
    payload = {
        "schema": "glm53-p8-mcg-device-decode-closure.v1",
        "law": "procedural MCG alpha 2.0 rounded directly to E4M3",
        "products": ["P8-K3", "P8-K4"],
        "ldlq": False,
        "results": results,
        "decision": "pass" if all(item["bit_exact"] for item in results) else "fail",
        "runtime_source": {
            "path": "runtime_patch/b12x_h16/b12x/moe/_shared/kernels/dynamic.py",
            "sha256": sha256_file(
                Path(__file__).resolve().parents[1]
                / "runtime_patch/b12x_h16/b12x/moe/_shared/kernels/dynamic.py"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    if payload["decision"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
