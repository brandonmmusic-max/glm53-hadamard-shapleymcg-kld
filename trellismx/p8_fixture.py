"""Deterministic CPU-only K3/K4/K5 TrellisMX P8 reference fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import safe_open, save_file

from .protocol import (
    P8_ALPHABET,
    P8_BLOCK_SIZE,
    P8_COMPANDER_SCALE,
    P8_LAW,
    P8_MMA,
    P8_RATES,
    decode_p8_payload,
    pack_trellis_edges,
    p8_state_table,
)


P8_FIXTURE_SCHEMA = "trellismx.p8-cpu-reference-fixture.v1"
ROWS = 16
WIDTH = 32


def _payload(bits: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260907 + bits)
    edges = torch.randint(
        0,
        1 << bits,
        (WIDTH // 16, ROWS // 16, 256),
        dtype=torch.int64,
        generator=generator,
    )
    trellis = pack_trellis_edges(edges, bits)
    codebook = p8_state_table(bits)
    scale = torch.randint(0, 255, (ROWS, WIDTH // P8_BLOCK_SIZE), generator=generator, dtype=torch.uint8)
    decoded = decode_p8_payload(
        trellis,
        codebook,
        scale,
        bits=bits,
        rows=ROWS,
        width=WIDTH,
    ).contiguous()
    return {
        f"k{bits}_trellis": trellis,
        f"k{bits}_codebook_e4m3": codebook,
        f"k{bits}_scale_ue8m0": scale,
        f"k{bits}_decoded_reference": decoded,
    }


def _metadata() -> dict[str, str]:
    value: dict[str, Any] = {
        "schema": P8_FIXTURE_SCHEMA,
        "role": "synthetic-codec-test-only",
        "rows": ROWS,
        "width": WIDTH,
        "rates": list(P8_RATES),
        "alphabet": P8_ALPHABET,
        "law": P8_LAW,
        "mma": P8_MMA,
        "block_scale": f"ue8m0-k{P8_BLOCK_SIZE}",
        "compander_scale": P8_COMPANDER_SCALE,
        "gpu_executed": False,
        "not_tested": [
            "GPU execution",
            "Viterbi/Hessian encoding",
            "Tensor Core accumulation",
            "model quality",
            "serving performance",
        ],
    }
    return {key: json.dumps(item, separators=(",", ":")) for key, item in value.items()}


def write_fixture(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    tensors: dict[str, torch.Tensor] = {}
    for bits in P8_RATES:
        tensors.update(_payload(bits))
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, path, metadata=_metadata())


def verify_fixture(path: Path) -> dict[str, Any]:
    with safe_open(path, framework="pt", device="cpu") as source:
        metadata = source.metadata()
        if metadata.get("schema") != json.dumps(P8_FIXTURE_SCHEMA, separators=(",", ":")):
            raise ValueError("unsupported P8 fixture schema")
        if sorted(source.keys()) != sorted(
            f"k{bits}_{field}"
            for bits in P8_RATES
            for field in ("trellis", "codebook_e4m3", "scale_ue8m0", "decoded_reference")
        ):
            raise ValueError("P8 fixture tensor inventory is invalid")
        mismatches: dict[str, int] = {}
        for bits in P8_RATES:
            expected = _payload(bits)
            for field, value in expected.items():
                observed = source.get_tensor(field)
                mismatches[field] = int((observed != value).sum().item())

    return {
        "schema": P8_FIXTURE_SCHEMA,
        "status": "passed" if not any(mismatches.values()) else "failed",
        "rates": list(P8_RATES),
        "exact_tensor_mismatches": mismatches,
        "gpu_executed": False,
        "evidence_level": "structural-cpu-reference",
        "not_tested": [
            "GPU execution",
            "Viterbi/Hessian encoding",
            "Tensor Core accumulation",
            "model quality",
            "serving performance",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    write_parser = subparsers.add_parser("write")
    write_parser.add_argument("path", type=Path)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    if args.command == "write":
        write_fixture(args.path)
    print(json.dumps(verify_fixture(args.path), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
