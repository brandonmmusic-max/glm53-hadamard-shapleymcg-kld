"""Reopen and source-close serialized coupled P8 TP4 sidecars.

This verifier is intentionally independent of the packer's in-memory checks.
For every rank it rebuilds the expected tensor set from the immutable encoder
chunks, reopens the written safetensors file, and requires byte-exact equality
for every tensor plus exact metadata and receipt identities.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from safetensors import safe_open

from .build_p8_coupled_rate_tp4_sidecars import (
    _load_rank_coupled,
    _payload_rates,
    _rank_metadata,
)
from .p8_coupled_scale import (
    COUPLED_RANK_SCHEMA,
    COUPLED_SIDECAR_SCHEMA,
    SUPPORTED_LAYERS,
    _tensor_sha256,
)
from .shard_index import sha256_file


POSTWRITE_SCHEMA = "glm53-p8-coupled-tp4-postwrite-closure.v1"


def _fail(message: str) -> None:
    raise RuntimeError(f"coupled postwrite closure failed: {message}")


def verify_postwrite(
    *,
    chunks: list[Path],
    sidecars: list[Path],
    packer_receipt: Path,
    layer: int,
    expected_hidden: int | None = 4096,
    expected_intermediate: int | None = 2048,
) -> dict[str, object]:
    """Verify all four written ranks against their source chunk tensors."""

    if layer not in SUPPORTED_LAYERS:
        raise ValueError("coupled verifier requires GLM routed layers 3..44")
    if len(chunks) != 4 or len({path.resolve() for path in chunks}) != 4:
        raise ValueError("postwrite closure requires four distinct encoder chunks")
    if len(sidecars) != 4 or len({path.resolve() for path in sidecars}) != 4:
        raise ValueError("postwrite closure requires four distinct TP rank sidecars")
    if not packer_receipt.is_file():
        raise FileNotFoundError(packer_receipt)
    packer = json.loads(packer_receipt.read_text())
    if (
        packer.get("schema") != COUPLED_SIDECAR_SCHEMA
        or packer.get("layer") != layer
        or packer.get("world_size") != 4
        or packer.get("boundary") != "coupled-h512-h128-suh-svh-v1"
        or packer.get("bits") not in (3, 4, 5)
        or packer.get("weight_payload_bpw") != packer.get("bits") + 0.25
        or packer.get("ldlq") is not False
    ):
        _fail("packer receipt header differs from the frozen TP4 product")
    rank_entries = packer.get("ranks")
    if not isinstance(rank_entries, list) or [row.get("rank") for row in rank_entries] != list(range(4)):
        _fail("packer receipt does not contain ranks 0,1,2,3 exactly once")

    closed_ranks: list[dict[str, object]] = []
    shared_sources: list[dict[str, object]] | None = None
    shared_design: str | None = None
    shared_scale_source: str | None = None
    for rank, sidecar in enumerate(sidecars):
        if not sidecar.is_file():
            raise FileNotFoundError(sidecar)
        expected, sources, design_sha256, scale_source_sha256, bits = _load_rank_coupled(
            chunks,
            layer=layer,
            rank=rank,
            world_size=4,
            expected_hidden=expected_hidden,
            expected_intermediate=expected_intermediate,
        )
        weight_bytes, metadata_bytes, weight_bpw, metadata_bpw = _payload_rates(expected)
        expected_metadata = _rank_metadata(
            expected,
            layer=layer,
            rank=rank,
            design_sha256=design_sha256,
            scale_source_sha256=scale_source_sha256,
            metadata_bpw=metadata_bpw,
            bits=bits,
        )
        if bits != packer.get("bits"):
            _fail("packer receipt rate differs from the chunk rate")
        entry = rank_entries[rank]
        actual_file_sha256 = sha256_file(sidecar)
        if (
            Path(entry.get("path", "")).resolve() != sidecar.resolve()
            or entry.get("bytes") != sidecar.stat().st_size
            or entry.get("sha256") != actual_file_sha256
            or entry.get("weight_payload_bytes") != weight_bytes
            or entry.get("metadata_bytes") != metadata_bytes
            or entry.get("weight_payload_bpw") != weight_bpw
            or entry.get("metadata_bpw") != metadata_bpw
            or entry.get("stored_bpw") != weight_bpw + metadata_bpw
            or entry.get("source_design_sha256") != design_sha256
            or entry.get("exl3_scale_source_sha256") != scale_source_sha256
            or entry.get("sources") != sources
        ):
            _fail(f"rank {rank} packer receipt differs from file/source derivation")

        expected_hashes = {
            name: _tensor_sha256(value) for name, value in expected.items()
        }
        expected_shapes = {
            name: list(value.shape) for name, value in expected.items()
        }
        expected_dtypes = {
            name: str(value.dtype).removeprefix("torch.")
            for name, value in expected.items()
        }
        if entry.get("tensor_sha256") != expected_hashes:
            _fail(f"rank {rank} receipt tensor hashes differ from source chunks")
        if entry.get("shapes") != expected_shapes:
            _fail(f"rank {rank} receipt shapes differ from source chunks")

        with safe_open(sidecar, framework="pt", device="cpu") as written:
            actual_metadata = written.metadata() or {}
            if actual_metadata.get("schema") != COUPLED_RANK_SCHEMA:
                _fail(f"rank {rank} sidecar schema differs")
            if actual_metadata != expected_metadata:
                _fail(f"rank {rank} metadata differs from source-derived metadata")
            if set(written.keys()) != set(expected):
                _fail(f"rank {rank} tensor inventory differs from source chunks")
            for name in sorted(expected):
                actual = written.get_tensor(name)
                if (
                    actual.dtype != expected[name].dtype
                    or tuple(actual.shape) != tuple(expected[name].shape)
                    or not torch.equal(actual, expected[name])
                ):
                    _fail(f"rank {rank} tensor {name} differs from source chunks")
                actual_hash = _tensor_sha256(actual)
                if (
                    actual_hash != expected_hashes[name]
                    or actual_metadata.get(f"sha256_{name}") != actual_hash
                ):
                    _fail(f"rank {rank} tensor {name} hash chain differs")
                del actual

        if shared_sources is None:
            shared_sources = sources
            shared_design = design_sha256
            shared_scale_source = scale_source_sha256
        elif (
            sources != shared_sources
            or design_sha256 != shared_design
            or scale_source_sha256 != shared_scale_source
        ):
            _fail("TP ranks do not derive from one chunk/design/scale identity")
        closed_ranks.append(
            {
                "rank": rank,
                "path": str(sidecar.resolve()),
                "bytes": sidecar.stat().st_size,
                "sha256": actual_file_sha256,
                "tensor_count": len(expected),
                "tensor_sha256": expected_hashes,
                "shapes": expected_shapes,
                "dtypes": expected_dtypes,
                "source_exact": True,
            }
        )
        del expected
        gc.collect()

    assert shared_sources is not None
    return {
        "schema": POSTWRITE_SCHEMA,
        "status": "pass",
        "evidence_level": "postwrite-source-exact-structural-closure",
        "layer": layer,
        "world_size": 4,
        "packer_receipt": {
            "path": str(packer_receipt.resolve()),
            "sha256": sha256_file(packer_receipt),
        },
        "source_chunks": shared_sources,
        "source_design_sha256": shared_design,
        "exl3_scale_source_sha256": shared_scale_source,
        "ranks": closed_ranks,
        "chunk_retirement_gate_closed": "postwrite-all-tensor-source-closure",
        "runtime_loader_closure": "not tested",
        "retirement_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=Path, action="append", required=True)
    parser.add_argument("--sidecar", type=Path, action="append", required=True)
    parser.add_argument("--packer-receipt", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--layer", type=int, choices=SUPPORTED_LAYERS, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(f"refusing to overwrite {args.receipt}")
    result = verify_postwrite(
        chunks=args.chunk,
        sidecars=args.sidecar,
        packer_receipt=args.packer_receipt,
        layer=args.layer,
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
