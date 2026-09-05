"""Verify a reconstructed P8 dense shard against its historical NMSE receipt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from .shard_index import IndexedCheckpoint, sha256_file


def _metric_key(row: dict[str, object]) -> tuple[int, str]:
    return int(row["expert"]), str(row["projection"])


def _tensor_key(name: str) -> tuple[int, str]:
    expert = int(name.split(".experts.", 1)[1].split(".", 1)[0])
    projection = name.split(f".experts.{expert}.", 1)[1].split(".", 1)[0]
    return expert, projection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--historical-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--absolute-tolerance", type=float, default=1e-15)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.absolute_tolerance < 0:
        raise ValueError("absolute tolerance must be nonnegative")

    historical = json.loads(args.historical_receipt.read_text())
    expected = {
        _metric_key(row): float(row["weight_nmse"])
        for row in historical["projection_metrics"]
    }
    checkpoint = IndexedCheckpoint(args.source, args.source_index)
    rows: list[dict[str, object]] = []
    maximum_error = 0.0
    with safe_open(args.dense, framework="pt", device="cpu") as dense:
        names = sorted(name for name in dense.keys() if name.endswith(".weight"))
        actual_keys = {_tensor_key(name) for name in names}
        if actual_keys != set(expected):
            raise RuntimeError("dense tensor identities disagree with historical receipt")
        for index, name in enumerate(names, start=1):
            key = _tensor_key(name)
            quantized = dense.get_tensor(name).to(args.device).float()
            source = checkpoint.get(name).to(args.device).float()
            nmse = float(
                (
                    (quantized - source).double().square().sum()
                    / source.double().square().sum()
                ).item()
            )
            error = abs(nmse - expected[key])
            maximum_error = max(maximum_error, error)
            rows.append(
                {
                    "expert": key[0],
                    "projection": key[1],
                    "historical_nmse": expected[key],
                    "reconstructed_nmse": nmse,
                    "absolute_error": error,
                }
            )
            if index % 24 == 0:
                print(json.dumps({"verified_tensors": index}), flush=True)

    passed = len(rows) == len(expected) and maximum_error <= args.absolute_tolerance
    result = {
        "schema": "glm53-p8.historical-dense-semantic-closure.v1",
        "passed": passed,
        "dense": str(args.dense.resolve()),
        "dense_sha256": sha256_file(args.dense),
        "dense_bytes": args.dense.stat().st_size,
        "source_index_sha256": sha256_file(args.source_index),
        "historical_receipt": str(args.historical_receipt.resolve()),
        "historical_receipt_sha256": sha256_file(args.historical_receipt),
        "verified_tensors": len(rows),
        "absolute_tolerance": args.absolute_tolerance,
        "maximum_nmse_absolute_error": maximum_error,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("passed", "verified_tensors", "maximum_nmse_absolute_error")}, sort_keys=True))
    if not passed:
        raise RuntimeError("reconstructed dense shard failed historical semantic closure")


if __name__ == "__main__":
    main()
