"""Redirect BF16 weight tensors inside an already-BF16 sparse overlay."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from safetensors import safe_open

from .shard_index import sha256_file


PROJECTIONS = {"gate_proj", "up_proj", "down_proj"}


def _projection_for(name: str) -> str | None:
    return next(
        (projection for projection in PROJECTIONS if name.endswith(f".{projection}.weight")),
        None,
    )


def build(
    carrier: Path,
    output: Path,
    chunks: list[Path],
    layers: set[int],
    projections: set[str] | None = None,
) -> dict:
    if not layers or not layers.issubset(set(range(3, 45))):
        raise ValueError(f"invalid layer set: {sorted(layers)}")
    projections = PROJECTIONS if projections is None else projections
    if not projections or not projections.issubset(PROJECTIONS):
        raise ValueError(f"invalid projection set: {sorted(projections)}")
    output.mkdir(parents=True, exist_ok=True)
    original = json.loads((carrier / "model.safetensors.index.json").read_text())
    weight_map = dict(original["weight_map"])
    redirected: dict[str, str] = {}
    for item in carrier.iterdir():
        if item.name in {"model.safetensors.index.json", "OVERLAY.json"} or item.name.startswith(".cache"):
            continue
        target = output / item.name
        if not target.exists() and not target.is_symlink():
            os.symlink(item.resolve(), target)
    seen_layers: set[int] = set()
    for chunk in chunks:
        target = output / chunk.name
        if not target.exists() and not target.is_symlink():
            os.symlink(chunk.resolve(), target)
        with safe_open(str(chunk), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            layer = int(metadata["layer"])
            if layer not in layers:
                raise RuntimeError(f"chunk layer {layer} is not declared")
            seen_layers.add(layer)
            for name in handle.keys():
                if not name.endswith(".weight") or name not in weight_map:
                    raise RuntimeError(f"invalid BF16 replacement tensor: {name}")
                if handle.get_slice(name).get_dtype() != "BF16":
                    raise RuntimeError(f"replacement is not BF16: {name}")
                if _projection_for(name) not in projections:
                    continue
                weight_map[name] = chunk.name
                redirected[name] = chunk.name
    if seen_layers != layers:
        raise RuntimeError(f"missing chunks for layers {sorted(layers - seen_layers)}")
    updated = dict(original)
    updated["weight_map"] = weight_map
    index = output / "model.safetensors.index.json"
    index.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n")
    overlay = {
        "schema": "glm53-bf16-weight-overlay.v1",
        "carrier": str(carrier.resolve()),
        "layers": sorted(layers),
        "projections": sorted(projections),
        "chunks": [str(path.resolve()) for path in chunks],
        "redirected_tensors": len(redirected),
        "required_load_format": "instanttensor",
        "config_unchanged": True,
    }
    (output / "OVERLAY.json").write_text(json.dumps(overlay, indent=2, sort_keys=True) + "\n")
    receipt = {
        **overlay,
        "index_sha256": sha256_file(index),
        "overlay_sha256": sha256_file(output / "OVERLAY.json"),
        "chunk_files": [
            {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in chunks
        ],
        "ldlq": False,
    }
    (output / "BF16_WEIGHT_OVERLAY_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument(
        "--projections",
        nargs="+",
        choices=sorted(PROJECTIONS),
        default=sorted(PROJECTIONS),
    )
    parser.add_argument("--chunk", type=Path, action="append", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.carrier,
                args.output,
                args.chunk,
                set(args.layers),
                set(args.projections),
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
