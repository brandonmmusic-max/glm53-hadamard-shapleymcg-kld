"""Assemble a selective fixed-H16 overlay from the completed per-layer build."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .candidate import ROUTED_LAYERS, build


def parse_layers(spec: str) -> list[int]:
    layers: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            first_text, last_text = item.split("-", 1)
            first, last = int(first_text), int(last_text)
            if first > last:
                raise ValueError(f"invalid layer range {item!r}")
            layers.update(range(first, last + 1))
        else:
            layers.add(int(item))
    allowed = set(ROUTED_LAYERS)
    if not layers or not layers.issubset(allowed):
        raise ValueError(f"invalid routed layer set: {sorted(layers)}")
    return sorted(layers)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_chunks(full_root: Path, layer3_root: Path, layers: list[int]) -> list[Path]:
    chunks: list[Path] = []
    for layer in layers:
        base = layer3_root if layer == 3 else full_root / "layers" / f"layer-{layer:03d}" / "had16"
        found = sorted((base / "chunks").glob(f"had16-layer-{layer:03d}-*.safetensors"))
        if len(found) != 5:
            raise RuntimeError(f"expected five H16 chunks for layer {layer}, found {len(found)}")
        chunks.extend(found)
    return chunks


def assemble(
    carrier: Path,
    full_root: Path,
    layer3_root: Path,
    output: Path,
    layer_spec: str,
    receipt_path: Path,
) -> dict:
    layers = parse_layers(layer_spec)
    chunks = find_chunks(full_root, layer3_root, layers)
    overlay = build(carrier, output, chunks)
    expected = 3456 * len(layers)
    if overlay["redirected_tensors"] != expected:
        raise RuntimeError(
            f"selective overlay redirected {overlay['redirected_tensors']} tensors; expected {expected}"
        )
    index = output / "model.safetensors.index.json"
    overlay_path = output / "OVERLAY.json"
    receipt = {
        "schema": "glm53-nvfp4-v10.selective-h16-overlay.v1",
        "status": "pass",
        "rotation": "had16",
        "rotation_scope": "all",
        "runtime_rotation_placement": "humming-inner",
        "required_load_format": "instanttensor",
        "layer_spec": layer_spec,
        "layers": layers,
        "layer_count": len(layers),
        "candidate": str(output.resolve()),
        "candidate_index_sha256": sha256_file(index),
        "overlay_sha256": sha256_file(overlay_path),
        "redirected_tensors": overlay["redirected_tensors"],
        "chunks": [
            {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in chunks
        ],
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--layer3-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            assemble(
                args.carrier,
                args.full_root,
                args.layer3_root,
                args.output,
                args.layers,
                args.receipt,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
