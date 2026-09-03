"""Materialize lightweight checkpoint indexes for every sealed Shapley coalition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .mixed_candidate import build
from .shard_index import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--chunk-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    design = json.loads(args.design.read_text())
    by_layer: dict[int, list[Path]] = {}
    for root in args.chunk_root:
        for path in root.glob("mxfp6-layer-*-experts-*.safetensors"):
            layer = int(path.name.split("-")[2])
            by_layer.setdefault(layer, []).append(path)
    for layer in design["layers"]:
        if len(by_layer.get(layer, [])) != 4:
            raise RuntimeError(f"layer {layer}: expected four MXFP6 chunks")

    rows = []
    for cid, layer_list in design["coalitions"].items():
        layers = set(layer_list)
        if not layers:
            rows.append({"coalition_id": cid, "layers": [], "path": str(args.carrier.resolve()), "carrier": True})
            continue
        chunks = sorted(path for layer in layers for path in by_layer[layer])
        output = args.output / cid
        receipt = build(args.carrier, output, chunks, layers)
        rows.append({
            "coalition_id": cid,
            "layers": sorted(layers),
            "path": str(output.resolve()),
            "carrier": False,
            "payload_bpw": receipt["payload_bpw"],
            "receipt_sha256": sha256_file(output / "MIXED_RECEIPT.json"),
        })
        print(json.dumps({"coalition": cid, "layers": len(layers)}), flush=True)
    payload = {
        "schema": "glm53-nvfp4-v3.shapley-candidate-set.v1",
        "design": str(args.design.resolve()),
        "design_sha256": sha256_file(args.design),
        "carrier": str(args.carrier.resolve()),
        "candidates": rows,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(args.manifest), "candidates": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
