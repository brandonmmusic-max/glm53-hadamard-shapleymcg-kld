"""Build a sparse mixed checkpoint with selected routed layers served as BF16."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from safetensors import safe_open

from .shard_index import sha256_file


ROUTED_LAYERS = tuple(range(3, 45))


def _layer_for(name: str) -> int | None:
    for layer in ROUTED_LAYERS:
        if f".layers.{layer}.mlp.experts." in name:
            return layer
    return None


def build(carrier: Path, output: Path, chunks: list[Path], bf16_layers: set[int]) -> dict:
    if not bf16_layers or not bf16_layers.issubset(set(ROUTED_LAYERS)):
        raise ValueError(f"invalid BF16 layer set: {sorted(bf16_layers)}")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to reuse BF16 overlay destination: {output}")
    output.mkdir(parents=True)
    original = json.loads((carrier / "model.safetensors.index.json").read_text())
    weight_map = dict(original["weight_map"])
    redirected: dict[str, str] = {}
    logical_elements = 0
    bf16_payload_bytes = 0

    for item in carrier.iterdir():
        if item.name in {
            "model.safetensors.index.json",
            "config.json",
            "hf_quant_config.json",
            "OVERLAY.json",
        } or item.name.startswith(".cache"):
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
            if layer not in bf16_layers:
                raise ValueError(f"chunk for undeclared BF16 layer {layer}: {chunk}")
            seen_layers.add(layer)
            for name in handle.keys():
                if name not in weight_map or not name.endswith(".weight"):
                    raise KeyError(f"invalid BF16 replacement tensor {name}")
                view = handle.get_slice(name)
                if view.get_dtype() != "BF16":
                    raise ValueError(f"replacement is not BF16: {name}")
                count = math.prod(view.get_shape())
                logical_elements += count
                bf16_payload_bytes += 2 * count
                weight_map[name] = chunk.name
                redirected[name] = chunk.name
    if seen_layers != bf16_layers:
        raise ValueError(f"missing BF16 chunks for layers {sorted(bf16_layers - seen_layers)}")

    removed_quantization_tensors = []
    for name in list(weight_map):
        if (
            _layer_for(name) in bf16_layers
            and not name.endswith(".weight")
            and name.endswith((".weight_scale", ".weight_scale_2", ".input_scale"))
        ):
            removed_quantization_tensors.append(name)
            del weight_map[name]

    updated = dict(original)
    updated["weight_map"] = weight_map
    index_path = output / "model.safetensors.index.json"
    index_path.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n")

    config = json.loads((carrier / "config.json").read_text())
    qc = config["quantization_config"]
    targets = {f"model.language_model.layers.{layer}.mlp.experts" for layer in bf16_layers}
    for target in targets:
        if target not in qc["quantized_layers"]:
            raise RuntimeError(f"carrier lacks quantized routed target {target}")
        del qc["quantized_layers"][target]
    for group in qc.get("config_groups", {}).values():
        group["targets"] = [target for target in group.get("targets", []) if target not in targets]
    config_path = output / "config.json"
    hf_path = output / "hf_quant_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    hf_path.write_text(json.dumps(qc, indent=2, sort_keys=True) + "\n")

    overlay = {
        "schema": "glm53-trellis-mxf.bf16-layer-overlay.v1",
        "carrier": str(carrier.resolve()),
        "chunks": [str(path.resolve()) for path in chunks],
        "bf16_layers": sorted(bf16_layers),
        "redirected_tensors": len(redirected),
        "removed_quantization_tensors": len(removed_quantization_tensors),
        "required_load_format": "instanttensor",
    }
    overlay_path = output / "OVERLAY.json"
    overlay_path.write_text(json.dumps(overlay, indent=2, sort_keys=True) + "\n")
    receipt = {
        **overlay,
        "logical_elements": logical_elements,
        "bf16_payload_bytes": bf16_payload_bytes,
        "bf16_layer_payload_bpw": 8.0 * bf16_payload_bytes / logical_elements,
        "chunks": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in chunks
        ],
        "index_sha256": sha256_file(index_path),
        "config_sha256": sha256_file(config_path),
        "hf_quant_config_sha256": sha256_file(hf_path),
        "overlay_sha256": sha256_file(overlay_path),
        "interpretation": "weight-only pseudoquant diagnostic; not a native 4.25-bpw runtime",
    }
    (output / "BF16_LAYER_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bf16-layers", nargs="+", type=int, required=True)
    parser.add_argument("--chunk", type=Path, action="append", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.carrier, args.output, args.chunk, set(args.bf16_layers)), sort_keys=True))


if __name__ == "__main__":
    main()
