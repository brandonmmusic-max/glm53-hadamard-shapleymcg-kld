"""Create a sparse checkpoint view that serves selected NVFP4 layers as W4A16."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .shard_index import sha256_file


def build(source: Path, output: Path, layers: set[int]) -> dict:
    if not layers or not layers.issubset(set(range(3, 45))):
        raise ValueError(f"invalid routed layer set {sorted(layers)}")
    output.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name in {"config.json", "hf_quant_config.json"} or item.name.startswith(".cache"):
            continue
        target = output / item.name
        if not target.exists() and not target.is_symlink():
            os.symlink(item.resolve(), target)

    config = json.loads((source / "config.json").read_text())
    qc = config["quantization_config"]
    targets = {f"model.language_model.layers.{layer}.mlp.experts" for layer in layers}
    for target in targets:
        info = qc["quantized_layers"].get(target)
        if info is None or info.get("quant_algo") != "NVFP4":
            raise RuntimeError(f"{target} is not an NVFP4 source layer")
        info["quant_algo"] = "W4A16_NVFP4"
    # The mixed-config loader intentionally coerces W4A16 back to W4A4 when a
    # target remains in a static-W4A4 config group.  Remove only selected
    # targets; quantized_layers remains the authoritative method mapping.
    for group in qc.get("config_groups", {}).values():
        group["targets"] = [target for target in group.get("targets", []) if target not in targets]

    config_path = output / "config.json"
    hf_path = output / "hf_quant_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    hf_path.write_text(json.dumps(qc, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema": "glm53-nvfp4-v4.w4a16-control.v1",
        "source": str(source.resolve()),
        "layers": sorted(layers),
        "weight_index_sha256": sha256_file(output / "model.safetensors.index.json"),
        "config_sha256": sha256_file(config_path),
        "hf_quant_config_sha256": sha256_file(hf_path),
        "weight_payload_unchanged": True,
    }
    (output / "W4A16_CONTROL.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output, set(args.layers)), sort_keys=True))


if __name__ == "__main__":
    main()
