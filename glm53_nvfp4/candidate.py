"""Create a sparse-overlay checkpoint by redirecting carrier index entries to candidate chunks."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from safetensors import safe_open

ALLOWED_SUFFIXES = (".weight", ".weight_scale", ".weight_scale_2")
ROUTED_LAYERS = range(3, 45)


def routed_expert_payload_names(weight_map: dict[str, str]) -> set[str]:
    prefixes = tuple(f"model.language_model.layers.{layer}.mlp.experts." for layer in ROUTED_LAYERS)
    return {
        name
        for name in weight_map
        if name.startswith(prefixes)
        and name.endswith(ALLOWED_SUFFIXES)
        and not name.endswith(".input_scale")
    }


def build(carrier: Path, output: Path, chunks: list[Path]) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    original = json.loads((carrier / "model.safetensors.index.json").read_text())
    weight_map = dict(original["weight_map"])
    redirected = {}
    for item in carrier.iterdir():
        if item.name == "model.safetensors.index.json" or item.name.startswith(".cache"):
            continue
        target = output / item.name
        if target.exists() or target.is_symlink():
            continue
        os.symlink(item.resolve(), target)
    for chunk in chunks:
        target = output / chunk.name
        if target.exists() or target.is_symlink():
            if target.resolve() != chunk.resolve():
                raise FileExistsError(target)
        else:
            os.symlink(chunk.resolve(), target)
        with safe_open(str(chunk), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name not in weight_map:
                    raise KeyError(f"candidate tensor absent from carrier index: {name}")
                if not name.endswith(ALLOWED_SUFFIXES):
                    raise ValueError(f"candidate tensor has forbidden suffix: {name}")
                weight_map[name] = chunk.name
                redirected[name] = chunk.name
    updated = dict(original)
    updated["weight_map"] = weight_map
    (output / "model.safetensors.index.json").write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n")
    overlay = {"schema": "glm53-nvfp4-v2.candidate-overlay.v1", "carrier": str(carrier.resolve()), "chunks": [str(x.resolve()) for x in chunks], "redirected_tensors": len(redirected)}
    (output / "OVERLAY.json").write_text(json.dumps(overlay, indent=2, sort_keys=True) + "\n")
    return overlay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk", type=Path, action="append", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.carrier, args.output, args.chunk), sort_keys=True))


if __name__ == "__main__":
    main()
