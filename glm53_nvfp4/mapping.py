"""Validate the immutable BF16-to-ModelOpt routed-expert tensor mapping."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .shard_index import IndexedCheckpoint

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def validate(source: Path, carrier: Path) -> dict:
    src = IndexedCheckpoint(source)
    dst = IndexedCheckpoint(carrier)
    errors = []
    layers = {}
    for layer in range(3, 45):
        source_names = []
        carrier_names = []
        shards = Counter()
        for expert in range(288):
            for projection in PROJECTIONS:
                stem = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}"
                source_key = stem + ".weight"
                expected = (stem + ".weight", stem + ".weight_scale", stem + ".weight_scale_2", stem + ".input_scale")
                if source_key not in src.weight_map:
                    errors.append(f"missing source {source_key}")
                else:
                    source_names.append(source_key)
                    shards[src.weight_map[source_key]] += 1
                for key in expected:
                    if key not in dst.weight_map:
                        errors.append(f"missing carrier {key}")
                    else:
                        carrier_names.append(key)
        layers[str(layer)] = {
            "source_weights": len(source_names),
            "carrier_tensors": len(carrier_names),
            "source_shards": dict(sorted(shards.items())),
        }
    return {
        "schema": "glm53-nvfp4-v2.mapping.v1",
        "source": str(source),
        "carrier": str(carrier),
        "layers": layers,
        "expected": {"layers": 42, "experts_per_layer": 288, "source_weights_per_layer": 864, "carrier_tensors_per_layer": 3456},
        "errors": errors,
        "passed": not errors and all(x["source_weights"] == 864 and x["carrier_tensors"] == 3456 for x in layers.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.source, args.carrier)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": result["passed"], "errors": len(result["errors"]), "output": str(args.output)}, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
