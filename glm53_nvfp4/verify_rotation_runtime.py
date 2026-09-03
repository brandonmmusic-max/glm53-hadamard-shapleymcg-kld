"""Verify first-forward evidence for every rotated layer on every runtime rank."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


MARKER = re.compile(
    r"GLM53_BLOCK_ROTATION_FORWARD layer=(?P<layer>\d+) rank=(?P<rank>\d+) "
    r"rotation_sha256=(?P<sha>[0-9a-f]{64}) hidden_width=(?P<width>\d+) "
    r"dtype=(?P<dtype>torch\.[a-z0-9_]+)"
)


def parse_layers(spec: str) -> set[int]:
    result: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if "-" in item:
            first, last = (int(x) for x in item.split("-", 1))
            result.update(range(first, last + 1))
        elif item:
            result.add(int(item))
    if not result:
        raise ValueError("empty layer specification")
    return result


def verify(text: str, layers: set[int], ranks: set[int]) -> dict:
    rows = [m.groupdict() for m in MARKER.finditer(text)]
    observed = {(int(row["layer"]), int(row["rank"])) for row in rows}
    expected = {(layer, rank) for layer in layers for rank in ranks}
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    bad_geometry = sorted(
        (int(row["layer"]), int(row["rank"]), int(row["width"]), row["dtype"])
        for row in rows
        if int(row["width"]) != 4096 or row["dtype"] != "torch.bfloat16"
    )
    if missing or unexpected or bad_geometry:
        raise RuntimeError(
            f"rotation forward evidence mismatch missing={missing} "
            f"unexpected={unexpected} bad_geometry={bad_geometry}"
        )
    by_layer = {
        str(layer): sorted(
            {
                row["sha"]
                for row in rows
                if int(row["layer"]) == layer
            }
        )
        for layer in sorted(layers)
    }
    if any(len(shas) != 1 for shas in by_layer.values()):
        raise RuntimeError(f"rotation checksum differs across ranks: {by_layer}")
    return {
        "schema": "glm53-nvfp4-v4.rotation-runtime-proof.v1",
        "status": "pass",
        "layers": sorted(layers),
        "ranks": sorted(ranks),
        "forward_pairs": len(observed),
        "rotation_sha256_by_layer": by_layer,
        "hidden_width": 4096,
        "dtype": "torch.bfloat16",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.log.read_text(errors="replace"), parse_layers(args.layers), set(range(args.ranks)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
