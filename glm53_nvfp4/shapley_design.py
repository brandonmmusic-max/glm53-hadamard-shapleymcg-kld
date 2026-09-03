"""Freeze deterministic antithetic layer-Shapley coalitions for the 6-bpw arm."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .shard_index import sha256_file


LAYERS = tuple(range(3, 45))


def coalition_id(layers: set[int]) -> str:
    payload = ",".join(map(str, sorted(layers))).encode()
    return "c-" + hashlib.sha256(payload).hexdigest()[:16]


def build(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    first = list(map(int, rng.permutation(LAYERS)))
    permutations = [first, list(reversed(first))]
    coalitions: dict[str, list[int]] = {}
    paths = []
    for permutation_index, order in enumerate(permutations):
        current: set[int] = set()
        path = [coalition_id(current)]
        coalitions[path[-1]] = []
        for layer in order:
            current.add(layer)
            cid = coalition_id(current)
            coalitions[cid] = sorted(current)
            path.append(cid)
        paths.append({"permutation_index": permutation_index, "order": order, "coalition_ids": path})
    return {
        "schema": "glm53-nvfp4-v3.layer-shapley-design.v1",
        "estimator": "two seeded antithetic permutation Monte Carlo paths",
        "seed": seed,
        "unit": "routed MoE layer",
        "unit_deviation": "Fused runtime kernels require one precision ABI per routed layer; expert/projection allocation is not executable in this runtime.",
        "value": "mean end-to-end teacher KLD reduction on conditional-fit when adding an MXFP6 layer to the rotated NVFP4 coalition",
        "layers": list(LAYERS),
        "permutations": paths,
        "coalitions": coalitions,
        "unique_coalitions": len(coalitions),
        "allocation": {
            "mxfp6_layers": 35,
            "nvfp4_layers": 7,
            "reason": "largest whole-layer assignment below exact 6.0 bpw after all per-matrix scale/input metadata",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--roles", type=Path, required=True)
    args = parser.parse_args()
    payload = build(args.seed)
    payload["roles_sha256"] = sha256_file(args.roles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "unique_coalitions": payload["unique_coalitions"]}, sort_keys=True))


if __name__ == "__main__":
    main()
