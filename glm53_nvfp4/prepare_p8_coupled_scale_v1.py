"""Freeze the three-layer coupled-scale P8 encoder preparation manifest."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .p8_coupled_scale import (
    COUPLED_BOUNDARY,
    SUPPORTED_LAYERS,
    _tensor_sha256,
    load_exact_exl3_scales,
)
from .shard_index import sha256_file


TARGET_EXPERTS = 288
TARGET_HIDDEN = 4096
TARGET_INTERMEDIATE = 2048
FROZEN_INTERMEDIATE_DRAW = 0


def _file(path: Path) -> dict[str, object]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _parse_layer_paths(values: list[str]) -> dict[int, Path]:
    parsed: dict[int, Path] = {}
    for value in values:
        layer_text, separator, path_text = value.partition("=")
        if not separator or not layer_text.isdigit() or not path_text:
            raise ValueError("--exl3-scale must use LAYER=/absolute/path")
        layer = int(layer_text)
        if layer in parsed:
            raise ValueError(f"duplicate scale source for layer {layer}")
        parsed[layer] = Path(path_text)
    if set(parsed) != set(SUPPORTED_LAYERS):
        raise ValueError(f"scale sources must be exactly layers {SUPPORTED_LAYERS}")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--exl3-scale", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    scale_paths = _parse_layer_paths(args.exl3_scale)
    draws = {
        str(layer): [FROZEN_INTERMEDIATE_DRAW] * TARGET_EXPERTS
        for layer in SUPPORTED_LAYERS
    }

    encoder = Path(__file__).with_name("quantize_p8_coupled_scale_layer.py")
    reference = Path(__file__).with_name("p8_coupled_scale.py")
    trellis = Path(__file__).with_name("trellis_mxf.py")
    inputs = {
        "encoder": _file(encoder),
        "coupled_reference": _file(reference),
        "trellis_codec": _file(trellis),
        "source_index": _file(args.source_index),
        "fit_roles": _file(args.roles),
        "capture_manifest": _file(args.capture_root / "capture-manifest.json"),
        "exl3_scale_sources": {},
    }
    scale_receipts: dict[str, object] = {}
    for layer, path in scale_paths.items():
        # Validation precedes design creation. No resizing, repetition, or
        # cross-model substitution is permitted.
        loaded = load_exact_exl3_scales(
            path,
            layer=layer,
            expected_experts=TARGET_EXPERTS,
            expected_hidden=TARGET_HIDDEN,
            expected_intermediate=TARGET_INTERMEDIATE,
        )
        resolved = path.resolve()
        inputs["exl3_scale_sources"][str(layer)] = {
            "path": str(resolved),
            "bytes": resolved.stat().st_size,
            "sha256": loaded.source_sha256,
        }
        scale_receipts[str(layer)] = {
            "source_metadata": loaded.source_metadata,
            "raw_tensor_sha256": loaded.tensor_hashes,
            "aggregate_tensor_sha256": {
                name: _tensor_sha256(tensor)
                for name, tensor in loaded.tensors().items()
            },
        }

    payload = {
        "schema": "glm53-p8.coupled-scale-preparation.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision_before_result": True,
        "layers": list(SUPPORTED_LAYERS),
        "experts_per_layer": TARGET_EXPERTS,
        "geometry": {
            "hidden": TARGET_HIDDEN,
            "intermediate": TARGET_INTERMEDIATE,
            "tp": 4,
        },
        "bits": 4,
        "weight_payload_bpw": 4.25,
        "metadata_bpw": 0.003979859528718171,
        "boundary": COUPLED_BOUNDARY,
        "cast_order": "bf16-fp16-h512-fp16-suh-h128-e4m3",
        "encoder": "GPTQ-style inter-group Hessian feedback; static in-group activation order",
        "ldlq": False,
        "draw_policy": "QSRT default draw zero for every expert; no draw tuning",
        "intermediate_draws_by_layer": draws,
        "inputs": inputs,
        "scale_receipts": scale_receipts,
        "data_policy": {
            "opened": ["fit"],
            "protected": ["selection", "confirmation", "final"],
            "protected_roles_opened": [],
            "sampling": "domain-balanced routes per expert",
        },
        "execution_status": "preparation only; encoding, GPU, runtime, and KLD not launched",
        "ports": {
            "coupled_transform": "Luke B12X W4A8 trellis and local QSRT coupled H512/H128 reference",
            "scale_roles": "local GLM-5.3 EXL3 gate/up suh/svh and down suh/svh metadata",
            "trellis": "existing repository P8 K4 procedural MCG E4M3 codec",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
