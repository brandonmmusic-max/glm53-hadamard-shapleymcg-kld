"""Freeze the three-layer coupled-scale P8 encoder preparation manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .p8_coupled_scale import (
    COUPLED_ACTIVATION,
    COUPLED_BOUNDARY,
    COUPLED_CAST_ORDER,
    COUPLED_FC1_INTERLEAVE,
    COUPLED_QUANTIZED_DOWN_ORDER,
    COUPLED_QUANTIZED_INPUT_ORDER,
    COUPLED_SIGN_DRAW,
    COUPLED_SIGN_GENERATOR,
    COUPLED_TP_SLICE,
    COUPLED_TRANSFORM_CONTRACT,
    COUPLED_TRANSFORM_ID,
    COUPLED_TRANSFORM_SHA256,
    SUPPORTED_LAYERS,
    _tensor_sha256,
    load_exact_exl3_scales,
)
from .shard_index import sha256_file


TARGET_EXPERTS = 288
TARGET_HIDDEN = 4096
TARGET_INTERMEDIATE = 2048
FROZEN_INTERMEDIATE_DRAW = COUPLED_SIGN_DRAW


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _validate_reap_capture(path: Path) -> dict[str, object]:
    manifest = json.loads(path.read_text())
    digest = manifest.get("capture_sha256")
    body = dict(manifest)
    body.pop("capture_sha256", None)
    if (
        not isinstance(digest, str)
        or hashlib.sha256(_canonical_json(body)).hexdigest() != digest
        or manifest.get("schema")
        != "quant-pipeline.glm53-main-calibration-capture.v1"
        or manifest.get("layers") != list(range(3, 45))
        or manifest.get("roles")
        != ["fit", "conditional-fit", "selection", "confirmation"]
        or manifest.get("geometry", {}).get("hidden_size") != TARGET_HIDDEN
        or manifest.get("geometry", {}).get("experts") != TARGET_EXPERTS
        or manifest.get("geometry", {}).get("top_k") != 8
    ):
        raise RuntimeError("REAP capture seal, role partition, or Flash geometry differs")
    return manifest


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
    scale_source = parser.add_mutually_exclusive_group(required=True)
    scale_source.add_argument("--exl3-scale", action="append")
    scale_source.add_argument("--exl3-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    capture_manifest = args.capture_root / "capture-manifest.json"
    capture = _validate_reap_capture(capture_manifest)
    scale_paths = (
        _parse_layer_paths(args.exl3_scale)
        if args.exl3_scale is not None
        else {layer: args.exl3_checkpoint for layer in SUPPORTED_LAYERS}
    )
    draws = {
        str(layer): [FROZEN_INTERMEDIATE_DRAW] * TARGET_EXPERTS
        for layer in SUPPORTED_LAYERS
    }

    encoder = Path(__file__).with_name("quantize_p8_coupled_scale_layer.py")
    reference = Path(__file__).with_name("p8_coupled_scale.py")
    trellis = Path(__file__).with_name("trellis_mxf.py")
    transform_receipt = (
        Path(__file__).resolve().parents[1]
        / "experiments/p8-coupled-transform-draw0-silu10-v1.json"
    )
    if sha256_file(transform_receipt) != COUPLED_TRANSFORM_SHA256:
        raise RuntimeError("coupled transform receipt differs from the code contract")
    inputs = {
        "encoder": _file(encoder),
        "coupled_reference": _file(reference),
        "trellis_codec": _file(trellis),
        "encoder_transform": _file(transform_receipt),
        "source_index": _file(args.source_index),
        "fit_roles": _file(args.roles),
        "capture_manifest": _file(capture_manifest),
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
            "kind": "checkpoint" if resolved.is_dir() else "layer-file",
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
        "schema": "glm53-p8.coupled-scale-preparation.v2",
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
        "stored_scale_bytes_per_rank_layer": 901120,
        "stored_draw_bytes_per_rank_layer": 288,
        "metadata_bpw": 0.003979859528718171,
        "runtime_regenerated_sign_bytes_per_rank_layer": 3072,
        "full_coupled_accounted_metadata_bpw": 0.003993422896773727,
        "full_coupled_accounted_bpw": 4.253993422896774,
        "boundary": COUPLED_BOUNDARY,
        "activation": COUPLED_ACTIVATION,
        "cast_order": COUPLED_CAST_ORDER,
        "quantized_input_order": COUPLED_QUANTIZED_INPUT_ORDER,
        "quantized_down_order": COUPLED_QUANTIZED_DOWN_ORDER,
        "transform_id": COUPLED_TRANSFORM_ID,
        "transform_contract": COUPLED_TRANSFORM_CONTRACT,
        "encoder_transform_sha256": COUPLED_TRANSFORM_SHA256,
        "sign_generator": COUPLED_SIGN_GENERATOR,
        "sign_draw": COUPLED_SIGN_DRAW,
        "sign_pre_axis": 1,
        "sign_post_axis": 2,
        "fc1_interleave": COUPLED_FC1_INTERLEAVE,
        "tp_slice": COUPLED_TP_SLICE,
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
            "capture_sha256": capture["capture_sha256"],
        },
        "execution_status": "preparation only; encoding, GPU, runtime, and KLD not launched",
        "ports": {
            "coupled_transform": "Luke B12X W4A8 trellis and local QSRT coupled H512/H128 reference",
            "scale_roles": "local GLM-5.3 EXL3 gate/up suh/svh and down suh/svh metadata",
            "trellis": "existing repository P8 K4 procedural MCG E4M3 codec",
        },
        "storage_budget": {
            "max_new_bytes": 30000000000,
            "weights_per_layer": 7247757312,
            "k4_4_25bpw_bytes_per_layer": 3850371072,
            "three_layer_weight_payload_bytes": 11551113216,
            "forecast_chunk_plus_tp4_bytes": 23123890176,
            "other_new_bytes_allowance": 6876109824,
            "dense_bf16_three_layer_bytes": 43486543872,
            "dense_output": "prohibited",
            "peak_gate": (
                "before every write require projected campaign bytes including "
                "safetensors headers, receipts, temporary files, and caches <=30000000000; "
                "also require filesystem free bytes >= projected remaining writes"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
