"""Build an NVFP4/MXFP6 layer-mixed checkpoint and exact payload receipt."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from safetensors import safe_open

from .shard_index import sha256_file


ROUTED_LAYERS = tuple(range(3, 45))
SUFFIXES = (".weight", ".weight_scale", ".weight_scale_2", ".input_scale")
DTYPE_BYTES = {
    "U8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "I32": 4,
    "I64": 8,
}
_SHA_CACHE: dict[Path, str] = {}


def _cached_sha256(path: Path) -> str:
    resolved = path.resolve()
    if resolved not in _SHA_CACHE:
        _SHA_CACHE[resolved] = sha256_file(resolved)
    return _SHA_CACHE[resolved]


def _slice_nbytes(handle, name: str) -> tuple[int, int]:
    view = handle.get_slice(name)
    count = math.prod(view.get_shape())
    dtype = view.get_dtype()
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"unsupported safetensors dtype {dtype} for {name}")
    return count * DTYPE_BYTES[dtype], count


def _parse_layers(spec: str) -> set[int]:
    layers = {int(x) for x in spec.replace(",", " ").split()}
    if not layers.issubset(set(ROUTED_LAYERS)):
        raise ValueError(f"invalid routed layer set: {sorted(layers)}")
    return layers


def build(carrier: Path, output: Path, chunks: list[Path], mxfp6_layers: set[int]) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    original = json.loads((carrier / "model.safetensors.index.json").read_text())
    weight_map = dict(original["weight_map"])
    redirected: dict[str, str] = {}
    payload_bytes = 0
    logical_elements = 0

    for item in carrier.iterdir():
        if item.name in {"model.safetensors.index.json", "config.json", "hf_quant_config.json", "OVERLAY.json"} or item.name.startswith(".cache"):
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
            meta = handle.metadata() or {}
            layer = int(meta["layer"])
            if layer not in mxfp6_layers:
                raise ValueError(f"chunk for non-MXFP6 layer {layer}: {chunk}")
            seen_layers.add(layer)
            for name in handle.keys():
                if name not in weight_map or not name.endswith(SUFFIXES):
                    raise KeyError(f"invalid replacement tensor {name}")
                nbytes, count = _slice_nbytes(handle, name)
                weight_map[name] = chunk.name
                redirected[name] = chunk.name
                payload_bytes += nbytes
                if name.endswith(".weight"):
                    # Packed FP6 holds four logical values in three bytes.
                    logical_elements += count * 4 // 3
    if seen_layers != mxfp6_layers:
        raise ValueError(f"missing MXFP6 chunks for layers {sorted(mxfp6_layers - seen_layers)}")

    # Add exact carrier NVFP4 payload for the layers left at the frozen rotated
    # endpoint. Input scales live in a shared shard but are still counted.
    carrier_names: dict[str, list[str]] = {}
    for name, shard in original["weight_map"].items():
        layer = next((l for l in ROUTED_LAYERS if f".layers.{l}.mlp.experts." in name), None)
        if layer is not None and layer not in mxfp6_layers and name.endswith(SUFFIXES):
            carrier_names.setdefault(shard, []).append(name)
    for shard, names in carrier_names.items():
        with safe_open(str(carrier / shard), framework="pt", device="cpu") as handle:
            for name in names:
                nbytes, count = _slice_nbytes(handle, name)
                payload_bytes += nbytes
                if name.endswith(".weight"):
                    logical_elements += count * 2

    updated = dict(original)
    updated["weight_map"] = weight_map
    (output / "model.safetensors.index.json").write_text(
        json.dumps(updated, indent=2, sort_keys=True) + "\n"
    )

    qconfig = json.loads((carrier / "config.json").read_text())
    qc = qconfig["quantization_config"]
    nv_targets = [f"model.language_model.layers.{l}.mlp.experts" for l in ROUTED_LAYERS if l not in mxfp6_layers]
    fp6_targets = [f"model.language_model.layers.{l}.mlp.experts" for l in ROUTED_LAYERS if l in mxfp6_layers]
    qc["quant_algo"] = "MIXED_PRECISION"
    # B12X reconstructs its exact source format from these global tags.  The
    # FP6 weight encoding is E2M3 and this campaign's declared runtime ladder
    # is W6A8 with dynamic E4M3 activations.
    qc["weight_format"] = "e2m3"
    qc["activation_format"] = "e4m3"
    qc["config_groups"]["group_nvfp4_routed_experts"]["targets"] = nv_targets
    qc["config_groups"]["group_mxfp6_routed_experts"] = {
        "targets": fp6_targets,
        "weights": {"type": "float", "num_bits": 6, "group_size": 32, "dynamic": False},
        "input_activations": {"type": "float", "num_bits": 8, "group_size": 32, "dynamic": True},
    }
    for layer in ROUTED_LAYERS:
        target = f"model.language_model.layers.{layer}.mlp.experts"
        qc["quantized_layers"][target] = {
            "quant_algo": "MXFP6" if layer in mxfp6_layers else "NVFP4",
            "group_size": 32 if layer in mxfp6_layers else 16,
        }
    (output / "config.json").write_text(json.dumps(qconfig, indent=2, sort_keys=True) + "\n")
    (output / "hf_quant_config.json").write_text(json.dumps(qc, indent=2, sort_keys=True) + "\n")

    overlay = {
        "schema": "glm53-nvfp4-v10.mixed-candidate-overlay.v1",
        "carrier": str(carrier.resolve()),
        "chunks": [str(p.resolve()) for p in chunks],
        "redirected_tensors": len(redirected),
        "required_load_format": "instanttensor",
    }
    (output / "OVERLAY.json").write_text(
        json.dumps(overlay, indent=2, sort_keys=True) + "\n"
    )

    receipt = {
        "schema": "glm53-nvfp4-v3.mixed-mxfp6-candidate.v1",
        "carrier": str(carrier.resolve()),
        "mxfp6_layers": sorted(mxfp6_layers),
        "nvfp4_layers": sorted(set(ROUTED_LAYERS) - mxfp6_layers),
        "chunks": [{"path": str(p.resolve()), "sha256": _cached_sha256(p)} for p in chunks],
        "redirected_tensors": len(redirected),
        "logical_elements": logical_elements,
        "payload_bytes": payload_bytes,
        "payload_bpw": 8.0 * payload_bytes / logical_elements,
        "source_format": "mxfp6_w6a8",
        "index_sha256": sha256_file(output / "model.safetensors.index.json"),
        "config_sha256": sha256_file(output / "config.json"),
        "hf_quant_config_sha256": sha256_file(output / "hf_quant_config.json"),
        "overlay_sha256": sha256_file(output / "OVERLAY.json"),
        "required_load_format": "instanttensor",
    }
    (output / "MIXED_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mxfp6-layers", required=True)
    parser.add_argument("--chunk", type=Path, action="append", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.carrier, args.output, args.chunk, _parse_layers(args.mxfp6_layers)), sort_keys=True))


if __name__ == "__main__":
    main()
