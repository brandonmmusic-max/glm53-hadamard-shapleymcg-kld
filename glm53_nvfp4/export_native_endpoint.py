"""Export a research codec artifact as a hot-path-only native NVFP4 endpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file


NATIVE_SUFFIXES = (".weight", ".weight_scale", ".weight_scale_2", ".input_scale")


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def export_native_endpoint(source: Path, output: Path) -> dict[str, object]:
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        source_metadata = handle.metadata() or {}
    tensors = load_file(str(source), device="cpu")
    native = {
        name: tensor.contiguous()
        for name, tensor in tensors.items()
        if name.endswith(NATIVE_SUFFIXES) and ".experts." in name
    }
    weight_names = sorted(
        name for name in native if name.endswith(".weight")
    )
    if not weight_names:
        raise ValueError("source contains no native expert NVFP4 endpoints")
    for weight_name in weight_names:
        stem = weight_name[: -len(".weight")]
        required = {
            weight_name,
            f"{stem}.weight_scale",
            f"{stem}.weight_scale_2",
            f"{stem}.input_scale",
        }
        missing = required.difference(native)
        if missing:
            raise ValueError(f"incomplete NVFP4 endpoint for {stem}: {sorted(missing)}")

    packed_weight_bytes = sum(native[name].numel() for name in weight_names)
    logical_weights = packed_weight_bytes * 2
    payload_bytes = sum(tensor.numel() * tensor.element_size() for tensor in native.values())
    payload_bpw = 8.0 * payload_bytes / logical_weights
    metadata = {
        "schema": "glm53-native-nvfp4-endpoint.v1",
        "source_artifact_sha256": sha256_file(source),
        "source_codec_schema": source_metadata.get("schema", "unknown"),
        "hot_path": "ordinary packed E2M1 weights plus UE4M3 per-16 scales",
        "runtime_trellis_decode": "false",
        "runtime_selector_lookup": "false",
        "runtime_second_mma": "false",
        "payload_bpw": f"{payload_bpw:.9f}",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(native, str(output), metadata=metadata)
    receipt = {
        "schema": "glm53-native-nvfp4-endpoint-export.v1",
        "source": {
            "path": str(source),
            "sha256": sha256_file(source),
        },
        "output": {
            "path": str(output),
            "sha256": sha256_file(output),
            "file_bytes": output.stat().st_size,
            "payload_bytes": payload_bytes,
            "payload_bpw": payload_bpw,
        },
        "logical_weights": logical_weights,
        "projection_count": len(weight_names),
        "tensor_names": sorted(native),
        "excluded_runtime_state": sorted(set(tensors).difference(native)),
        "runtime_claim": "format-compatible native NVFP4 endpoint; kernel execution and speed remain to be measured",
    }
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = export_native_endpoint(args.source, args.output)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt["output"], sort_keys=True))


if __name__ == "__main__":
    main()
