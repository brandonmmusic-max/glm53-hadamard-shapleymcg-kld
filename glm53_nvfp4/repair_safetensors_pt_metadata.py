"""Create a payload-identical safetensors copy with the reserved format=pt tag."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

from safetensors import safe_open

from .shard_index import sha256_file


CHUNK = 32 * 1024 * 1024


def repair(source: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    with source.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise RuntimeError("truncated safetensors prefix")
        header_bytes = struct.unpack("<Q", prefix)[0]
        header = handle.read(header_bytes)
        if len(header) != header_bytes:
            raise RuntimeError("truncated safetensors header")
        parsed = json.loads(header)
        metadata = parsed.get("__metadata__", {})
        source_format = metadata.get("format")
        if source_format != "ModelOpt NVFP4 E2M1/E4M3-per-16/FP32-global":
            raise RuntimeError(f"unexpected source format tag: {source_format}")
        metadata["format"] = "pt"
        metadata["weight_format"] = source_format
        encoded = json.dumps(parsed, separators=(",", ":"), sort_keys=True).encode()
        repaired_header_bytes = (len(encoded) + 7) // 8 * 8
        repaired_header = encoded + b" " * (repaired_header_bytes - len(encoded))
        repaired_prefix = struct.pack("<Q", repaired_header_bytes)

        output.parent.mkdir(parents=True, exist_ok=True)
        source_hash = hashlib.sha256(prefix + header)
        output_hash = hashlib.sha256(repaired_prefix + repaired_header)
        payload_hash = hashlib.sha256()
        payload_bytes = 0
        with output.open("xb") as target:
            target.write(repaired_prefix)
            target.write(repaired_header)
            while True:
                block = handle.read(CHUNK)
                if not block:
                    break
                target.write(block)
                source_hash.update(block)
                output_hash.update(block)
                payload_hash.update(block)
                payload_bytes += len(block)

    with safe_open(str(output), framework="pt", device="cpu") as repaired:
        repaired_meta = repaired.metadata() or {}
        if repaired_meta.get("format") != "pt" or repaired_meta.get("weight_format") != source_format:
            raise RuntimeError("repaired metadata did not round-trip")
        tensor_count = len(list(repaired.keys()))
    return {
        "schema": "glm53-rotation-v6.safetensors-pt-header-repair.v1",
        "status": "pass",
        "source": str(source.resolve()),
        "source_sha256": source_hash.hexdigest(),
        "output": str(output.resolve()),
        "output_sha256": output_hash.hexdigest(),
        "source_header_bytes": header_bytes,
        "output_header_bytes": repaired_header_bytes,
        "payload_bytes": payload_bytes,
        "payload_sha256": payload_hash.hexdigest(),
        "payload_identity": "the output payload is byte-for-byte copied from source",
        "tensor_count": tensor_count,
        "source_format": source_format,
        "output_format": "pt",
        "repair_code_sha256": sha256_file(Path(__file__)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(f"refusing to overwrite {args.receipt}")
    result = repair(args.source, args.output)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
