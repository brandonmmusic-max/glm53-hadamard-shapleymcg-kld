"""Strict CPU interchange from native-RNE matrix streams to GLM TP4 P4.

The tensor order is deliberately shared with the P4 kernel, not the P8 FC1
scale repack. All W13 planes use [gate, up]. This module never imports CUDA.
See docs/P4_SERVING_BRIDGE.md for arithmetic, provenance and claim limits.
"""
from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import time

import numpy as np

from .p4_codec import (
    P4Payload, _array, _unique_object, e4m3_scale_values,
    read_p4, validate_payload,
)


RANK_SCHEMA = "glm53-p4-mcg-tp-rank.v2"
MANIFEST_SCHEMA = "glm53-p4-mcg-tp4-manifest.v1"
MATRIX_MANIFEST_SCHEMA = "glm53-p4-mcg-matrix-manifest.v1"
LAW = "procedural-mcg-alpha1-rne-e2m1"
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
RANK_CONTRACT = {
    "schema": RANK_SCHEMA, "role": "physical-codec", "world_size": "4",
    "bits": "4", "state_bits": "16", "tile_values": "256",
    "state_boundary": "cyclic-per-tile", "alphabet": "e2m1",
    "scale": "e4m3-k16", "law": LAW, "compander": "1",
    "weight_rounding": "nearest-even-satfinite", "signed_zero": "preserve",
    "mcg_arithmetic": "u32-wrap-mask-xor-add-rn-f16", "byte_order": "little",
    "trellis_layout": "k16-n16-exl3-lane-pair-swapped-i16",
    "scale_layout": "row-major-n-k16", "global_scale": "positive-f32-per-projection-expert",
    "boundary": "identity", "w13_order": "gate,up",
    "expert_order": "global-contiguous-zero-based", "tp_sharding": "gate-up-rows-down-columns",
    "state_lut_bytes": "0", "ldlq": "false",
}
# Stable data-region order; sorted JSON keys do not change this order.
TENSOR_DTYPES = {
    "w13_global_scale": ("F32", "<f4"), "w2_global_scale": ("F32", "<f4"),
    "w13_scale_e4m3": ("U8", "u1"), "w2_scale_e4m3": ("U8", "u1"),
    "w13_trellis": ("I16", "<i2"), "w2_trellis": ("I16", "<i2"),
}


def _sha(value: str, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} requires a lowercase SHA-256")
    return value


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def _dimensions(experts: int, hidden: int, intermediate: int) -> None:
    if any(type(v) is not int or v <= 0 for v in (experts, hidden, intermediate)):
        raise ValueError("P4 dimensions must be positive integers")
    if experts > 288 or hidden % 64 or intermediate % 256:
        raise ValueError("P4 requires <=288 experts, H divisible by 64 and full I divisible by 256")


def rank_metadata(*, layer: int, rank: int, experts: int, hidden: int,
                  intermediate: int, source_design_sha256: str) -> dict[str, str]:
    """Describe full I; each rank stores I/4 intermediate features."""
    _dimensions(experts, hidden, intermediate)
    if type(layer) is not int or not 3 <= layer <= 44 or type(rank) is not int or not 0 <= rank < 4:
        raise ValueError("P4 requires GLM layer 3..44 and TP4 rank 0..3")
    _sha(source_design_sha256, "source design")
    return {**RANK_CONTRACT, "layer": str(layer), "rank": str(rank),
            "experts": str(experts), "hidden": str(hidden),
            "intermediate": str(intermediate), "intermediate_per_rank": str(intermediate // 4),
            "source_design_sha256": source_design_sha256}


def tensor_specs(experts: int, hidden: int, intermediate: int) -> dict[str, tuple]:
    _dimensions(experts, hidden, intermediate)
    local = intermediate // 4
    shapes = {
        "w13_global_scale": (2, experts), "w2_global_scale": (experts,),
        "w13_scale_e4m3": (2, experts, local, hidden // 16),
        "w2_scale_e4m3": (experts, hidden, local // 16),
        "w13_trellis": (2, experts, hidden // 16, local // 16, 64),
        "w2_trellis": (experts, local // 16, hidden // 16, 64),
    }
    return {name: (*TENSOR_DTYPES[name], shapes[name]) for name in TENSOR_DTYPES}


def _metadata_dimensions(metadata: Mapping[str, str]) -> tuple[int, int, int, int, int]:
    try:
        fields = ("layer", "rank", "experts", "hidden", "intermediate")
        if any(not isinstance(metadata[field], str) for field in fields):
            raise ValueError("dimension metadata must use decimal strings")
        numbers = tuple(int(metadata[field]) for field in fields)
        if any(metadata[field] != str(value) for field, value in zip(fields, numbers, strict=True)):
            raise ValueError("dimension metadata is not canonical")
        return numbers
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid rank dimension metadata") from error


def validate_rank(tensors: Mapping[str, np.ndarray], metadata: Mapping[str, str], *,
                  expected_layer: int | None = None, expected_rank: int | None = None,
                  expected_design_sha256: str | None = None,
                  expected_experts: int | None = None) -> tuple[int, int, int]:
    """Return (E,H,I/4). Accept only the exact base or hashed metadata inventory."""
    layer, rank, experts, hidden, intermediate = _metadata_dimensions(metadata)
    canonical = rank_metadata(layer=layer, rank=rank, experts=experts, hidden=hidden,
                              intermediate=intermediate,
                              source_design_sha256=metadata.get("source_design_sha256", ""))
    hash_keys = {name + "_sha256" for name in TENSOR_DTYPES}
    if set(metadata) not in (set(canonical), set(canonical) | hash_keys):
        raise ValueError("P4 rank metadata inventory mismatch")
    if any(metadata[key] != value for key, value in canonical.items()):
        raise ValueError("P4 rank metadata contract mismatch")
    for expected, actual, label in ((expected_layer, layer, "layer"),
                                    (expected_rank, rank, "rank"),
                                    (expected_design_sha256, metadata["source_design_sha256"], "design"),
                                    (expected_experts, experts, "experts")):
        if expected is not None and expected != actual:
            raise ValueError(f"P4 consumer {label} mismatch")
    if set(tensors) != set(TENSOR_DTYPES):
        raise ValueError("P4 rank requires exactly six physical tensors")
    for name, (_, dtype, shape) in tensor_specs(experts, hidden, intermediate).items():
        value = _array(tensors[name], dtype, name, shape)
        if "scale_e4m3" in name:
            e4m3_scale_values(value)
        if "global_scale" in name and (not np.isfinite(value).all() or np.any(value <= 0)):
            raise ValueError("P4 globals must be positive finite FP32 values")
        if name + "_sha256" in metadata:
            expected_hash = _sha(metadata[name + "_sha256"], name)
            if hashlib.sha256(value.tobytes()).hexdigest() != expected_hash:
                raise ValueError(f"P4 rank tensor hash mismatch: {name}")
    return experts, hidden, intermediate // 4


def serialize_rank(tensors: Mapping[str, np.ndarray], metadata: Mapping[str, str]) -> bytes:
    validate_rank(tensors, metadata)
    base = {key: value for key, value in metadata.items() if key not in {n + "_sha256" for n in TENSOR_DTYPES}}
    header = {"__metadata__": base}
    data, offset = [], 0
    for name, (tag, _) in TENSOR_DTYPES.items():
        value = tensors[name]
        raw = value.tobytes()
        base[name + "_sha256"] = hashlib.sha256(raw).hexdigest()
        header[name] = {"dtype": tag, "shape": list(value.shape), "data_offsets": [offset, offset + len(raw)]}
        data.append(raw)
        offset += len(raw)
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + b"".join(data)


def deserialize_rank(raw: bytes, *, expected_sha256: str | None = None, **expectations):
    if not isinstance(raw, bytes) or len(raw) < 8:
        raise ValueError("truncated P4 rank container")
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != _sha(expected_sha256, "rank file"):
        raise ValueError("P4 rank file hash mismatch")
    header_size = struct.unpack_from("<Q", raw)[0]
    if header_size % 8 or not 0 < header_size <= min(len(raw) - 8, 1 << 20):
        raise ValueError("invalid P4 rank header size")
    try:
        header = json.loads(raw[8:8 + header_size], object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError) as error:
        raise ValueError("invalid P4 rank JSON") from error
    if not isinstance(header, dict) or set(header) != {"__metadata__", *TENSOR_DTYPES}:
        raise ValueError("P4 rank tensor inventory mismatch")
    metadata = header["__metadata__"]
    if not isinstance(metadata, dict):
        raise ValueError("P4 rank metadata must be an object")
    _, _, experts, hidden, intermediate = _metadata_dimensions(metadata)
    tensors, offset = {}, 0
    for name, (tag, dtype, shape) in tensor_specs(experts, hidden, intermediate).items():
        size = math.prod(shape) * np.dtype(dtype).itemsize
        if header[name] != {"dtype": tag, "shape": list(shape), "data_offsets": [offset, offset + size]}:
            raise ValueError(f"P4 rank tensor shape/type/offset mismatch: {name}")
        data = raw[8 + header_size + offset:8 + header_size + offset + size]
        if len(data) != size:
            raise ValueError("truncated P4 rank data")
        tensors[name] = np.frombuffer(data, dtype=dtype).copy().reshape(shape)
        offset += size
    if 8 + header_size + offset != len(raw):
        raise ValueError("P4 rank trailing bytes")
    validate_rank(tensors, metadata, **expectations)
    if serialize_rank(tensors, metadata) != raw:
        raise ValueError("noncanonical P4 rank container")
    return tensors, metadata


def read_rank(path: Path, **expectations):
    return deserialize_rank(Path(path).read_bytes(), **expectations)


def rank_accounting(tensors, metadata, *, serialized: bytes | None = None) -> dict:
    experts, hidden, local = validate_rank(tensors, metadata)
    weights = 3 * experts * hidden * local
    raw = serialize_rank(tensors, metadata) if serialized is None else serialized
    components = {name: value.nbytes for name, value in tensors.items()}
    payload = sum(components.values())
    globals_bytes = components["w13_global_scale"] + components["w2_global_scale"]
    if (payload - globals_bytes) * 8 != weights * 9 // 2:
        raise ValueError("P4 stream-plus-scale storage invariant failed")
    result = {"logical_weights": weights, "tensor_bytes": components,
              "runtime_state_lut_bytes": 0, "global_scale_bytes": globals_bytes,
              "payload_bytes": payload, "container_bytes": len(raw) - payload, "file_bytes": len(raw)}
    for name, amount in (("stream_and_scales", payload - globals_bytes), ("payload", payload), ("file", len(raw))):
        value = Fraction(amount * 8, weights)
        result[name + "_bpw"] = {"numerator": value.numerator, "denominator": value.denominator, "decimal": float(value)}
    return result


def assemble_rank(experts: list[Mapping[str, P4Payload]], *, layer: int, rank: int,
                  source_design_sha256: str):
    """Slice matrix streams on tile boundaries; no state reconstruction or re-encoding."""
    if not experts or set(experts[0]) != set(PROJECTIONS):
        raise ValueError("all three projections are required")
    hidden, intermediate = experts[0]["gate_proj"].width, experts[0]["gate_proj"].rows
    metadata = rank_metadata(layer=layer, rank=rank, experts=len(experts), hidden=hidden,
                             intermediate=intermediate, source_design_sha256=source_design_sha256)
    tensors = {name: np.empty(shape, dtype=dtype) for name, (_, dtype, shape) in tensor_specs(len(experts), hidden, intermediate).items()}
    for expert, projections in enumerate(experts):
        install_expert(tensors, projections, expert=expert, rank=rank, hidden=hidden, intermediate=intermediate)
    validate_rank(tensors, metadata)
    return tensors, metadata


def install_expert(tensors, projections, *, expert: int, rank: int, hidden: int, intermediate: int):
    if set(projections) != set(PROJECTIONS):
        raise ValueError("expert requires exactly gate_proj, up_proj and down_proj")
    local, start = intermediate // 4, rank * (intermediate // 4)
    for p, name in enumerate(PROJECTIONS):
        matrix = projections[name]
        validate_payload(matrix)
        expected = (hidden, intermediate) if name == "down_proj" else (intermediate, hidden)
        if (matrix.rows, matrix.width) != expected:
            raise ValueError(f"inconsistent expert shape: {name}")
        if name == "down_proj":
            tensors["w2_trellis"][expert] = matrix.trellis[start // 16:(start + local) // 16]
            tensors["w2_scale_e4m3"][expert] = matrix.scale_e4m3[:, start // 16:(start + local) // 16]
            tensors["w2_global_scale"][expert] = matrix.global_scale
        else:
            tensors["w13_trellis"][p, expert] = matrix.trellis[:, start // 16:(start + local) // 16]
            tensors["w13_scale_e4m3"][p, expert] = matrix.scale_e4m3[start:start + local]
            tensors["w13_global_scale"][p, expert] = matrix.global_scale


def rank_projection(tensors, metadata, *, expert: int, projection: str) -> P4Payload:
    """CPU oracle view of one rank projection; values retain the stored global."""
    experts, hidden, local = validate_rank(tensors, metadata)
    if type(expert) is not int or not 0 <= expert < experts or projection not in PROJECTIONS:
        raise ValueError("invalid expert or projection")
    if projection == "down_proj":
        return P4Payload(hidden, local, tensors["w2_trellis"][expert].copy(),
                         tensors["w2_scale_e4m3"][expert].copy(),
                         np.array(tensors["w2_global_scale"][expert], dtype="<f4"))
    p = PROJECTIONS.index(projection)
    return P4Payload(local, hidden, tensors["w13_trellis"][p, expert].copy(),
                     tensors["w13_scale_e4m3"][p, expert].copy(),
                     np.array(tensors["w13_global_scale"][p, expert], dtype="<f4"))


def _input_manifest(path: Path, expected_design_sha256: str) -> dict:
    manifest = json.loads(path.read_text(), object_pairs_hook=_unique_object)
    if (manifest.get("schema") != MATRIX_MANIFEST_SCHEMA
            or manifest.get("law") != LAW or manifest.get("ldlq") is not False
            or manifest.get("source_design_sha256") != expected_design_sha256):
        raise ValueError("P4 matrix manifest contract/design mismatch")
    _dimensions(manifest["experts"], manifest["hidden"], manifest["intermediate"])
    if type(manifest["layer"]) is not int or not 3 <= manifest["layer"] <= 44:
        raise ValueError("P4 matrix manifest layer invalid")
    return manifest


def _matrix_path(parent: Path, text: str) -> Path:
    candidate = Path(text)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise ValueError("matrix paths must stay relative to their manifest")
    resolved = (parent / candidate).resolve()
    if not resolved.is_relative_to(parent.resolve()):
        raise ValueError("matrix path escapes manifest directory")
    return resolved


def export_tp4(manifest_paths: list[Path] | Path, output_dir: Path, *,
               expected_design_sha256: str, resume: bool = False) -> dict:
    """Export complete layer(s), reading each source expert once for all ranks.

    Input manifests may be disjoint expert chunks. An existing output directory
    is allowed, but completed weights are never overwritten. Resume requires
    identical source-plan hashes and validates each completed rank against its
    receipt and canonical container before skipping it. Partial files survive
    failures; only complete files with receipts are reusable.
    """
    started = time.perf_counter()
    timing = {"schema": "glm53-p4-export-timings.v1", "source_matrix_reads": 0,
              "source_matrix_bytes": 0, "reused_ranks": 0, "written_ranks": 0,
              "stages": []}
    _sha(expected_design_sha256, "source design")
    paths = [manifest_paths] if isinstance(manifest_paths, Path) else list(manifest_paths)
    if not paths:
        raise ValueError("at least one matrix manifest is required")
    layers, sources = {}, []
    for path in paths:
        path = Path(path)
        data = _input_manifest(path, expected_design_sha256)
        layer = data["layer"]
        identity = (data["experts"], data["hidden"], data["intermediate"])
        state = layers.setdefault(layer, {"identity": identity, "matrices": {}})
        if state["identity"] != identity:
            raise ValueError("mixed layer matrix dimensions")
        sources.append({"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": file_sha256(path)})
        for entry in data["matrices"]:
            expert, projection = entry["expert"], entry["projection"]
            if type(expert) is not int or not 0 <= expert < identity[0] or projection not in PROJECTIONS:
                raise ValueError("invalid matrix manifest projection key")
            key = expert, projection
            if key in state["matrices"]:
                raise ValueError("duplicate expert projection")
            matrix_path = _matrix_path(path.parent, entry["path"])
            if type(entry["bytes"]) is not int or matrix_path.stat().st_size != entry["bytes"]:
                raise ValueError("matrix manifest byte count mismatch")
            _sha(entry["sha256"], "matrix file")
            state["matrices"][key] = (matrix_path, entry["sha256"])
    for state in layers.values():
        experts, _, _ = state["identity"]
        if set(state["matrices"]) != {(expert, projection) for expert in range(experts) for projection in PROJECTIONS}:
            raise ValueError("missing expert projections or incomplete layer")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources.sort(key=lambda item: item["path"])
    plan = {"schema": "glm53-p4-export-inputs.v1", "source_design_sha256": expected_design_sha256,
            "sources": sources, "layers": sorted(layers), "codec_schema": RANK_SCHEMA}
    plan_raw = (json.dumps(plan, indent=2, sort_keys=True) + "\n").encode()
    plan_sha = hashlib.sha256(plan_raw).hexdigest()
    plan_path = output_dir / "export-inputs.json"
    if plan_path.exists():
        if not resume or plan_path.read_bytes() != plan_raw:
            raise FileExistsError("resume requires the identical P4 export input plan")
    else:
        targets = [output_dir / "manifest.json"] + [output_dir / f"p4-layer-{layer:03d}-tp4-rank-{rank}.safetensors" for layer in layers for rank in range(4)]
        if any(path.exists() for path in targets):
            raise FileExistsError("refusing unbound existing P4 outputs")
        with plan_path.open("xb") as stream:
            stream.write(plan_raw)
    result = {"schema": MANIFEST_SCHEMA, "source_design_sha256": expected_design_sha256,
              "codec_schema": RANK_SCHEMA, "world_size": 4, "law": LAW,
              "layers": sorted(layers), "entries": [], "source_manifests": sources,
              "evidence_level": "structural", "ldlq": False}
    for layer in sorted(layers):
        state = layers[layer]
        experts, hidden, intermediate = state["identity"]
        layer_start = time.perf_counter()
        pending = {}
        entries = {}
        for rank in range(4):
            metadata = rank_metadata(layer=layer, rank=rank, experts=experts, hidden=hidden,
                                     intermediate=intermediate, source_design_sha256=expected_design_sha256)
            filename = f"p4-layer-{layer:03d}-tp4-rank-{rank}.safetensors"
            target, receipt_path = output_dir / filename, output_dir / (filename + ".receipt.json")
            if target.exists() or receipt_path.exists():
                if not resume or not target.exists() or not receipt_path.exists():
                    raise FileExistsError("existing rank requires a complete plan-bound receipt to resume")
                receipt = json.loads(receipt_path.read_text(), object_pairs_hook=_unique_object)
                entry = receipt["entry"]
                if receipt.get("export_inputs_sha256") != plan_sha or entry.get("layer") != layer or entry.get("rank") != rank or entry.get("path") != filename:
                    raise ValueError("rank resume receipt identity mismatch")
                raw = target.read_bytes()
                restored, restored_meta = deserialize_rank(raw, expected_sha256=entry["sha256"],
                    expected_layer=layer, expected_rank=rank, expected_design_sha256=expected_design_sha256,
                    expected_experts=experts)
                if any(restored_meta.get(key) != value for key, value in metadata.items()):
                    raise ValueError("rank resume geometry/metadata differs from input plan")
                expected_entry = {"layer": layer, "rank": rank, "path": filename,
                                  "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                                  **rank_accounting(restored, restored_meta, serialized=raw)}
                if entry != expected_entry:
                    raise ValueError("rank resume accounting mismatch")
                entries[rank] = entry
                timing["reused_ranks"] += 1
            else:
                tensors = {name: np.empty(shape, dtype=dtype) for name, (_, dtype, shape) in tensor_specs(experts, hidden, intermediate).items()}
                pending[rank] = tensors, metadata
        for expert in range(experts) if pending else ():
            projections = {}
            for projection in PROJECTIONS:
                path, sha = state["matrices"][expert, projection]
                shape = (hidden, intermediate) if projection == "down_proj" else (intermediate, hidden)
                projections[projection] = read_p4(path, expected_sha256=sha, expected_shape=shape)
                timing["source_matrix_reads"] += 1
                timing["source_matrix_bytes"] += path.stat().st_size
            for rank, (tensors, _) in pending.items():
                install_expert(tensors, projections, expert=expert, rank=rank, hidden=hidden, intermediate=intermediate)
        timing["stages"].append({"layer": layer, "stage": "read-validate-shard", "seconds": time.perf_counter() - layer_start})
        for rank, (tensors, metadata) in pending.items():
            write_start = time.perf_counter()
            raw = serialize_rank(tensors, metadata)
            filename = f"p4-layer-{layer:03d}-tp4-rank-{rank}.safetensors"
            target = output_dir / filename
            partial = output_dir / (filename + f".partial-{os.getpid()}-{time.time_ns()}")
            with partial.open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(partial, target)  # atomic no-overwrite publication
            partial.unlink()
            entry = {"layer": layer, "rank": rank, "path": filename,
                     "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                     **rank_accounting(tensors, metadata, serialized=raw)}
            with (output_dir / (filename + ".receipt.json")).open("x") as stream:
                stream.write(json.dumps({"export_inputs_sha256": plan_sha, "entry": entry}, indent=2, sort_keys=True) + "\n")
            entries[rank] = entry
            timing["written_ranks"] += 1
            timing["stages"].append({"layer": layer, "rank": rank, "stage": "serialize-hash-write", "seconds": time.perf_counter() - write_start})
        result["entries"].extend(entries[rank] for rank in range(4))
    weights = sum(entry["logical_weights"] for entry in result["entries"])
    payload_bytes = sum(entry["payload_bytes"] for entry in result["entries"])
    ratio = Fraction(payload_bytes * 8, weights)
    result["total"] = {"logical_weights": weights, "payload_bytes": payload_bytes,
                       "global_scale_bytes_including_tp_replication": sum(entry["global_scale_bytes"] for entry in result["entries"]),
                       "rank_file_bytes": sum(entry["bytes"] for entry in result["entries"]),
                       "payload_bpw": {"numerator": ratio.numerator, "denominator": ratio.denominator, "decimal": float(ratio)}}
    manifest_raw = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        if not resume or manifest_path.read_text() != manifest_raw:
            raise ValueError("completed manifest differs from verified rank receipts")
    else:
        with manifest_path.open("x") as stream:
            stream.write(manifest_raw)
    timing["elapsed_seconds"] = time.perf_counter() - started
    timing["export_inputs_sha256"] = plan_sha
    with (output_dir / f"export-timings-{time.time_ns()}.json").open("x") as stream:
        stream.write(json.dumps(timing, indent=2, sort_keys=True) + "\n")
    return result
