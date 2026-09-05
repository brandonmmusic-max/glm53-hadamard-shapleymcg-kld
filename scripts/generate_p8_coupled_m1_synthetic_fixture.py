#!/usr/bin/env python3
"""Prepare or explicitly generate a synthetic full-shaped coupled-P8 sidecar.

The default command is plan-only. ``--generate`` is required to write the
~0.963 GB rank-0 fixture. No model, calibration, Hessian, or quality artifact
is read. Tensor bytes are generated directly into one safetensors file in
bounded chunks; no dense or duplicate payload plane is created.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import time
from typing import Iterator, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRANSFORM = ROOT / "experiments/p8-coupled-transform-draw0-silu10-v1.json"
TRANSFORM_SHA256 = "093d219b18ba32471adcee746442b1481c7ba5659bbea62b94f1a665d4343a12"
SEED = 20260905129
CHUNK_BYTES = 1 << 20
MAX_AGGREGATE_BYTES = 30_000_000_000
RECEIPT_ALLOWANCE_BYTES = 1 << 20
TENSOR_ORDER = (
    "coupled_sign_draw_u8",
    "down_svh_fp16",
    "gate_up_suh_fp16",
    "intermediate_scales_fp16",
    "w13_scale_ue8m0",
    "w13_trellis",
    "w2_scale_ue8m0",
    "w2_trellis",
)
WEIGHT_NAMES = {
    "w13_trellis", "w2_trellis", "w13_scale_ue8m0", "w2_scale_ue8m0"
}
STREAM_TAGS = {
    "w13_trellis": 0x13579BDF,
    "w2_trellis": 0x2468ACE1,
    "w13_scale_ue8m0": 1,
    "w2_scale_ue8m0": 2,
    "gate_up_suh_fp16": 3,
    "intermediate_scales_fp16": 4,
    "down_svh_fp16": 5,
    "coupled_sign_draw_u8": 0,
}
SIGNED_PATTERN = np.asarray(
    [0.625, -0.75, 0.875, -1.125, 1.25, -1.375, 0.5, -1.5],
    dtype="<f2",
)
UE8M0_PATTERN = np.asarray([126, 127, 128, 129], dtype=np.uint8)


@dataclass(frozen=True)
class Geometry:
    experts: int
    hidden: int
    intermediate: int
    rank: int = 0
    layer: int = 3

    def validate(self, *, full: bool = False) -> None:
        if self.rank not in range(4) or self.layer != 3:
            raise ValueError("synthetic fixture is frozen to layer 3 and TP rank 0")
        if self.hidden % 512 or self.intermediate % 128:
            raise ValueError("coupled fixture requires H512 and I128 alignment")
        if self.experts <= 0:
            raise ValueError("experts must be positive")
        if full and self != FULL_GEOMETRY:
            raise ValueError("--generate only permits the frozen full-shaped geometry")


FULL_GEOMETRY = Geometry(experts=288, hidden=4096, intermediate=512)
SMALL_GEOMETRY = Geometry(experts=2, hidden=512, intermediate=128)


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]
    element_bytes: int
    recipe: str

    @property
    def elements(self) -> int:
        result = 1
        for value in self.shape:
            result *= value
        return result

    @property
    def nbytes(self) -> int:
        return self.elements * self.element_bytes


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
        + "\n"
    ).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_specs(geometry: Geometry) -> dict[str, TensorSpec]:
    geometry.validate()
    e, h, i = geometry.experts, geometry.hidden, geometry.intermediate
    specs = {
        "w13_trellis": TensorSpec(
            "w13_trellis", "I16", (2, e, h // 16, i // 16, 64), 2,
            "splitmix64-counter-low16-v1",
        ),
        "w2_trellis": TensorSpec(
            "w2_trellis", "I16", (e, i // 16, h // 16, 64), 2,
            "splitmix64-counter-low16-v1",
        ),
        "w13_scale_ue8m0": TensorSpec(
            "w13_scale_ue8m0", "U8", (e, 2 * i, h // 32), 1,
            "ue8m0-cycle-126-127-128-129-v1",
        ),
        "w2_scale_ue8m0": TensorSpec(
            "w2_scale_ue8m0", "U8", (e, h, i // 32), 1,
            "ue8m0-cycle-126-127-128-129-v1",
        ),
        "gate_up_suh_fp16": TensorSpec(
            "gate_up_suh_fp16", "F16", (h,), 2,
            "signed-eight-value-fp16-cycle-v1",
        ),
        "intermediate_scales_fp16": TensorSpec(
            "intermediate_scales_fp16", "F16", (e, 3 * i), 2,
            "signed-eight-value-fp16-cycle-v1",
        ),
        "down_svh_fp16": TensorSpec(
            "down_svh_fp16", "F16", (h,), 2,
            "signed-eight-value-fp16-cycle-v1",
        ),
        "coupled_sign_draw_u8": TensorSpec(
            "coupled_sign_draw_u8", "U8", (e,), 1, "constant-draw0-v1",
        ),
    }
    return {name: specs[name] for name in TENSOR_ORDER}


def _splitmix_u16(start: int, count: int, tag: int) -> bytes:
    values = np.arange(start, start + count, dtype=np.uint64)
    values += np.uint64((SEED + tag * 0x9E3779B9) & ((1 << 64) - 1))
    values += np.uint64(0x9E3779B97F4A7C15)
    values ^= values >> np.uint64(30)
    values *= np.uint64(0xBF58476D1CE4E5B9)
    values ^= values >> np.uint64(27)
    values *= np.uint64(0x94D049BB133111EB)
    values ^= values >> np.uint64(31)
    return (values & np.uint64(0xFFFF)).astype("<u2").tobytes()


def _pattern_bytes(pattern: np.ndarray, start: int, count: int, tag: int) -> bytes:
    indices = (np.arange(start, start + count, dtype=np.uint64) + tag) % len(pattern)
    return pattern[indices].tobytes()


def iter_tensor_bytes(spec: TensorSpec, *, chunk_bytes: int = CHUNK_BYTES) -> Iterator[bytes]:
    if chunk_bytes <= 0:
        raise ValueError("chunk size must be positive")
    chunk_elements = max(1, chunk_bytes // spec.element_bytes)
    tag = STREAM_TAGS[spec.name]
    for start in range(0, spec.elements, chunk_elements):
        count = min(chunk_elements, spec.elements - start)
        if spec.dtype == "I16":
            yield _splitmix_u16(start, count, tag)
        elif spec.name.endswith("scale_ue8m0"):
            yield _pattern_bytes(UE8M0_PATTERN, start, count, tag)
        elif spec.dtype == "F16":
            yield _pattern_bytes(SIGNED_PATTERN, start, count, tag)
        elif spec.name == "coupled_sign_draw_u8":
            yield bytes(count)
        else:
            raise AssertionError(f"unhandled synthetic recipe for {spec.name}")


def design_record(geometry: Geometry) -> dict[str, object]:
    return {
        "schema": "glm53.p8-coupled.synthetic-m1-fixture-design.v1",
        "evidence_role": "developmental-device-closure-only",
        "synthetic": True,
        "quality_claim_allowed": False,
        "model_reads": [],
        "calibration_reads": [],
        "seed": SEED,
        "layer": geometry.layer,
        "rank": geometry.rank,
        "world_size": 4,
        "geometry": {
            "experts": geometry.experts,
            "hidden": geometry.hidden,
            "local_intermediate": geometry.intermediate,
        },
        "codec": {
            "bits": 4,
            "law": "procedural-mcg-alpha2",
            "alphabet": "e4m3",
            "scale": "ue8m0-k32",
            "trellis_stream": "splitmix64-counter-low16-v1",
        },
        "scale_fixture": {
            "suh_svh": "signed-eight-value-fp16-cycle-v1",
            "ue8m0": "cycle-126-127-128-129-v1",
        },
        "transform_sha256": TRANSFORM_SHA256,
        "transform_id": "normalized-h512-outer-h128-inner-sign-draw0-v1",
        "ldlq": False,
    }


def _scale_recipe_sha256() -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "kind": "synthetic-no-exl3-read",
                "pattern": SIGNED_PATTERN.astype("<f2").view(np.uint8).tolist(),
                "seed": SEED,
            }
        )
    ).hexdigest()


def base_metadata(
    geometry: Geometry,
    *,
    design_sha256: str,
    tensor_hashes: dict[str, str],
) -> dict[str, str]:
    specs = tensor_specs(geometry)
    weight_bytes = sum(spec.nbytes for name, spec in specs.items() if name in WEIGHT_NAMES)
    metadata_bytes = sum(spec.nbytes for name, spec in specs.items() if name not in WEIGHT_NAMES)
    logical = 3 * geometry.experts * geometry.hidden * geometry.intermediate
    runtime_sign_bytes = 3 * geometry.intermediate * 2
    signs_hash = hashlib.sha256(
        np.ones(3 * geometry.intermediate, dtype="<f2").tobytes()
    ).hexdigest()
    metadata = {
        "schema": "glm53-p8-coupled-h512-h128-tp4-rank.v1",
        "layer": str(geometry.layer),
        "rank": str(geometry.rank),
        "world_size": "4",
        "bits": "4",
        "alphabet": "e4m3",
        "scale": "ue8m0-k32",
        "law": "procedural-mcg-alpha2",
        "boundary": "coupled-h512-h128-suh-svh-v1",
        "component": "p8-coupled-h512-h128-sign-v1",
        "composition_target": "coupled-h512-h128-suh-svh-v1",
        "full_coupled": "true",
        "activation": "silu-cap10",
        "cast_order": "luke-qsrt-coupled-reference-v1",
        "quantized_input_order": (
            "bf16-cvt-rn-fp16-h512-fp32-suh-fp32-h128-fp32-e4m3-ue8m0-k32"
        ),
        "quantized_down_order": (
            "silu-cap10-fp32-sign-h128-fp32-down-suh-fp32-h128-fp32-e4m3-ue8m0-k32"
        ),
        "h512": "normalized-sylvester-512-v1",
        "h128": "normalized-sylvester-128-v1",
        "transform_id": "normalized-h512-outer-h128-inner-sign-draw0-v1",
        "encoder_transform_sha256": TRANSFORM_SHA256,
        "sign_generator": "qsrt-coupled-signs-v1",
        "sign_draw": "0",
        "sign_pre_axis": "1",
        "sign_post_axis": "2",
        "fc1_interleave": "slot0-atom32-slot1-atom32-v1",
        "tp_slice": "contiguous-atom32-v1",
        "global_intermediate": str(geometry.intermediate * 4),
        "local_atom_begin": str(geometry.rank * (geometry.intermediate // 32)),
        "gate_up_suh_shared": "true",
        "down_svh_shared": "true",
        "coupled_signs_shared": "true",
        "signed_scales": "true",
        "ldlq": "false",
        "fc1_trellis_slot_order": "gate-up",
        "fc1_scale_plane_order": "up-gate",
        "coupled_scale_order": "gate_svh-up_svh-down_suh",
        "source_design_sha256": design_sha256,
        # Legacy field name retained by the runtime format. Its value hashes
        # the synthetic recipe, and the adjacent kind forbids EXL3 provenance.
        "exl3_scale_source_sha256": _scale_recipe_sha256(),
        "exl3_scale_source_kind": "synthetic-no-exl3-read",
        "synthetic_fixture": "true",
        "quality_claim_allowed": "false",
        "seed": str(SEED),
        "weight_payload_bpw": format(8 * weight_bytes / logical, ".17g"),
        "metadata_bpw": format(8 * metadata_bytes / logical, ".17g"),
        "stored_metadata_bytes": str(metadata_bytes),
        "runtime_regenerated_sign_bytes": str(runtime_sign_bytes),
        "full_coupled_accounted_metadata_bpw": format(
            8 * (metadata_bytes + runtime_sign_bytes) / logical, ".17g"
        ),
        "full_coupled_accounted_bpw": format(
            8 * (weight_bytes + metadata_bytes + runtime_sign_bytes) / logical,
            ".17g",
        ),
        "sha256_coupled_signs_fp16": signs_hash,
    }
    metadata.update({f"sha256_{name}": tensor_hashes[name] for name in TENSOR_ORDER})
    return metadata


def _header(
    specs: dict[str, TensorSpec], metadata: dict[str, str]
) -> tuple[bytes, dict[str, tuple[int, int]]]:
    header: dict[str, object] = {"__metadata__": metadata}
    offset = 0
    offsets = {}
    for name, spec in specs.items():
        offsets[name] = (offset, offset + spec.nbytes)
        header[name] = {
            "dtype": spec.dtype,
            "shape": list(spec.shape),
            "data_offsets": list(offsets[name]),
        }
        offset += spec.nbytes
    encoded = json.dumps(
        header, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    padded = encoded + b" " * ((-len(encoded)) % 8)
    return padded, offsets


def forecast(geometry: Geometry = FULL_GEOMETRY) -> dict[str, object]:
    specs = tensor_specs(geometry)
    design = canonical_json_bytes(design_record(geometry))
    design_sha = hashlib.sha256(design).hexdigest()
    placeholder = {name: "0" * 64 for name in specs}
    header, _ = _header(specs, base_metadata(
        geometry, design_sha256=design_sha, tensor_hashes=placeholder
    ))
    payload = sum(spec.nbytes for spec in specs.values())
    return {
        "schema": "glm53.p8-coupled.synthetic-m1-fixture-plan.v1",
        "synthetic": True,
        "quality_claim_allowed": False,
        "seed": SEED,
        "geometry": geometry.__dict__,
        "transform": str(TRANSFORM),
        "transform_sha256": TRANSFORM_SHA256,
        "payload_bytes": payload,
        "header_bytes_including_u64": 8 + len(header),
        "sidecar_bytes": 8 + len(header) + payload,
        "receipt_allowance_bytes": RECEIPT_ALLOWANCE_BYTES,
        "max_aggregate_bytes": MAX_AGGREGATE_BYTES,
        "max_generated_chunk_bytes": CHUNK_BYTES,
        "tensor_bytes": {name: spec.nbytes for name, spec in specs.items()},
    }


def _tree_bytes(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            candidate = Path(root) / name
            if not candidate.is_symlink():
                total += candidate.stat().st_size
    return total


def _external_budget(path: Path, expected_sha256: str, budget_root: Path) -> dict[str, object]:
    """Bind the external charge to an audited receipt; a hash alone is not an audit."""
    if not path.is_absolute() or not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError("external budget receipt identity differs")
    record = json.loads(path.read_text())
    if record.get("schema") != "glm53.p8-coupled.external-budget.v1":
        raise ValueError("external budget receipt schema differs")
    if record.get("budget_root") != str(budget_root.resolve()):
        raise ValueError("external budget receipt is for another budget root")
    if record.get("max_aggregate_bytes") != MAX_AGGREGATE_BYTES:
        raise ValueError("external budget ceiling differs")
    amount = record.get("external_bytes_upper_bound")
    if type(amount) is not int or amount < 0:
        raise ValueError("external budget charge must be a nonnegative integer")
    measured, expires = record.get("measured_unix"), record.get("valid_until_unix")
    if (type(measured) not in (int, float) or type(expires) not in (int, float)
            or not measured <= time.time() <= expires):
        raise ValueError("external budget receipt is stale or future-dated")
    evidence = record.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("external budget receipt requires audited evidence")
    for item in evidence:
        source = Path(item["path"])
        if not source.is_absolute() or sha256_file(source) != item["sha256"]:
            raise ValueError("external budget evidence identity differs")
    return {"external_bytes_upper_bound": amount,
            "external_budget_receipt": str(path),
            "external_budget_receipt_sha256": expected_sha256}


def _validate_budget(output_dir: Path, budget_root: Path, plan: dict[str, object],
                     *, external_bytes: int = 0) -> dict[str, int]:
    if type(external_bytes) is not int or external_bytes < 0:
        raise ValueError("external charge must be a nonnegative integer")
    if not budget_root.is_absolute() or not budget_root.is_dir():
        raise ValueError("--budget-root must be an existing absolute directory")
    if not output_dir.is_absolute() or output_dir.exists():
        raise ValueError("--output-dir must be a fresh absolute directory")
    if not output_dir.parent.is_dir():
        raise ValueError("--output-dir parent must already exist")
    resolved_parent = output_dir.parent.resolve()
    root = budget_root.resolve()
    if root != resolved_parent and root not in resolved_parent.parents:
        raise ValueError("output directory must be inside --budget-root")
    existing = _tree_bytes(root)
    forecast_new = (
        int(plan["sidecar_bytes"])
        + len(canonical_json_bytes(design_record(FULL_GEOMETRY)))
        + RECEIPT_ALLOWANCE_BYTES
    )
    if external_bytes + existing + forecast_new > MAX_AGGREGATE_BYTES:
        raise RuntimeError("synthetic fixture would exceed the 30 GB aggregate budget")
    free = shutil.disk_usage(resolved_parent).free
    if free < forecast_new:
        raise RuntimeError("filesystem free bytes are below projected fixture writes")
    return {
        "existing_budget_root_bytes": existing,
        "forecast_new_bytes_with_receipt_allowance": forecast_new,
        "external_bytes_upper_bound": external_bytes,
        "projected_aggregate_bytes": external_bytes + existing + forecast_new,
        "filesystem_free_bytes_before": free,
    }


def _write_sidecar(
    output: Path,
    geometry: Geometry,
    *,
    design_sha256: str,
) -> tuple[dict[str, str], int]:
    specs = tensor_specs(geometry)
    placeholders = {name: "0" * 64 for name in specs}
    provisional, offsets = _header(
        specs,
        base_metadata(
            geometry, design_sha256=design_sha256, tensor_hashes=placeholders
        ),
    )
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output} or {partial}")
    hashes: dict[str, str] = {}
    try:
        with partial.open("xb") as stream:
            stream.write(struct.pack("<Q", len(provisional)))
            stream.write(provisional)
            data_begin = stream.tell()
            for name, spec in specs.items():
                if stream.tell() != data_begin + offsets[name][0]:
                    raise RuntimeError(f"stream offset drift before {name}")
                digest = hashlib.sha256()
                written = 0
                for raw in iter_tensor_bytes(spec):
                    stream.write(raw)
                    digest.update(raw)
                    written += len(raw)
                if written != spec.nbytes:
                    raise RuntimeError(f"stream byte count differs for {name}")
                hashes[name] = digest.hexdigest()
            final_metadata = base_metadata(
                geometry, design_sha256=design_sha256, tensor_hashes=hashes
            )
            final_header, final_offsets = _header(specs, final_metadata)
            if len(final_header) != len(provisional) or final_offsets != offsets:
                raise RuntimeError("final hashes changed safetensors header geometry")
            stream.seek(8)
            stream.write(final_header)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, output)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise
    return hashes, 8 + len(provisional) + sum(spec.nbytes for spec in specs.values())


def _validate_written(
    path: Path,
    geometry: Geometry,
    *,
    design_sha256: str,
    tensor_hashes: dict[str, str],
) -> dict[str, object]:
    import torch
    from safetensors import safe_open
    from glm53_nvfp4.trellis_mxf import (
        decode_trellis_mxf,
        state_lut,
        unpack_ue8m0,
    )
    from runtime_patch.p8_coupled_scales import (
        SCALE_NAMES,
        validate_coupled_component,
    )

    specs = tensor_specs(geometry)
    with path.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_size))
        data_begin = 8 + header_size
        for name, spec in specs.items():
            begin, end = header[name]["data_offsets"]
            digest = hashlib.sha256()
            stream.seek(data_begin + begin)
            remaining = end - begin
            while remaining:
                raw = stream.read(min(CHUNK_BYTES, remaining))
                if not raw:
                    raise RuntimeError(f"truncated tensor bytes for {name}")
                digest.update(raw)
                remaining -= len(raw)
            if digest.hexdigest() != tensor_hashes[name]:
                raise RuntimeError(f"post-write tensor hash mismatch for {name}")

    with safe_open(path, framework="pt", device="cpu") as source:
        metadata = source.metadata() or {}
        if set(source.keys()) != set(specs):
            raise RuntimeError("written safetensors key set differs")
        for name, spec in specs.items():
            view = source.get_slice(name)
            if tuple(view.get_shape()) != spec.shape:
                raise RuntimeError(f"written shape differs for {name}")
        scale_tensors = {name: source.get_tensor(name) for name in SCALE_NAMES}
        component = validate_coupled_component(
            metadata,
            scale_tensors,
            layer=geometry.layer,
            rank=geometry.rank,
            experts=geometry.experts,
            hidden=geometry.hidden,
            intermediate=geometry.intermediate,
            expected_transform_sha256=TRANSFORM_SHA256,
        )
        flat_roles = {
            "gate_up_suh": component.gate_up_suh.float().reshape(-1),
            "intermediate_scales": component.intermediate_scales.float().reshape(-1),
            "down_svh": component.down_svh.float().reshape(-1),
        }
        if any(
            not bool((value > 0).any() and (value < 0).any() and (value != 1).any())
            for value in flat_roles.values()
        ):
            raise RuntimeError("synthetic suh/svh roles are not signed and non-unit")
        scale_samples = {
            "w13": source.get_slice("w13_scale_ue8m0")[0, :2, :8],
            "w2": source.get_slice("w2_scale_ue8m0")[0, :2, :4],
        }
        if any(
            value.dtype != torch.uint8
            or bool((value == 255).any())
            or not bool((value != 127).any())
            for value in scale_samples.values()
        ):
            raise RuntimeError("synthetic UE8M0 sample is invalid or identity-only")
        if metadata.get("source_design_sha256") != design_sha256:
            raise RuntimeError("written sidecar design identity differs")
        lut = state_lut(
            4, alphabet="e4m3", law="mcg", compander_scale=2.0, device="cpu"
        )
        w13 = source.get_slice("w13_trellis")[:, 0]
        w2 = source.get_slice("w2_trellis")[0]
        w13_sf = source.get_slice("w13_scale_ue8m0")[0]
        w2_sf = source.get_slice("w2_scale_ue8m0")[0]
        decoded = {
            "gate": (
                decode_trellis_mxf(
                    w13[0], lut, w13_sf[geometry.intermediate :],
                    bits=4, block_size=32, rows=geometry.intermediate,
                    width=geometry.hidden, device="cpu",
                ),
                w13_sf[geometry.intermediate :],
            ),
            "up": (
                decode_trellis_mxf(
                    w13[1], lut, w13_sf[: geometry.intermediate],
                    bits=4, block_size=32, rows=geometry.intermediate,
                    width=geometry.hidden, device="cpu",
                ),
                w13_sf[: geometry.intermediate],
            ),
            "down": (
                decode_trellis_mxf(
                    w2, lut, w2_sf,
                    bits=4, block_size=32, rows=geometry.hidden,
                    width=geometry.intermediate, device="cpu",
                ),
                w2_sf,
            ),
        }
        decode_receipt = {}
        for name, (value, codes) in decoded.items():
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"synthetic {name} MCG decode is nonfinite")
            blocks = value.reshape(value.shape[0], -1, 32)
            scale = unpack_ue8m0(codes).unsqueeze(-1)
            payload = (blocks / scale).to(torch.float8_e4m3fn)
            if not torch.equal(payload.float() * scale, blocks):
                raise RuntimeError(f"synthetic {name} is not exact E4M3 times UE8M0")
            decode_receipt[name] = {
                "finite": True,
                "decoded_sha256": hashlib.sha256(
                    value.contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
                "normalized_e4m3_sha256": hashlib.sha256(
                    payload.contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
            }
    return {
        "runtime_cpu_schema_validation": "pass",
        "post_write_tensor_hash_validation": "pass",
        "expert0_k4_mcg_decode": decode_receipt,
        "signed_nonunit_scale_roles": True,
        "sampled_ue8m0_codes": {
            name: value.reshape(-1).tolist() for name, value in scale_samples.items()
        },
    }


def generate(output_dir: Path, budget_root: Path, *, external_budget_receipt: Path,
             external_budget_sha256: str) -> dict[str, object]:
    geometry = FULL_GEOMETRY
    geometry.validate(full=True)
    if sha256_file(TRANSFORM) != TRANSFORM_SHA256:
        raise RuntimeError("repository draw0 transform receipt differs")
    plan = forecast(geometry)
    external = _external_budget(external_budget_receipt, external_budget_sha256, budget_root)
    budget = _validate_budget(output_dir, budget_root, plan,
                              external_bytes=external["external_bytes_upper_bound"])
    budget.update(external)
    output_dir.mkdir(mode=0o700)
    design_path = output_dir / "synthetic-design.json"
    sidecar = output_dir / "p8-synthetic-layer-003-tp4-rank-0.safetensors"
    receipt_path = output_dir / "receipt.json"
    design_bytes = canonical_json_bytes(design_record(geometry))
    design_path.write_bytes(design_bytes)
    design_sha = hashlib.sha256(design_bytes).hexdigest()
    tensor_hashes, expected_size = _write_sidecar(
        sidecar, geometry, design_sha256=design_sha
    )
    if sidecar.stat().st_size != expected_size or expected_size != plan["sidecar_bytes"]:
        raise RuntimeError("generated sidecar size differs from frozen forecast")
    validation = _validate_written(
        sidecar,
        geometry,
        design_sha256=design_sha,
        tensor_hashes=tensor_hashes,
    )
    receipt = {
        "schema": "glm53.p8-coupled.synthetic-m1-fixture-receipt.v1",
        "synthetic": True,
        "quality_claim_allowed": False,
        "generated_from_model": False,
        "plan": plan,
        "budget": budget,
        "identities": {
            "sidecar": {
                "path": str(sidecar.resolve()),
                "bytes": sidecar.stat().st_size,
                "sha256": sha256_file(sidecar),
            },
            "design": {
                "path": str(design_path.resolve()),
                "bytes": design_path.stat().st_size,
                "sha256": design_sha,
            },
            "transform": {
                "path": str(TRANSFORM.resolve()),
                "bytes": TRANSFORM.stat().st_size,
                "sha256": TRANSFORM_SHA256,
            },
        },
        "tensor_sha256": tensor_hashes,
        "validation": validation,
        "closure_command": [
            "python3", str((ROOT / "scripts/run_p8_coupled_m1_device_closure.py").resolve()),
            "--execute", "--image", "sha256:<PIN_REPAIRED_IMAGE_ID>",
            "--gpu-device", "<IDLE_SM120_GPU>",
            "--sidecar", str(sidecar.resolve()),
            "--sidecar-sha256", "<FROM_THIS_RECEIPT>",
            "--design", str(design_path.resolve()),
            "--design-sha256", design_sha,
            "--transform", str(TRANSFORM.resolve()),
            "--transform-sha256", TRANSFORM_SHA256,
            "--runtime-manifest-sha256", "<PIN_REPAIRED_RUNTIME_MANIFEST>",
            "--output", str((output_dir / "device-closure-evidence").resolve()),
        ],
    }
    receipt_path.write_bytes(canonical_json_bytes(receipt))
    print(json.dumps(receipt, sort_keys=True))
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--budget-root", type=Path)
    parser.add_argument("--external-budget-receipt", type=Path)
    parser.add_argument("--external-budget-sha256")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.generate:
        if any(value is not None for value in (args.output_dir, args.budget_root,
                                               args.external_budget_receipt, args.external_budget_sha256)):
            raise ValueError("output arguments require explicit --generate")
        print(json.dumps(forecast(), sort_keys=True))
        return
    if args.output_dir is None or args.budget_root is None:
        raise ValueError("--generate requires --output-dir and --budget-root")
    if args.external_budget_receipt is None or args.external_budget_sha256 is None:
        raise ValueError("--generate requires a pinned campaign-wide external budget receipt")
    generate(args.output_dir, args.budget_root,
             external_budget_receipt=args.external_budget_receipt,
             external_budget_sha256=args.external_budget_sha256)


if __name__ == "__main__":
    main()
