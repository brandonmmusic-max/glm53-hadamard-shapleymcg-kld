"""CPU-only validation for full-coupled TrellisMX P8 TP4 sidecars."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import safe_open, save_file

from glm53_nvfp4.p8_coupled_scale import (
    COUPLED_ACTIVATION,
    COUPLED_BOUNDARY,
    COUPLED_CAST_ORDER,
    COUPLED_COMPONENT,
    COUPLED_FC1_INTERLEAVE,
    COUPLED_QUANTIZED_DOWN_ORDER,
    COUPLED_QUANTIZED_INPUT_ORDER,
    COUPLED_RANK_SCHEMA,
    COUPLED_SIGN_DRAW,
    COUPLED_SIGN_GENERATOR,
    COUPLED_TP_SLICE,
    COUPLED_TRANSFORM_ID,
    COUPLED_TRANSFORM_SHA256,
    rank_local_coupled_signs,
)
from .protocol import P8_RATES


PRODUCTION_GEOMETRY = {
    "experts": 288,
    "hidden": 4096,
    "local_intermediate": 512,
}
WEIGHT_TENSORS = (
    "w13_trellis",
    "w2_trellis",
    "w13_scale_ue8m0",
    "w2_scale_ue8m0",
)
METADATA_TENSORS = (
    "gate_up_suh_fp16",
    "intermediate_scales_fp16",
    "down_svh_fp16",
    "coupled_sign_draw_u8",
)
SIDECAR_TENSORS = WEIGHT_TENSORS + METADATA_TENSORS


def tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _hex64(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{field} must be a lowercase 64-character SHA-256")
    return value


def _expected_shapes(
    *, bits: int, experts: int, hidden: int, local_intermediate: int
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    stream_words = 16 * bits
    return {
        "w13_trellis": (
            (2, experts, hidden // 16, local_intermediate // 16, stream_words),
            torch.int16,
        ),
        "w2_trellis": (
            (experts, local_intermediate // 16, hidden // 16, stream_words),
            torch.int16,
        ),
        "w13_scale_ue8m0": (
            (experts, 2 * local_intermediate, hidden // 32),
            torch.uint8,
        ),
        "w2_scale_ue8m0": (
            (experts, hidden, local_intermediate // 32),
            torch.uint8,
        ),
        "gate_up_suh_fp16": ((hidden,), torch.float16),
        "intermediate_scales_fp16": (
            (experts, 3 * local_intermediate),
            torch.float16,
        ),
        "down_svh_fp16": ((hidden,), torch.float16),
        "coupled_sign_draw_u8": ((experts,), torch.uint8),
    }


def _required_metadata(
    *,
    layer: int,
    rank: int,
    bits: int,
    design_sha256: str,
    scale_source_sha256: str,
    experts: int,
    hidden: int,
    local_intermediate: int,
) -> dict[str, str]:
    return {
        "schema": COUPLED_RANK_SCHEMA,
        "layer": str(layer),
        "rank": str(rank),
        "world_size": "4",
        "bits": str(bits),
        "alphabet": "e4m3",
        "scale": "ue8m0-k32",
        "law": "procedural-mcg-alpha2",
        "boundary": COUPLED_BOUNDARY,
        "component": COUPLED_COMPONENT,
        "composition_target": COUPLED_BOUNDARY,
        "full_coupled": "true",
        "activation": COUPLED_ACTIVATION,
        "cast_order": COUPLED_CAST_ORDER,
        "quantized_input_order": COUPLED_QUANTIZED_INPUT_ORDER,
        "quantized_down_order": COUPLED_QUANTIZED_DOWN_ORDER,
        "h512": "normalized-sylvester-512-v1",
        "h128": "normalized-sylvester-128-v1",
        "transform_id": COUPLED_TRANSFORM_ID,
        "encoder_transform_sha256": COUPLED_TRANSFORM_SHA256,
        "sign_generator": COUPLED_SIGN_GENERATOR,
        "sign_draw": str(COUPLED_SIGN_DRAW),
        "sign_pre_axis": "1",
        "sign_post_axis": "2",
        "fc1_interleave": COUPLED_FC1_INTERLEAVE,
        "tp_slice": COUPLED_TP_SLICE,
        "global_intermediate": str(local_intermediate * 4),
        "local_atom_begin": str(rank * (local_intermediate // 32)),
        "gate_up_suh_shared": "true",
        "down_svh_shared": "true",
        "coupled_signs_shared": "true",
        "signed_scales": "true",
        "ldlq": "false",
        "fc1_trellis_slot_order": "gate-up",
        "fc1_scale_plane_order": "up-gate",
        "coupled_scale_order": "gate_svh-up_svh-down_suh",
        "source_design_sha256": design_sha256,
        "exl3_scale_source_sha256": scale_source_sha256,
    }


def _rates(
    tensors: Mapping[str, torch.Tensor], *, experts: int, hidden: int, local_intermediate: int
) -> tuple[int, int, float, float]:
    logical = 3 * experts * hidden * local_intermediate
    weight_bytes = sum(
        tensors[name].numel() * tensors[name].element_size() for name in WEIGHT_TENSORS
    )
    metadata_bytes = sum(
        tensors[name].numel() * tensors[name].element_size()
        for name in METADATA_TENSORS
    )
    return (
        weight_bytes,
        metadata_bytes,
        8.0 * weight_bytes / logical,
        8.0 * metadata_bytes / logical,
    )


@dataclass(frozen=True)
class SidecarReport:
    path: str
    schema: str
    layer: int
    rank: int
    bits: int
    experts: int
    hidden: int
    local_intermediate: int
    source_design_sha256: str
    exl3_scale_source_sha256: str
    tensor_sha256: dict[str, str]
    file_sha256: str
    weight_payload_bpw: float
    metadata_bpw: float
    runtime_loader_closure: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "passed",
            "evidence_level": "structural-metadata-and-byte-hash",
            "path": self.path,
            "schema": self.schema,
            "layer": self.layer,
            "rank": self.rank,
            "bits": self.bits,
            "experts": self.experts,
            "hidden": self.hidden,
            "local_intermediate": self.local_intermediate,
            "source_design_sha256": self.source_design_sha256,
            "exl3_scale_source_sha256": self.exl3_scale_source_sha256,
            "tensor_sha256": dict(self.tensor_sha256),
            "file_sha256": self.file_sha256,
            "weight_payload_bpw": self.weight_payload_bpw,
            "metadata_bpw": self.metadata_bpw,
            "runtime_loader_closure": self.runtime_loader_closure,
            "gpu_executed": False,
        }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_sidecar(
    path: Path,
    *,
    expected_layer: int | None = None,
    expected_rank: int | None = None,
    expected_bits: int | None = None,
    expected_design_sha256: str | None = None,
    expected_transform_sha256: str | None = None,
    expected_scale_source_sha256: str | None = None,
    expected_file_sha256: str | None = None,
    production_geometry: bool = True,
    experts: int = 288,
    hidden: int = 4096,
    local_intermediate: int = 512,
) -> SidecarReport:
    """Validate one full-coupled sidecar without importing a CUDA runtime."""

    if production_geometry and (experts, hidden, local_intermediate) != (
        PRODUCTION_GEOMETRY["experts"],
        PRODUCTION_GEOMETRY["hidden"],
        PRODUCTION_GEOMETRY["local_intermediate"],
    ):
        raise ValueError("production sidecar validation requires GLM TP4 geometry")
    if min(experts, hidden, local_intermediate) <= 0:
        raise ValueError("sidecar geometry must be positive")
    if hidden % 32 or local_intermediate % 32 or local_intermediate % 16:
        raise ValueError("sidecar hidden/intermediate geometry is not aligned")

    with safe_open(path, framework="pt", device="cpu") as source:
        metadata = source.metadata() or {}
        if sorted(source.keys()) != sorted(SIDECAR_TENSORS):
            raise ValueError("sidecar tensor inventory is not exact")
        tensors = {name: source.get_tensor(name) for name in SIDECAR_TENSORS}

    bits_text = metadata.get("bits", "")
    if bits_text not in {str(rate) for rate in P8_RATES}:
        raise ValueError(f"unsupported sidecar rate {bits_text!r}")
    bits = int(bits_text)
    if expected_bits is not None and bits != expected_bits:
        raise ValueError("sidecar rate does not match the expected pin")
    try:
        layer = int(metadata.get("layer", ""))
        rank = int(metadata.get("rank", ""))
    except ValueError as error:
        raise ValueError("sidecar layer/rank must be integers") from error
    if not 3 <= layer <= 44:
        raise ValueError("GLM sidecar layer must be in 3..44")
    if rank not in range(4):
        raise ValueError("GLM TP4 sidecar rank must be in 0..3")
    if expected_layer is not None and layer != expected_layer:
        raise ValueError("sidecar layer does not match the expected pin")
    if expected_rank is not None and rank != expected_rank:
        raise ValueError("sidecar rank does not match the expected pin")

    design_sha256 = _hex64(
        metadata.get("source_design_sha256"), "source_design_sha256"
    )
    scale_source_sha256 = _hex64(
        metadata.get("exl3_scale_source_sha256"), "exl3_scale_source_sha256"
    )
    transform_sha256 = _hex64(
        metadata.get("encoder_transform_sha256"),
        "encoder_transform_sha256",
    )
    if expected_design_sha256 is not None and design_sha256 != expected_design_sha256:
        raise ValueError("sidecar design identity does not match the expected pin")
    if (
        expected_transform_sha256 is not None
        and transform_sha256 != expected_transform_sha256
    ):
        raise ValueError("sidecar transform identity does not match the expected pin")
    if (
        expected_scale_source_sha256 is not None
        and scale_source_sha256 != expected_scale_source_sha256
    ):
        raise ValueError("sidecar scale-source identity does not match the expected pin")

    required = _required_metadata(
        layer=layer,
        rank=rank,
        bits=bits,
        design_sha256=design_sha256,
        scale_source_sha256=scale_source_sha256,
        experts=experts,
        hidden=hidden,
        local_intermediate=local_intermediate,
    )
    mismatches = {
        key: {"expected": value, "observed": metadata.get(key)}
        for key, value in required.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"sidecar metadata mismatch: {mismatches}")

    expected_shapes = _expected_shapes(
        bits=bits,
        experts=experts,
        hidden=hidden,
        local_intermediate=local_intermediate,
    )
    observed_hashes: dict[str, str] = {}
    for name, (shape, dtype) in expected_shapes.items():
        tensor = tensors[name]
        if tensor.dtype != dtype or tuple(tensor.shape) != shape:
            raise ValueError(
                f"invalid {name}: expected {dtype} {shape}, got "
                f"{tensor.dtype} {tuple(tensor.shape)}"
            )
        observed_hash = tensor_sha256(tensor)
        if metadata.get(f"sha256_{name}") != observed_hash:
            raise ValueError(f"sidecar tensor hash mismatch for {name}")
        observed_hashes[name] = observed_hash

    if not bool(torch.isfinite(tensors["gate_up_suh_fp16"]).all()) or not bool(
        torch.isfinite(tensors["intermediate_scales_fp16"]).all()
    ) or not bool(torch.isfinite(tensors["down_svh_fp16"]).all()):
        raise ValueError("sidecar coupled scales contain non-finite values")
    if bool((tensors["gate_up_suh_fp16"] == 0).any()) or bool(
        (tensors["down_svh_fp16"] == 0).any()
    ):
        raise ValueError("sidecar shared coupled scales cannot contain zero")

    signs = rank_local_coupled_signs(intermediate=local_intermediate, rank=rank)
    if metadata.get("sha256_coupled_signs_fp16") != tensor_sha256(signs):
        raise ValueError("runtime-regenerated coupled signs do not match sidecar identity")

    weight_bytes, metadata_bytes, weight_bpw, metadata_bpw = _rates(
        tensors, experts=experts, hidden=hidden, local_intermediate=local_intermediate
    )
    if production_geometry:
        if not math.isclose(weight_bpw, bits + 0.25, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("production sidecar payload rate is not K+0.25")
        expected_metadata_bytes = (
            4 * hidden + 6 * experts * local_intermediate + experts
        )
        if metadata_bytes != expected_metadata_bytes:
            raise ValueError("production sidecar metadata byte count mismatch")
    if metadata.get("weight_payload_bpw") != format(weight_bpw, ".17g"):
        raise ValueError("sidecar weight payload rate does not match tensors")
    if metadata.get("metadata_bpw") != format(metadata_bpw, ".17g"):
        raise ValueError("sidecar metadata rate does not match tensors")

    file_hash = _file_sha256(path)
    if expected_file_sha256 is not None and file_hash != expected_file_sha256:
        raise ValueError("sidecar file hash does not match the expected pin")
    return SidecarReport(
        path=str(path),
        schema=COUPLED_RANK_SCHEMA,
        layer=layer,
        rank=rank,
        bits=bits,
        experts=experts,
        hidden=hidden,
        local_intermediate=local_intermediate,
        source_design_sha256=design_sha256,
        exl3_scale_source_sha256=scale_source_sha256,
        tensor_sha256=observed_hashes,
        file_sha256=file_hash,
        weight_payload_bpw=weight_bpw,
        metadata_bpw=metadata_bpw,
        runtime_loader_closure="not tested",
    )


def write_synthetic_sidecar(
    path: Path,
    *,
    layer: int = 3,
    rank: int = 0,
    bits: int = 4,
    experts: int = 1,
    hidden: int = 32,
    local_intermediate: int = 32,
    seed: int = 20260907,
)-> SidecarReport:
    """Write a tiny structural fixture; this is not a model weight artifact."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    if layer not in range(3, 45) or rank not in range(4) or bits not in P8_RATES:
        raise ValueError("invalid synthetic sidecar identity")
    generator = torch.Generator().manual_seed(seed)
    shapes = _expected_shapes(
        bits=bits,
        experts=experts,
        hidden=hidden,
        local_intermediate=local_intermediate,
    )
    tensors: dict[str, torch.Tensor] = {}
    for name, (shape, dtype) in shapes.items():
        if dtype == torch.int16:
            tensors[name] = torch.randint(
                -32768, 32768, shape, dtype=torch.int64, generator=generator
            ).to(torch.int16)
        elif dtype == torch.uint8:
            tensors[name] = torch.randint(
                0, 255, shape, dtype=torch.int64, generator=generator
            ).to(torch.uint8)
        else:
            tensors[name] = (
                torch.randn(shape, dtype=torch.float32, generator=generator)
                .clamp_(-4, 4)
                .to(dtype)
            )
    # Avoid accidental zeros in shared vectors; synthetic intermediate scales may
    # legally contain signed values but should also remain finite and nonzero.
    for name in ("gate_up_suh_fp16", "down_svh_fp16"):
        tensors[name] = torch.where(
            tensors[name] == 0, torch.ones_like(tensors[name]), tensors[name]
        )

    design = hashlib.sha256(f"synthetic-design-{layer}-{rank}-{bits}".encode()).hexdigest()
    scale_source = hashlib.sha256(
        f"synthetic-scale-source-{layer}-{rank}-{bits}".encode()
    ).hexdigest()
    weight_bytes, metadata_bytes, weight_bpw, metadata_bpw = _rates(
        tensors, experts=experts, hidden=hidden, local_intermediate=local_intermediate
    )
    signs = rank_local_coupled_signs(intermediate=local_intermediate, rank=rank)
    metadata = _required_metadata(
        layer=layer,
        rank=rank,
        bits=bits,
        design_sha256=design,
        scale_source_sha256=scale_source,
        experts=experts,
        hidden=hidden,
        local_intermediate=local_intermediate,
    )
    metadata.update(
        {
            "weight_payload_bpw": format(weight_bpw, ".17g"),
            "metadata_bpw": format(metadata_bpw, ".17g"),
            "stored_metadata_bytes": str(metadata_bytes),
            "runtime_regenerated_sign_bytes": str(signs.numel() * signs.element_size()),
            "sha256_coupled_signs_fp16": tensor_sha256(signs),
        }
    )
    metadata.update(
        {f"sha256_{name}": tensor_sha256(value) for name, value in tensors.items()}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, path, metadata=metadata)
    return validate_sidecar(
        path,
        expected_layer=layer,
        expected_rank=rank,
        expected_bits=bits,
        expected_design_sha256=design,
        expected_scale_source_sha256=scale_source,
        production_geometry=False,
        experts=experts,
        hidden=hidden,
        local_intermediate=local_intermediate,
    )
