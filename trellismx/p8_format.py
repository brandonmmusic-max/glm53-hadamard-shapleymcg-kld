"""Portable, model-independent TrellisMX P8 tensor container."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import safe_open, save_file

from .protocol import (
    P8_ALPHABET,
    P8_BLOCK_SIZE,
    P8_COMPANDER_SCALE,
    P8_LAW,
    P8_MMA,
    P8_RATES,
    decode_p8_payload,
    pack_trellis_edges,
    p8_state_table,
)


P8_TENSOR_SCHEMA = "trellismx.p8-tensors.v1"
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")


@dataclass(frozen=True)
class P8TensorPayload:
    name: str
    rows: int
    width: int
    bits: int
    trellis: torch.Tensor
    scale_ue8m0: torch.Tensor

    def __post_init__(self) -> None:
        validate_p8_tensor_payload(self)

    @property
    def trellis_key(self) -> str:
        return f"tensor.{self.name}.trellis"

    @property
    def scale_key(self) -> str:
        return f"tensor.{self.name}.scale_ue8m0"

    def storage(self) -> dict[str, Any]:
        weights = self.rows * self.width
        stream_bytes = self.trellis.numel() * self.trellis.element_size()
        scale_bytes = self.scale_ue8m0.numel() * self.scale_ue8m0.element_size()
        return {
            "name": self.name,
            "shape": [self.rows, self.width],
            "bits": self.bits,
            "logical_weights": weights,
            "trellis_bytes": stream_bytes,
            "scale_ue8m0_bytes": scale_bytes,
            "payload_bytes": stream_bytes + scale_bytes,
            "payload_bpw": 8.0 * (stream_bytes + scale_bytes) / weights,
        }

    def decode(self, *, device: str | torch.device = "cpu") -> torch.Tensor:
        return decode_p8_payload(
            self.trellis,
            p8_state_table(self.bits, device=device),
            self.scale_ue8m0,
            bits=self.bits,
            rows=self.rows,
            width=self.width,
            device=device,
        )


def validate_p8_tensor_payload(payload: P8TensorPayload) -> None:
    if not _NAME.fullmatch(payload.name):
        raise ValueError("P8 tensor names must be stable safe identifiers")
    if payload.bits not in P8_RATES:
        raise ValueError("portable P8 payload rate must be K3, K4, or K5")
    if payload.rows <= 0 or payload.rows % 16 or payload.width <= 0 or payload.width % 32:
        raise ValueError("portable P8 rows must be divisible by 16 and width by 32")
    expected_trellis = (
        payload.width // 16,
        payload.rows // 16,
        16 * payload.bits,
    )
    if payload.trellis.dtype != torch.int16 or tuple(payload.trellis.shape) != expected_trellis:
        raise ValueError(
            f"invalid trellis for {payload.name}: expected int16 {expected_trellis}"
        )
    expected_scale = (payload.rows, payload.width // P8_BLOCK_SIZE)
    if (
        payload.scale_ue8m0.dtype != torch.uint8
        or tuple(payload.scale_ue8m0.shape) != expected_scale
    ):
        raise ValueError(
            f"invalid UE8M0 scale for {payload.name}: expected {expected_scale}"
        )
    if bool((payload.scale_ue8m0 == 255).any()):
        raise ValueError("UE8M0 code 255 is reserved and cannot be stored")


def _contract_metadata(records: Mapping[str, P8TensorPayload]) -> dict[str, str]:
    record_values: dict[str, dict[str, int]] = {}
    for name, payload in records.items():
        if name != payload.name:
            raise ValueError("payload registry key and name differ")
        record_values[name] = {
            "rows": payload.rows,
            "width": payload.width,
            "bits": payload.bits,
        }
    return {
        "schema": P8_TENSOR_SCHEMA,
        "alphabet": P8_ALPHABET,
        "law": P8_LAW,
        "mma": P8_MMA,
        "block_size": str(P8_BLOCK_SIZE),
        "compander_scale": format(P8_COMPANDER_SCALE, ".17g"),
        "codebook": "procedural-not-stored",
        "ldlq": "false",
        "records": json.dumps(record_values, sort_keys=True, separators=(",", ":")),
    }


def tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def write_p8_tensor_file(
    path: Path,
    payloads: Mapping[str, P8TensorPayload],
    *,
    source_metadata: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Write a safe multi-tensor payload; refuses overwrite."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    tensors: dict[str, torch.Tensor] = {}
    for name, payload in payloads.items():
        tensors[payload.trellis_key] = payload.trellis.contiguous()
        tensors[payload.scale_key] = payload.scale_ue8m0.contiguous()
    metadata = _contract_metadata(payloads)
    metadata.update(
        {
            f"sha256.{name}.{field}": tensor_sha256(getattr(payload, field))
            for name, payload in payloads.items()
            for field in ("trellis", "scale_ue8m0")
        }
    )
    if source_metadata:
        metadata.update(
            {f"source.{key}": str(value) for key, value in source_metadata.items()}
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, path, metadata=metadata)
    return read_p8_tensor_file(path).storage_report()


@dataclass(frozen=True)
class P8TensorFile:
    path: str
    records: dict[str, P8TensorPayload]
    file_sha256: str
    source_metadata: dict[str, str]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.records))

    def decode(self, name: str, *, device: str | torch.device = "cpu") -> torch.Tensor:
        return self.records[name].decode(device=device)

    def storage_report(self) -> dict[str, Any]:
        payload_storage = [self.records[name].storage() for name in self.names()]
        payload_bytes = sum(item["payload_bytes"] for item in payload_storage)
        logical_weights = sum(item["logical_weights"] for item in payload_storage)
        file_bytes = Path(self.path).stat().st_size
        return {
            "schema": P8_TENSOR_SCHEMA,
            "path": self.path,
            "file_sha256": self.file_sha256,
            "tensors": payload_storage,
            "payload_bytes": payload_bytes,
            "logical_weights": logical_weights,
            "aggregate_payload_bpw": 8.0 * payload_bytes / logical_weights,
            "file_bytes": file_bytes,
            "file_bpw": 8.0 * file_bytes / logical_weights,
            "codebook_bytes": 0,
            "safe_loading": "safetensors-no-pickle",
        }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_p8_tensor_file(
    path: Path, *, expected_sha256: str | None = None
) -> P8TensorFile:
    expected_fields = {
        "schema": P8_TENSOR_SCHEMA,
        "alphabet": P8_ALPHABET,
        "law": P8_LAW,
        "mma": P8_MMA,
        "block_size": str(P8_BLOCK_SIZE),
        "compander_scale": format(P8_COMPANDER_SCALE, ".17g"),
        "codebook": "procedural-not-stored",
        "ldlq": "false",
    }
    with safe_open(path, framework="pt", device="cpu") as source:
        metadata = source.metadata() or {}
        mismatches = {
            key: {"expected": value, "observed": metadata.get(key)}
            for key, value in expected_fields.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"portable P8 metadata mismatch: {mismatches}")
        try:
            record_metadata = json.loads(metadata.get("records", ""))
        except json.JSONDecodeError as error:
            raise ValueError("portable P8 records metadata is not valid JSON") from error
        if not isinstance(record_metadata, dict):
            raise ValueError("portable P8 records must be an object")
        expected_keys: set[str] = set()
        for name, record in record_metadata.items():
            if not isinstance(record, dict) or set(record) != {"rows", "width", "bits"}:
                raise ValueError(f"invalid portable P8 record {name!r}")
            expected_keys.add(f"tensor.{name}.trellis")
            expected_keys.add(f"tensor.{name}.scale_ue8m0")
        if set(source.keys()) != expected_keys:
            raise ValueError("portable P8 tensor inventory is not exact")

        payloads: dict[str, P8TensorPayload] = {}
        for name, record in record_metadata.items():
            payload = P8TensorPayload(
                name=str(name),
                rows=int(record["rows"]),
                width=int(record["width"]),
                bits=int(record["bits"]),
                trellis=source.get_tensor(f"tensor.{name}.trellis")
                .clone()
                .contiguous(),
                scale_ue8m0=source.get_tensor(f"tensor.{name}.scale_ue8m0")
                .clone()
                .contiguous(),
            )
            expected_trellis_hash = metadata.get(f"sha256.{name}.trellis")
            expected_scale_hash = metadata.get(f"sha256.{name}.scale_ue8m0")
            if expected_trellis_hash != tensor_sha256(payload.trellis):
                raise ValueError(f"portable P8 trellis hash mismatch for {name}")
            if expected_scale_hash != tensor_sha256(payload.scale_ue8m0):
                raise ValueError(f"portable P8 scale hash mismatch for {name}")
            payloads[name] = payload

    file_hash = _file_sha256(path)
    if expected_sha256 is not None and file_hash != expected_sha256:
        raise ValueError("portable P8 file hash mismatch")
    source_metadata = {
        key.removeprefix("source."): value
        for key, value in metadata.items()
        if key.startswith("source.")
    }
    return P8TensorFile(
        path=str(path),
        records=payloads,
        file_sha256=file_hash,
        source_metadata=source_metadata,
    )


def synthetic_p8_payload(
    name: str, *, bits: int, seed: int = 20260907
) -> P8TensorPayload:
    generator = torch.Generator().manual_seed(seed)
    edges = torch.randint(
        0,
        1 << bits,
        (2, 1, 256),
        dtype=torch.int64,
        generator=generator,
    )
    return P8TensorPayload(
        name=name,
        rows=16,
        width=32,
        bits=bits,
        trellis=pack_trellis_edges(edges, bits),
        scale_ue8m0=torch.randint(
            0, 255, (16, 1), dtype=torch.int64, generator=generator
        ).to(torch.uint8),
    )
