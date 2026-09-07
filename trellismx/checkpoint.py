"""Full TrellisMX checkpoint manifest planning."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .encoder import native_coupled_encoder
from .workflow import WorkflowConfig


CHECKPOINT_PLAN_SCHEMA = "trellismx.checkpoint-plan.v1"
CHECKPOINT_MANIFEST_SCHEMA = "trellismx.checkpoint-manifest.v1"


def sidecar_name(layer: int, rank: int) -> str:
    return f"p8-layer-{layer:03d}-tp4-rank-{rank}.safetensors"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


@dataclass(frozen=True)
class SidecarReference:
    layer: int
    rank: int
    path: str
    sha256: str
    bits: int
    bytes: int | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SidecarReference":
        required = {"layer", "rank", "path", "sha256", "bits"}
        if not isinstance(value, dict) or not required <= set(value):
            raise ValueError("sidecar reference is missing required fields")
        digest = value["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("sidecar reference has invalid SHA-256")
        size = value.get("bytes")
        if size is not None and (not isinstance(size, int) or size < 0):
            raise ValueError("sidecar byte count must be nonnegative")
        return cls(
            layer=int(value["layer"]),
            rank=int(value["rank"]),
            path=str(value["path"]),
            sha256=digest,
            bits=int(value["bits"]),
            bytes=size,
        )


class CheckpointPlan:
    def __init__(self, config: WorkflowConfig):
        self.config = config
        architecture_plan = config.plan()
        self.architecture_plan = architecture_plan

    def layer_rates(self) -> dict[int, int]:
        return dict(self.architecture_plan.layer_rates)

    def expected_sidecars(self) -> dict[tuple[int, int], str]:
        return {
            (layer, rank): sidecar_name(layer, rank)
            for layer in sorted(self.layer_rates())
            for rank in range(self.config.tensor_parallel_size)
        }

    def summary(self) -> dict[str, Any]:
        rates = self.layer_rates()
        expected = self.expected_sidecars()
        return {
            "schema": CHECKPOINT_PLAN_SCHEMA,
            "workflow": self.config.name,
            "architecture": self.config.architecture,
            "boundary": self.config.boundary,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "layers": len(rates),
            "rate_counts": {
                str(rate): sum(value == rate for value in rates.values())
                for rate in (3, 4, 5)
                if any(value == rate for value in rates.values())
            },
            "expected_sidecar_files": len(expected),
            "sidecar_naming": "p8-layer-{layer:03d}-tp4-rank-{rank}.safetensors",
            "logical_weights_per_layer": self.architecture_plan.logical_weights_per_layer,
            "encoder": native_coupled_encoder(),
            "execution": {"uses_device": False, "writes_checkpoint": False},
        }

    def manifest(
        self,
        sidecars: Mapping[tuple[int, int], Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        expected = self.expected_sidecars()
        supplied = sidecars or {}
        if set(supplied) != set(expected) and supplied:
            raise ValueError(
                "concrete checkpoint manifest must cover every expected layer/rank"
            )
        rates = self.layer_rates()
        payload_layers: dict[str, Any] = {}
        for layer in sorted(rates):
            ranks: dict[str, Any] = {}
            for rank in range(self.config.tensor_parallel_size):
                filename = expected[(layer, rank)]
                if supplied:
                    reference = SidecarReference.from_mapping(supplied[(layer, rank)])
                    if reference.layer != layer or reference.rank != rank:
                        raise ValueError("sidecar reference coordinates do not match key")
                    if reference.bits != rates[layer]:
                        raise ValueError("sidecar rate does not match the workflow plan")
                    ranks[str(rank)] = {
                        "file": reference.path,
                        "sha256": reference.sha256,
                        "bits": reference.bits,
                        "bytes": reference.bytes,
                    }
                else:
                    ranks[str(rank)] = {
                        "file": filename,
                        "bits": rates[layer],
                        "status": "planned",
                    }
            payload_layers[str(layer)] = {"bits": rates[layer], "ranks": ranks}

        value: dict[str, Any] = {
            "schema": CHECKPOINT_MANIFEST_SCHEMA,
            "status": "planned" if not supplied else "structurally-complete",
            "workflow": self.config.name,
            "architecture": self.config.architecture,
            "boundary": self.config.boundary,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "layers": payload_layers,
            "logical_weights_per_layer": self.architecture_plan.logical_weights_per_layer,
            "encoder": native_coupled_encoder(),
            "runtime": {
                "required_image": "separately-pinned-sm120-image",
                "mixed_rate_compile_key": "stored-trellis-rate",
                "loader_closure": "not tested by manifest construction",
            },
            "provenance": {
                "required_inputs": sorted(
                    name
                    for name, declared in self.config.inputs.items()
                    if declared.get("required")
                ),
                "optional_inputs": sorted(
                    name
                    for name, declared in self.config.inputs.items()
                    if not declared.get("required")
                ),
                "artifact_bytes_read": 0,
            },
        }
        canonical = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        value["payload_sha256"] = hashlib.sha256(canonical).hexdigest()
        return value


def write_manifest(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically write a checkpoint manifest without overwriting silently."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
