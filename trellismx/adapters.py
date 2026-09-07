"""Fail-closed architecture adapter registry.

The tensor codec is model-independent, but a real conversion needs a mapping
for named weights, tensor-parallel shards, calibration captures, transforms,
and runtime integration.  Unsupported architectures are rejected before any
checkpoint path is opened.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .protocol import P8EncoderConfig


class UnsupportedArchitectureError(ValueError):
    """Raised before model I/O when no qualified adapter exists."""


@dataclass(frozen=True)
class ArchitectureInfo:
    architecture: str
    boundary: str
    required_runtime: tuple[str, ...]


@dataclass(frozen=True)
class ArchitecturePlan:
    schema: str
    architecture: str
    boundary: str
    tensor_parallel_size: int
    layer_rates: dict[int, int]
    logical_weights_per_layer: int
    projections: tuple[str, ...]
    encoder: P8EncoderConfig
    execution: dict[str, bool]

    def summary(self) -> dict[str, Any]:
        rates = sorted(set(self.layer_rates.values()))
        return {
            "schema": self.schema,
            "architecture": self.architecture,
            "boundary": self.boundary,
            "tensor_parallel_size": self.tensor_parallel_size,
            "layers": len(self.layer_rates),
            "layer_range": [min(self.layer_rates), max(self.layer_rates)],
            "stored_rates": rates,
            "rate_counts": {str(rate): sum(x == rate for x in self.layer_rates.values()) for rate in rates},
            "logical_weights_per_layer": self.logical_weights_per_layer,
            "projections": list(self.projections),
            "encoder_base_contract": self.encoder.metadata(),
            "rate_payload_bpw": {
                str(rate): rate + 8.0 / self.encoder.block_size
                for rate in rates
            },
            "execution": self.execution,
        }


class ArchitectureAdapter(ABC):
    @property
    @abstractmethod
    def architecture(self) -> str: ...

    @property
    @abstractmethod
    def info(self) -> ArchitectureInfo: ...

    @abstractmethod
    def plan(self, layer_rates: Mapping[int, int], *, tensor_parallel_size: int) -> ArchitecturePlan: ...


class GLM53FlashAdapter(ArchitectureAdapter):
    layers = tuple(range(3, 45))
    experts = 288
    hidden_size = 4096
    intermediate_size = 2048
    projections = ("gate_proj", "up_proj", "down_proj")
    boundary = "coupled-h512-h128-suh-svh-v1"

    @property
    def architecture(self) -> str:
        return "glm53-flash"

    @property
    def info(self) -> ArchitectureInfo:
        return ArchitectureInfo(
            architecture=self.architecture,
            boundary=self.boundary,
            required_runtime=(
                "B12X_MLA_SPARSE",
                "native-p8-mxf8f6f4-n128",
                "tp4-dcp1-noep",
            ),
        )

    def plan(
        self,
        layer_rates: Mapping[int, int],
        *,
        tensor_parallel_size: int = 4,
        encoder: P8EncoderConfig | None = None,
    ) -> ArchitecturePlan:
        if tensor_parallel_size != 4:
            raise ValueError("the measured GLM-5.3-Flash TrellisMX path requires TP4")
        if set(layer_rates) != set(self.layers):
            missing = sorted(set(self.layers) - set(layer_rates))
            extra = sorted(set(layer_rates) - set(self.layers))
            raise ValueError(f"GLM layer mapping mismatch: missing={missing}, extra={extra}")
        invalid = {layer: rate for layer, rate in layer_rates.items() if rate not in (3, 4, 5)}
        if invalid:
            raise ValueError(f"unsupported P8 rates {invalid}")
        return ArchitecturePlan(
            schema="trellismx.glm53-flash.plan.v1",
            architecture=self.architecture,
            boundary=self.boundary,
            tensor_parallel_size=tensor_parallel_size,
            layer_rates=dict(layer_rates),
            logical_weights_per_layer=self.experts * 3 * self.hidden_size * self.intermediate_size,
            projections=self.projections,
            encoder=encoder or P8EncoderConfig(),
            execution={"reads_checkpoint": False, "encodes_tensors": False, "uses_device": False},
        )

    def inspect_checkpoint(
        self, root: Path, *, index_path: Path | None = None, verify_files: bool = True
    ) -> dict[str, Any]:
        """Validate config/index mapping without reading safetensors payloads."""

        root = root.resolve()
        config_path = root / "config.json"
        index_file = index_path or (root / "model.safetensors.index.json")
        config_value = json.loads(config_path.read_text())
        text_config = config_value.get("text_config", config_value)
        observed = {
            "hidden_size": text_config.get("hidden_size"),
            "n_routed_experts": text_config.get("n_routed_experts"),
            "moe_intermediate_size": text_config.get("moe_intermediate_size"),
        }
        expected = {
            "hidden_size": self.hidden_size,
            "n_routed_experts": self.experts,
            "moe_intermediate_size": self.intermediate_size,
        }
        mismatches = {
            key: {"expected": value, "observed": observed[key]}
            for key, value in expected.items()
            if observed[key] != value
        }
        if mismatches:
            raise ValueError(f"GLM-5.3-Flash config mismatch: {mismatches}")

        index_value = json.loads(index_file.read_text())
        weight_map = index_value.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError("checkpoint index has no valid weight_map object")
        expected_names = {
            f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
            for layer in self.layers
            for expert in range(self.experts)
            for projection in self.projections
        }
        missing = sorted(expected_names - set(weight_map))
        if missing:
            raise ValueError(
                f"checkpoint index is missing {len(missing)} routed expert weights; "
                f"first={missing[0]}"
            )

        shard_names = sorted(set(weight_map.values()))
        invalid_shards: list[str] = []
        missing_files: list[str] = []
        shard_bytes = 0
        if verify_files:
            root_resolved = root.resolve()
            for shard in shard_names:
                if not isinstance(shard, str) or not shard:
                    invalid_shards.append(str(shard))
                    continue
                candidate = (root / shard).resolve()
                try:
                    candidate.relative_to(root_resolved)
                except ValueError:
                    invalid_shards.append(shard)
                    continue
                if not candidate.is_file():
                    missing_files.append(shard)
                else:
                    shard_bytes += candidate.stat().st_size
            if invalid_shards or missing_files:
                raise ValueError(
                    "checkpoint shard paths are invalid or missing: "
                    f"invalid={invalid_shards[:3]}, missing={missing_files[:3]}"
                )

        def file_hash(path: Path) -> str:
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()

        return {
            "schema": "trellismx.glm53-flash.checkpoint-inspection.v1",
            "status": "passed",
            "architecture": self.architecture,
            "config": observed,
            "expected_routed_weight_tensors": len(expected_names),
            "observed_routed_weight_tensors": len(expected_names),
            "layers": len(self.layers),
            "experts_per_layer": self.experts,
            "shards": len(shard_names),
            "verified_shard_bytes": shard_bytes,
            "config_sha256": file_hash(config_path),
            "index_sha256": file_hash(index_file),
            "safetensors_payload_bytes_read": 0,
            "checkpoint_tensor_reads": 0,
        }


_ADAPTERS: dict[str, ArchitectureAdapter] = {
    GLM53FlashAdapter().architecture: GLM53FlashAdapter(),
}


def adapter_for(architecture: str) -> ArchitectureAdapter:
    normalized = architecture.strip().lower()
    try:
        return _ADAPTERS[normalized]
    except KeyError as error:
        supported = ", ".join(sorted(_ADAPTERS))
        raise UnsupportedArchitectureError(
            f"unsupported architecture {architecture!r}; qualified adapters: {supported}"
        ) from error
