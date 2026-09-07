"""Fail-closed architecture adapter registry.

The tensor codec is model-independent, but a real conversion needs a mapping
for named weights, tensor-parallel shards, calibration captures, transforms,
and runtime integration.  Unsupported architectures are rejected before any
checkpoint path is opened.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
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
