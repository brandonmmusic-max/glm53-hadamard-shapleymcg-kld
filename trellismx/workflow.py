"""Portable TrellisMX workflow configuration and source-stage mapping."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .adapters import ArchitecturePlan, adapter_for
from .protocol import P8EncoderConfig, P8_MMA


WORKFLOW_SCHEMA = "trellismx.workflow.v1"


@dataclass(frozen=True)
class WorkflowConfig:
    name: str
    architecture: str
    boundary: str
    tensor_parallel_size: int
    rates: dict[int, int]
    encoder: P8EncoderConfig
    inputs: dict[str, dict[str, Any]]
    execution: dict[str, Any]
    source_path: str | None = None

    @classmethod
    def from_file(cls, path: Path) -> "WorkflowConfig":
        value = json.loads(path.read_text())
        return cls.from_mapping(value, source_path=str(path))

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, source_path: str | None = None
    ) -> "WorkflowConfig":
        if value.get("schema") != WORKFLOW_SCHEMA:
            raise ValueError(f"workflow schema must be {WORKFLOW_SCHEMA}")
        required = ("name", "architecture", "boundary", "tensor_parallel_size", "rates", "inputs", "execution")
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"workflow is missing fields: {', '.join(missing)}")
        raw_rates = value["rates"]
        if not isinstance(raw_rates, dict):
            raise ValueError("workflow rates must be an object keyed by layer")
        rates: dict[int, int] = {}
        for layer, rate in raw_rates.items():
            layer_int, rate_int = int(layer), int(rate)
            if layer_int in rates:
                raise ValueError("duplicate layer in workflow rates")
            rates[layer_int] = rate_int
        inputs = dict(value["inputs"])
        execution = dict(value["execution"])
        if not isinstance(inputs, dict) or not isinstance(execution, dict):
            raise ValueError("workflow inputs and execution must be objects")
        if execution.get("device") != "cuda":
            raise ValueError("the current P8 encode workflow requires CUDA")
        if execution.get("authorized") is not False:
            raise ValueError("portable workflow files must not authorize device execution")
        raw_encoder = value.get("encoder")
        if not isinstance(raw_encoder, dict):
            raise ValueError("workflow encoder contract must be an object")
        allowed_encoder = {
            "alphabet",
            "law",
            "compander_scale",
            "block_size",
            "scale_refinement_iterations",
            "ldlq",
            "mma",
        }
        unexpected = sorted(set(raw_encoder) - allowed_encoder)
        if unexpected:
            raise ValueError(f"unsupported encoder fields: {', '.join(unexpected)}")
        if raw_encoder.get("mma") != P8_MMA:
            raise ValueError("TrellisMX P8 requires mxf8f6f4 compute operands")
        encoder_fields = {
            key: value for key, value in raw_encoder.items() if key != "mma"
        }
        # Per-layer rates are authoritative; K4 supplies the invariant base
        # contract for shared alphabet, law, scale, and refinement settings.
        encoder = P8EncoderConfig(bits=4, **encoder_fields)
        return cls(
            name=str(value["name"]),
            architecture=str(value["architecture"]),
            boundary=str(value["boundary"]),
            tensor_parallel_size=int(value["tensor_parallel_size"]),
            rates=rates,
            encoder=encoder,
            inputs=inputs,
            execution=execution,
            source_path=source_path,
        )

    def plan(self) -> ArchitecturePlan:
        selected = adapter_for(self.architecture)
        if selected.info.boundary != self.boundary:
            raise ValueError("workflow boundary does not match the architecture adapter")
        return selected.plan(
            self.rates,
            tensor_parallel_size=self.tensor_parallel_size,
            encoder=self.encoder,
        )


WORKFLOW_STAGES: tuple[dict[str, Any], ...] = (
    {
        "name": "inspect-checkpoint",
        "status": "research-source-available",
        "source": ["glm53_nvfp4/shard_index.py"],
        "requires_device": False,
    },
    {
        "name": "load-calibration-capture",
        "status": "research-source-available",
        "source": ["glm53_nvfp4/capture.py"],
        "requires_device": False,
    },
    {
        "name": "fit-hessian",
        "status": "research-source-available",
        "source": ["glm53_nvfp4/output_aware.py"],
        "requires_device": True,
    },
    {
        "name": "coupled-transform",
        "status": "glm53-flash-only",
        "source": ["glm53_nvfp4/p8_coupled_scale.py"],
        "requires_device": True,
    },
    {
        "name": "encode-p8-tensor",
        "status": "typed-wrapper-plus-research-source",
        "source": ["trellismx/protocol.py", "glm53_nvfp4/trellis_mxf.py"],
        "requires_device": True,
    },
    {
        "name": "pack-tp-sidecars",
        "status": "research-source-available",
        "source": ["glm53_nvfp4/build_p8_coupled_rate_tp4_sidecars.py"],
        "requires_device": False,
    },
    {
        "name": "verify-sidecars",
        "status": "research-source-available",
        "source": ["glm53_nvfp4/verify_p8_coupled_rate_tp4_sidecars.py"],
        "requires_device": True,
    },
    {
        "name": "build-manifest-and-ledger",
        "status": "research-source-available",
        "source": ["glm53_nvfp4/full_coupled_build_support.py"],
        "requires_device": False,
    },
    {
        "name": "runtime-integration",
        "status": "glm53-flash-research-only",
        "source": ["runtime_patch/p8_native_kernel.py", "glm53_nvfp4/p8_full_coupled_runtime.py"],
        "requires_device": True,
    },
    {
        "name": "device-closure-and-quality",
        "status": "not-run-by-this-package",
        "source": ["scripts/closure-repro/", "glm53_nvfp4/p8_full_coupled_cf32_executor.py"],
        "requires_device": True,
    },
)


def workflow_report(config: WorkflowConfig) -> dict[str, Any]:
    plan = config.plan()
    return {
        "schema": "trellismx.workflow-report.v1",
        "config": config.name,
        "config_source": config.source_path,
        "portable_config_contains_artifact_paths": False,
        "plan": plan.summary(),
        "stages": [dict(stage) for stage in WORKFLOW_STAGES],
    }


def workflow_report_with_provenance(
    config: WorkflowConfig,
    provenance: Any,
    encoder_backend: Any | None = None,
) -> dict[str, Any]:
    provenance.validate_for_workflow(config)
    report = workflow_report(config)
    report["provenance"] = provenance.summary()
    report["encoder_backend"] = (
        encoder_backend.summary() if encoder_backend is not None else None
    )
    return report
