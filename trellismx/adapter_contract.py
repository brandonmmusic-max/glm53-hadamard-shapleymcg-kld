"""Generic architecture adapter contract for downstream TrellisMX implementers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


ADAPTER_CONTRACT_SCHEMA = "trellismx.architecture-adapter.v1"
REQUIRED_SECTIONS = {
    "architecture",
    "checkpoint",
    "tensor_mapping",
    "calibration",
    "encoding",
    "output",
    "runtime",
    "qualification",
}
REQUIRED_RUNTIME = {
    "compute_operand": "e4m3",
    "mma": "mxf8f6f4",
    "scale": "ue8m0-k32",
    "rates": [3, 4, 5],
    "unsupported_shape_policy": "fail-closed",
}


@dataclass(frozen=True)
class ArchitectureAdapterContract:
    architecture: str
    payload: dict[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ArchitectureAdapterContract":
        if not isinstance(value, dict):
            raise ValueError("adapter contract must be an object")
        missing = REQUIRED_SECTIONS - set(value)
        if missing:
            raise ValueError(
                "adapter contract is missing sections: " + ", ".join(sorted(missing))
            )
        architecture = value["architecture"]
        if not isinstance(architecture, dict) or not isinstance(
            architecture.get("name"), str
        ):
            raise ValueError("adapter contract requires architecture.name")
        if not isinstance(value["tensor_mapping"], dict) or not isinstance(
            value["tensor_mapping"].get("named_weight_families"), list
        ):
            raise ValueError("adapter contract requires named_weight_families")
        if not value["tensor_mapping"]["named_weight_families"]:
            raise ValueError("at least one named weight family is required")
        if value["tensor_mapping"].get("unsupported_architecture_policy") != "fail-closed":
            raise ValueError("unsupported architectures must fail closed")
        if not isinstance(value["calibration"], dict) or value["calibration"].get(
            "role_manifest_schema"
        ) not in {"trellismx.roles.v1", "glm53-codec.conditional-fit32-development-role.v1"}:
            raise ValueError("adapter must use a TrellisMX role-manifest schema")
        if not isinstance(value["encoding"], dict) or value["encoding"].get(
            "encoder"
        ) != "trellismx-native-coupled":
            raise ValueError("adapter must declare the TrellisMX coupled encoder")
        if not isinstance(value["output"], dict) or value["output"].get(
            "portable_tensor_schema"
        ) != "trellismx.p8-tensors.v1":
            raise ValueError("adapter must emit portable P8 tensors")
        runtime = value.get("runtime")
        if not isinstance(runtime, dict):
            raise ValueError("adapter runtime contract must be an object")
        for key, expected in REQUIRED_RUNTIME.items():
            if runtime.get(key) != expected:
                raise ValueError(f"adapter runtime field {key!r} violates TrellisMX P8")
        qualification = value.get("qualification")
        if not isinstance(qualification, dict) or qualification.get(
            "owner"
        ) != "downstream-adapter-implementer":
            raise ValueError(
                "generic adapter qualification is owned by the downstream implementer"
            )
        if qualification.get("device_gates_executed_by_this_package") is not False:
            raise ValueError("adapter contract must not claim package-executed device gates")
        return cls(architecture=architecture["name"], payload=dict(value))

    @classmethod
    def from_file(cls, path: Path) -> "ArchitectureAdapterContract":
        return cls.from_mapping(json.loads(path.read_text()))

    def summary(self) -> dict[str, Any]:
        return {
            "schema": ADAPTER_CONTRACT_SCHEMA,
            "architecture": self.architecture,
            "bundled_qualification": False,
            "qualification_owner": "downstream-adapter-implementer",
            "contract_valid": True,
            "uses_device": False,
        }
