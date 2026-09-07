"""Provenance overlays and optional encoder-backend declarations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


PROVENANCE_SCHEMA = "trellismx.provenance-overlay.v1"
ENCODER_BACKEND_SCHEMA = "trellismx.encoder-backend.v1"
SUPPORTED_BACKENDS = (
    "trellismx-native-coupled",
    "kquant-qsrt-viterbi",
)


def _sha256(value: Any, field: str, *, required: bool = True) -> str | None:
    if value is None and not required:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{field} must be a lowercase 64-character SHA-256")
    return value


@dataclass(frozen=True)
class ArtifactReference:
    name: str
    kind: str
    uri: str
    sha256: str
    bytes: int | None = None

    @classmethod
    def from_mapping(cls, name: str, value: Mapping[str, Any]) -> "ArtifactReference":
        if not isinstance(value, dict):
            raise ValueError(f"provenance artifact {name!r} must be an object")
        required = {"kind", "uri", "sha256"}
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"artifact {name!r} is missing {', '.join(missing)}")
        uri = value["uri"]
        if not isinstance(uri, str) or not uri or any(char in uri for char in "\r\n"):
            raise ValueError(f"artifact {name!r} URI must be a single nonempty string")
        size = value.get("bytes")
        if size is not None and (not isinstance(size, int) or size < 0):
            raise ValueError(f"artifact {name!r} byte count must be a nonnegative integer")
        return cls(
            name=name,
            kind=str(value["kind"]),
            uri=uri,
            sha256=_sha256(value["sha256"], f"artifact {name!r} sha256"),
            bytes=size,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "uri_read": False,
        }


@dataclass(frozen=True)
class ProvenanceOverlay:
    workflow: str
    artifacts: dict[str, ArtifactReference]
    source_path: str | None = None

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, source_path: str | None = None
    ) -> "ProvenanceOverlay":
        if value.get("schema") != PROVENANCE_SCHEMA:
            raise ValueError(f"provenance overlay schema must be {PROVENANCE_SCHEMA}")
        workflow = value.get("workflow")
        raw = value.get("artifacts")
        if not isinstance(workflow, str) or not workflow:
            raise ValueError("provenance workflow name must be nonempty")
        if not isinstance(raw, dict) or not raw:
            raise ValueError("provenance artifacts must be a nonempty object")
        artifacts = {
            str(name): ArtifactReference.from_mapping(str(name), artifact)
            for name, artifact in raw.items()
        }
        return cls(
            workflow=workflow,
            artifacts=artifacts,
            source_path=source_path,
        )

    @classmethod
    def from_file(cls, path: Path) -> "ProvenanceOverlay":
        return cls.from_mapping(json.loads(path.read_text()), source_path=str(path))

    def validate_for_workflow(self, config: Any) -> None:
        if self.workflow != config.name:
            raise ValueError("provenance overlay belongs to a different workflow")
        required_inputs = {
            name
            for name, declared in config.inputs.items()
            if declared.get("required")
        }
        optional_inputs = set(config.inputs) - required_inputs
        if not required_inputs <= set(self.artifacts):
            raise ValueError(
                "provenance overlay is missing one or more required workflow inputs"
            )
        if not set(self.artifacts) <= required_inputs | optional_inputs:
            raise ValueError(
                "provenance overlay contains an unknown optional workflow input"
            )
        for name in self.artifacts:
            declared = config.inputs[name]
            artifact = self.artifacts[name]
            required_kind = declared.get("kind")
            if required_kind is not None and artifact.kind != required_kind:
                raise ValueError(f"artifact {name!r} kind does not match workflow")

    def summary(self) -> dict[str, Any]:
        return {
            "schema": PROVENANCE_SCHEMA,
            "workflow": self.workflow,
            "source_path": self.source_path,
            "artifacts": {
                name: artifact.summary() for name, artifact in sorted(self.artifacts.items())
            },
            "credentials_stored": False,
            "artifact_bytes_read": 0,
        }


@dataclass(frozen=True)
class EncoderBackendConfig:
    backend: str = "trellismx-native-coupled"
    enabled: bool = False
    root_uri: str | None = None
    source_sha256: str | None = None
    license_status: str = "unresolved-unverified"
    device: str = "cuda"
    execution_authorized: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EncoderBackendConfig":
        if value.get("schema") != ENCODER_BACKEND_SCHEMA:
            raise ValueError(f"encoder backend schema must be {ENCODER_BACKEND_SCHEMA}")
        backend = value.get("backend")
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError("unsupported TrellisMX encoder backend")
        enabled = value.get("enabled")
        authorized = value.get("execution_authorized", False)
        if not isinstance(enabled, bool) or not isinstance(authorized, bool):
            raise ValueError("backend enabled/authorization flags must be boolean")
        device = value.get("device", "cuda")
        if device != "cuda":
            raise ValueError("the TrellisMX coupled encoder requires CUDA")
        result = cls(
            backend=str(backend),
            enabled=enabled,
            root_uri=value.get("root_uri"),
            source_sha256=value.get("source_sha256"),
            license_status=str(value.get("license_status", "unresolved-unverified")),
            device=device,
            execution_authorized=authorized,
        )
        result.validate()
        return result

    @classmethod
    def from_file(cls, path: Path) -> "EncoderBackendConfig":
        return cls.from_mapping(json.loads(path.read_text()))

    def validate(self) -> None:
        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError("unsupported TrellisMX encoder backend")
        if self.device != "cuda":
            raise ValueError("the TrellisMX coupled encoder requires CUDA")
        if not self.enabled:
            if self.execution_authorized:
                raise ValueError("a disabled backend cannot authorize execution")
            return
        if not isinstance(self.root_uri, str) or not self.root_uri:
            raise ValueError("enabled backend requires a pinned root URI")
        _sha256(self.source_sha256, "enabled backend source_sha256")
        if self.license_status == "unresolved-unverified":
            raise ValueError("enabled backend requires an operator-resolved license status")
        if (
            self.backend == "kquant-qsrt-viterbi"
            and self.license_status == "trellismx-source-available"
        ):
            raise ValueError(
                "optional KQuant/QSRT implementation requires its own license status"
            )

    @property
    def executable(self) -> bool:
        return bool(self.enabled and self.execution_authorized)

    def summary(self) -> dict[str, Any]:
        return {
            "schema": ENCODER_BACKEND_SCHEMA,
            "backend": self.backend,
            "enabled": self.enabled,
            "pinned": self.root_uri is not None and self.source_sha256 is not None,
            "license_status": self.license_status,
            "device": self.device,
            "execution_authorized": self.execution_authorized,
            "executable": self.executable,
        }
