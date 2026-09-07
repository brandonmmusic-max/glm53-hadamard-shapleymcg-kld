"""Typed calibration and evaluation role contracts for TrellisMX."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


ROLE_SCHEMA = "trellismx.roles.v1"
LEGACY_ROLE_SCHEMAS = {
    "glm53-nvfp4-v4.roles.v1",
    "glm53-codec.conditional-fit32-development-role.v1",
}
ROLE_NAMES = ("fit", "conditional-fit", "selection", "confirmation", "final")
EVALUATION_REFERENCE_SCHEMA = "trellismx.evaluation-reference.v1"

QAD_FUTURE_REFERENCE: dict[str, Any] = {
    "schema": EVALUATION_REFERENCE_SCHEMA,
    "status": "future-reference-only",
    "used_for_existing_trellismx_measurements": False,
    "existing_trellismx_measurement_provenance": "CF32 conditional-fit forced-decode campaign",
    "suite_id": "glm-5.3-flash-kimi-k3-source-fidelity-1024x-max2048-v1",
    "source_dataset": "festr2/kimi-k3-distribution-fidelity-1024x2048-v1",
    "pinned_revision": "402919ae70d61396087571b63fe9185d95491afb",
    "suite_manifest_sha256": "50520bdba81a9447b769e72da58720fb8468bb6dd19f641493c9110b04f9972b",
    "ordered_tokens_sha256": "e2c541bce4a3213f697cd5236eaa60393d044dc054ea3ec2cc35d79dcb089f9b",
    "partitions": {
        "analysis": {"contexts": 768, "scored_positions": 1_571_435},
        "qualification": {"contexts": 256, "scored_positions": 524_020},
    },
    "partition_invariant": "source clusters may not cross the analysis/qualification boundary",
    "metric": "full-vocabulary forward KL in FP64-compatible accumulation over aligned token positions",
    "routing_modes": ["natural-reference-vs-natural-candidate", "exact-bf16-route-replay"],
    "decision_rule_boundary": "freeze analysis decisions before reading qualification",
}


@dataclass(frozen=True)
class RoleWindow:
    id: str
    input_sha256: str
    prediction_positions: int
    role: str
    domain: str | None = None
    teacher_sha256: str | None = None
    teacher_source_role: str | None = None
    source_partition: str | None = None

    @classmethod
    def from_mapping(cls, role: str, value: Mapping[str, Any]) -> "RoleWindow":
        if not isinstance(value, dict):
            raise ValueError(f"{role} window must be an object")
        required = {"id", "input_sha256", "prediction_positions"}
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"{role} window is missing {', '.join(missing)}")
        identifier = value["id"]
        digest = value["input_sha256"]
        positions = value["prediction_positions"]
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"{role} window ID must be nonempty")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError(f"{role} window {identifier!r} has invalid input SHA-256")
        if not isinstance(positions, int) or positions <= 0:
            raise ValueError(f"{role} window {identifier!r} has invalid position count")
        return cls(
            id=identifier,
            input_sha256=digest,
            prediction_positions=positions,
            role=role,
            domain=value.get("domain"),
            teacher_sha256=value.get("teacher_sha256"),
            teacher_source_role=value.get(
                "teacher_source_role", value.get("experimental_role")
            ),
            source_partition=value.get("source_partition"),
        )


@dataclass(frozen=True)
class RoleManifest:
    schema: str
    roles: dict[str, tuple[RoleWindow, ...]]
    source: dict[str, Any] | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RoleManifest":
        schema = value.get("schema")
        if schema != ROLE_SCHEMA and schema not in LEGACY_ROLE_SCHEMAS:
            raise ValueError("unsupported TrellisMX role-manifest schema")
        raw_roles = value.get("roles")
        if not isinstance(raw_roles, dict):
            raise ValueError("role manifest must contain a roles object")
        observed = set(raw_roles)
        if schema == ROLE_SCHEMA and observed != set(ROLE_NAMES):
            raise ValueError(
                "TrellisMX role manifest must contain exactly the five sealed roles"
            )
        if schema in LEGACY_ROLE_SCHEMAS and not observed <= set(ROLE_NAMES):
            raise ValueError("legacy role manifest contains unsupported roles")
        roles: dict[str, tuple[RoleWindow, ...]] = {}
        all_windows: list[RoleWindow] = []
        input_roles: dict[str, str] = {}
        for role in ROLE_NAMES:
            raw = raw_roles.get(role, [])
            if not isinstance(raw, list):
                raise ValueError(f"role {role!r} must be a list")
            windows = tuple(RoleWindow.from_mapping(role, item) for item in raw)
            roles[role] = windows
            all_windows.extend(windows)
            for window in windows:
                if window.input_sha256 in input_roles:
                    raise ValueError("a token/input identity cannot occur in multiple roles")
                input_roles[window.input_sha256] = role
        ids = [window.id for window in all_windows]
        if len(ids) != len(set(ids)):
            raise ValueError("role window IDs must be globally unique")
        source = value.get("source")
        if source is not None and not isinstance(source, dict):
            raise ValueError("role-manifest source must be an object")
        return cls(schema=str(schema), roles=roles, source=source)

    @classmethod
    def from_file(cls, path: Path) -> "RoleManifest":
        return cls.from_mapping(json.loads(path.read_text()))

    def counts(self) -> dict[str, int]:
        return {role: len(windows) for role, windows in self.roles.items()}

    def validate_disjoint(self) -> None:
        by_input: dict[str, str] = {}
        for role, windows in self.roles.items():
            for window in windows:
                previous = by_input.setdefault(window.input_sha256, role)
                if previous != role:
                    raise ValueError(
                        f"input appears in both {previous!r} and {role!r}"
                    )

    def summary(self) -> dict[str, Any]:
        self.validate_disjoint()
        return {
            "schema": self.schema,
            "counts": self.counts(),
            "domains": {
                role: sorted(
                    {window.domain for window in windows if window.domain}
                )
                for role, windows in self.roles.items()
            },
            "source": self.source,
            "roles_disjoint": True,
            "qad_used_for_this_manifest": False,
        }


def evaluation_reference(value: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return a QAD future-reference contract without changing past provenance."""

    selected = dict(QAD_FUTURE_REFERENCE if value is None else value)
    if selected.get("schema") != EVALUATION_REFERENCE_SCHEMA:
        raise ValueError("unsupported TrellisMX evaluation-reference schema")
    if selected.get("used_for_existing_trellismx_measurements") is not False:
        raise ValueError(
            "QAD must not be represented as provenance for existing TrellisMX KLD results"
        )
    if set(selected.get("partitions", {})) != {"analysis", "qualification"}:
        raise ValueError(
            "evaluation reference requires analysis and qualification partitions"
        )
    return selected
