"""Resumable TrellisMX encoding orchestration state."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .workflow import WorkflowConfig


ORCHESTRATION_SCHEMA = "trellismx.encoding-orchestration.v1"
TASK_STATUSES = ("pending", "running", "completed", "failed", "blocked")


@dataclass(frozen=True)
class EncodingTask:
    id: str
    kind: str
    dependencies: tuple[str, ...]
    requires_device: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "dependencies": list(self.dependencies),
            "requires_device": self.requires_device,
        }


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


class EncodingPlan:
    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.tasks: dict[str, EncodingTask] = {}
        self._add("inspect-checkpoint", "inspect", (), False)
        self._add("validate-provenance", "provenance", ("inspect-checkpoint",), False)
        for layer in sorted(config.rates):
            encode_id = f"encode-layer-{layer:03d}"
            pack_id = f"pack-layer-{layer:03d}"
            verify_id = f"verify-layer-{layer:03d}"
            self._add(
                encode_id,
                "encode-layer",
                ("validate-provenance",),
                True,
            )
            self._add(pack_id, "pack-layer", (encode_id,), False)
            self._add(verify_id, "verify-layer", (pack_id,), True)
        dependencies = tuple(
            f"verify-layer-{layer:03d}" for layer in sorted(config.rates)
        )
        self._add("checkpoint-manifest", "manifest", dependencies, False)
        self._add("future-evaluation", "evaluation", ("checkpoint-manifest",), True)

    def _add(
        self, identifier: str, kind: str, dependencies: tuple[str, ...], device: bool
    ) -> None:
        if identifier in self.tasks:
            raise ValueError(f"duplicate orchestration task {identifier!r}")
        missing = [task for task in dependencies if task not in self.tasks]
        if missing:
            raise ValueError(f"task {identifier!r} has unresolved dependencies {missing}")
        self.tasks[identifier] = EncodingTask(
            id=identifier,
            kind=kind,
            dependencies=dependencies,
            requires_device=device,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "schema": ORCHESTRATION_SCHEMA,
            "workflow": self.config.name,
            "task_count": len(self.tasks),
            "device_task_count": sum(
                task.requires_device for task in self.tasks.values()
            ),
            "layer_task_count": sum(
                task.kind in {"encode-layer", "pack-layer", "verify-layer"}
                for task in self.tasks.values()
            ),
            "executes_tasks": False,
            "uses_device": False,
            "tasks": [task.as_dict() for task in self.tasks.values()],
        }

    def initial_state(self) -> dict[str, Any]:
        value = {
            "schema": ORCHESTRATION_SCHEMA,
            "workflow": self.config.name,
            "status": "pending",
            "tasks": {
                task.id: {"status": "pending", "receipt_sha256": None}
                for task in self.tasks.values()
            },
        }
        value["state_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
        return value


class EncodingStateFile:
    def __init__(self, path: Path, plan: EncodingPlan):
        self.path = path
        self.plan = plan

    def load(self) -> dict[str, Any]:
        value = json.loads(self.path.read_text())
        if value.get("schema") != ORCHESTRATION_SCHEMA:
            raise ValueError("unsupported encoding orchestration state schema")
        if value.get("workflow") != self.plan.config.name:
            raise ValueError("encoding state belongs to a different workflow")
        if set(value.get("tasks", {})) != set(self.plan.tasks):
            raise ValueError("encoding state task inventory differs from plan")
        observable = {
            key: item for key, item in value.items() if key != "state_sha256"
        }
        expected = hashlib.sha256(_canonical(observable)).hexdigest()
        if value.get("state_sha256") != expected:
            raise ValueError("encoding state integrity hash mismatch")
        return value

    def _write(self, value: dict[str, Any]) -> None:
        observable = {
            key: item for key, item in value.items() if key != "state_sha256"
        }
        value["state_sha256"] = hashlib.sha256(_canonical(observable)).hexdigest()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp-{os.getpid()}")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)

    def initialize(self) -> dict[str, Any]:
        if self.path.exists():
            return self.load()
        value = self.plan.initial_state()
        self._write(value)
        return value

    def _validate_transition(
        self, state: dict[str, Any], task_id: str, new_status: str
    ) -> None:
        task = self.plan.tasks[task_id]
        current = state["tasks"][task_id]["status"]
        allowed = {
            "pending": {"running", "blocked", "failed"},
            "running": {"completed", "failed"},
            "completed": set(),
            "failed": {"pending"},
            "blocked": {"pending"},
        }
        if new_status not in allowed[current]:
            raise ValueError(
                f"invalid task transition {task_id}: {current} -> {new_status}"
            )
        if new_status == "running":
            incomplete = [
                dependency
                for dependency in task.dependencies
                if state["tasks"][dependency]["status"] != "completed"
            ]
            if incomplete:
                raise ValueError(f"task {task_id!r} dependencies are incomplete")

    def update(
        self,
        task_id: str,
        status: str,
        *,
        receipt_sha256: str | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        if task_id not in self.plan.tasks:
            raise ValueError(f"unknown encoding task {task_id!r}")
        if status not in TASK_STATUSES:
            raise ValueError(f"unsupported encoding task status {status!r}")
        state = self.load()
        self._validate_transition(state, task_id, status)
        if status == "completed":
            if receipt_sha256 is None or len(receipt_sha256) != 64:
                raise ValueError("completed encoding tasks require a 64-digit receipt hash")
        state["tasks"][task_id] = {
            "status": status,
            "receipt_sha256": receipt_sha256,
            **({"error": error} if error is not None else {}),
        }
        all_statuses = {item["status"] for item in state["tasks"].values()}
        state["status"] = (
            "completed"
            if all_statuses == {"completed"}
            else "running"
            if "running" in all_statuses
            else "pending"
        )
        self._write(state)
        return self.load()
