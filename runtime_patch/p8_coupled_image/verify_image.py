#!/usr/bin/env python3
"""CPU-only installed-source verifier for the coupled P8 research image."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import py_compile
import sys


DEFAULT_MANIFEST = Path("/opt/p8-coupled-runtime/image-manifest.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _origin(module: str) -> str | None:
    spec = importlib.util.find_spec(module)
    return None if spec is None else spec.origin


def _parameters(callable_object: object) -> list[str]:
    """Return the positional source ABI, excluding the method receiver."""

    names = list(inspect.signature(callable_object).parameters)
    return names[1:] if names and names[0] == "self" else names


def _call_owner(cls: type) -> str | None:
    for base in cls.__mro__:
        if "__call__" in base.__dict__:
            return f"{base.__module__}.{base.__name__}"
    return None


def _fc1_resource_contract(instance: object) -> dict[str, int]:
    """Calculate the physical M64 FC1 shared-memory intervals.

    The two-stage MMA pipeline aliases the epilogue only after the kernel's
    wait/fence/barrier.  Within the epilogue, gate FP16, up FP16, and the
    transformed FP32 output must be pairwise disjoint.
    """

    tile_elements = int(instance.tile_m) * int(instance.owned_n)
    gate_end = tile_elements * 2
    up_end = gate_end + tile_elements * 2
    full_output_end = up_end + tile_elements * 4
    return {
        "pipeline_start": 0,
        "pipeline_end": 2 * int(instance.stage_bytes),
        "gate_fp16_start": 0,
        "gate_fp16_end": gate_end,
        "up_fp16_start": gate_end,
        "up_fp16_end": up_end,
        "full_output_fp32_start": up_end,
        "full_output_fp32_end": full_output_end,
        "allocation_end": int(instance.shared_bytes),
    }


def verify(manifest_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text())
    errors: list[str] = []
    installed_hashes: dict[str, str] = {}
    parent_phase_hashes: dict[str, str | None] = {}
    parent_phase_abis: dict[str, list[str] | None] = {}
    candidate_abis: dict[str, dict[str, object]] = {}

    for source, destinations in manifest["install"].items():
        expected = manifest["source_sha256"][source]
        for destination in destinations:
            path = Path(destination)
            if not path.is_file():
                errors.append(f"missing installed source: {destination}")
                continue
            actual = sha256(path)
            installed_hashes[destination] = actual
            if actual != expected:
                errors.append(
                    f"installed source hash mismatch: {destination}: "
                    f"expected {expected}, got {actual}"
                )

    tail = Path(manifest["tail_v2"]["path"])
    tail_actual = sha256(tail) if tail.is_file() else None
    if tail_actual != manifest["tail_v2"]["sha256"]:
        errors.append(
            f"tail-V2 identity mismatch: expected "
            f"{manifest['tail_v2']['sha256']}, got {tail_actual}"
        )

    origins = {
        module: _origin(module) for module in manifest["required_import_origins"]
    }
    for module, expected in manifest["required_import_origins"].items():
        if origins[module] != expected:
            errors.append(
                f"import origin mismatch: {module}: expected {expected}, "
                f"got {origins[module]}"
            )

    parent_contract = manifest["parent_phase_abi"]
    if any("/build/lib" in entry for entry in sys.path):
        errors.append("stale build/lib donor is present on sys.path")
    for phase_name, phase in parent_contract["phases"].items():
        for destination in phase["paths"]:
            path = Path(destination)
            actual = sha256(path) if path.is_file() else None
            parent_phase_hashes[destination] = actual
            if actual != phase["sha256"]:
                errors.append(
                    f"parent {phase_name} hash mismatch: {destination}: "
                    f"expected {phase['sha256']}, got {actual}"
                )
        module = importlib.import_module(phase["module"])
        cls = getattr(module, phase["class"])
        actual_parameters = _parameters(cls.__call__)
        parent_phase_abis[phase_name] = actual_parameters
        if actual_parameters != phase["call_parameters"]:
            errors.append(
                f"parent {phase_name} call ABI mismatch: expected "
                f"{phase['call_parameters']}, got {actual_parameters}"
            )

    active_hashes = set(parent_phase_hashes.values())
    stale_overlap = active_hashes.intersection(parent_contract["rejected_donor_sha256"])
    if stale_overlap:
        errors.append(f"active parent phase uses rejected stale donor: {stale_overlap}")

    for candidate_name, candidate in manifest["candidate_launch_abi"].items():
        module = importlib.import_module(candidate["module"])
        cls = getattr(module, candidate["class"])
        constructor_parameters = list(inspect.signature(cls).parameters)
        actual_parameters = _parameters(cls.__call__)
        owner = _call_owner(cls)
        source = "".join(inspect.getsource(cls.__call__).split())
        instance = cls()
        instance_attributes = {
            name: getattr(instance, name, None)
            for name in candidate["instance_attributes"]
        }
        resource_contract = (
            _fc1_resource_contract(instance)
            if "resource_contract" in candidate
            else None
        )
        candidate_abis[candidate_name] = {
            "constructor_parameters": constructor_parameters,
            "call_parameters": actual_parameters,
            "call_owner": owner,
            "instance_attributes": instance_attributes,
            "resource_contract": resource_contract,
        }
        if constructor_parameters != candidate["constructor_parameters"]:
            errors.append(
                f"candidate {candidate_name} constructor ABI mismatch: expected "
                f"{candidate['constructor_parameters']}, got {constructor_parameters}"
            )
        if actual_parameters != candidate["call_parameters"]:
            errors.append(
                f"candidate {candidate_name} call ABI mismatch: expected "
                f"{candidate['call_parameters']}, got {actual_parameters}"
            )
        if owner != candidate["call_owner"]:
            errors.append(
                f"candidate {candidate_name} call owner mismatch: expected "
                f"{candidate['call_owner']}, got {owner}"
            )
        if instance_attributes != candidate["instance_attributes"]:
            errors.append(
                f"candidate {candidate_name} constructed attributes mismatch: expected "
                f"{candidate['instance_attributes']}, got {instance_attributes}"
            )
        if resource_contract != candidate.get("resource_contract"):
            errors.append(
                f"candidate {candidate_name} resource contract mismatch: expected "
                f"{candidate.get('resource_contract')}, got {resource_contract}"
            )
        for snippet in candidate["normalized_launch_source"]:
            if snippet not in source:
                errors.append(
                    f"candidate {candidate_name} launch source missing: {snippet}"
                )

    compile_targets = sorted(installed_hashes)
    for target in compile_targets:
        try:
            py_compile.compile(target, doraise=True)
        except py_compile.PyCompileError as error:
            errors.append(f"py_compile failed: {target}: {error.msg}")

    result: dict[str, object] = {
        "schema": "glm53.p8-coupled-image-verification.v4",
        "status": "pass" if not errors else "fail",
        "parent_image_id": manifest["parent_image_id"],
        "tail_v2_sha256": tail_actual,
        "installed_sha256": installed_hashes,
        "import_origins": origins,
        "parent_phase_sha256": parent_phase_hashes,
        "parent_phase_call_abi": parent_phase_abis,
        "candidate_launch_abi": candidate_abis,
        "py_compile_targets": compile_targets,
        "gpu_used": False,
        "device_closure_tested": False,
        "errors": errors,
    }
    if errors:
        raise RuntimeError(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    print(json.dumps(verify(args.manifest), sort_keys=True))


if __name__ == "__main__":
    main()
