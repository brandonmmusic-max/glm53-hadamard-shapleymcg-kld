#!/usr/bin/env python3
"""CPU-only installed-source verifier for the coupled P8 research image."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import py_compile


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


def verify(manifest_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text())
    errors: list[str] = []
    installed_hashes: dict[str, str] = {}

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

    compile_targets = sorted(installed_hashes)
    for target in compile_targets:
        try:
            py_compile.compile(target, doraise=True)
        except py_compile.PyCompileError as error:
            errors.append(f"py_compile failed: {target}: {error.msg}")

    result: dict[str, object] = {
        "schema": "glm53.p8-coupled-image-verification.v1",
        "status": "pass" if not errors else "fail",
        "parent_image_id": manifest["parent_image_id"],
        "tail_v2_sha256": tail_actual,
        "installed_sha256": installed_hashes,
        "import_origins": origins,
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
