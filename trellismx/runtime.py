"""CPU-only inspection of the optional SM120 mixed-rate source overlay."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


RUNTIME_MANIFEST_SCHEMA = "glm53.p8-coupled-image-sources.v11"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_and_root() -> tuple[Path, Path, str]:
    spec = importlib.util.find_spec("trellismx_runtime_sm120")
    if spec is not None and spec.origin:
        package_root = Path(spec.origin).resolve().parent
        installed_manifest = package_root / "manifest.json"
        installed_root = package_root / "overlay"
        if installed_manifest.is_file() and installed_root.is_dir():
            return installed_manifest, installed_root, "installed-overlay"
    source_manifest = (
        Path(__file__).resolve().parents[1]
        / "runtime_patch/p8_mixed_rate_image/image_manifest.json"
    )
    if source_manifest.is_file():
        return source_manifest, source_manifest.parents[2], "source-checkout"
    raise FileNotFoundError("SM120 runtime overlay manifest is not installed")


def runtime_overlay_report() -> dict[str, Any]:
    """Hash-check the source overlay without importing CUDA, B12X, or vLLM."""

    manifest_path, root, origin = _manifest_and_root()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != RUNTIME_MANIFEST_SCHEMA:
        raise ValueError("unsupported SM120 runtime manifest schema")
    sources = manifest.get("source_sha256")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("SM120 runtime manifest has no source hash inventory")
    mismatches: dict[str, dict[str, str]] = {}
    for relative, expected in sorted(sources.items()):
        candidate = root / relative
        observed = _sha256(candidate) if candidate.is_file() else "missing"
        if observed != expected:
            mismatches[relative] = {"expected": expected, "observed": observed}
    return {
        "schema": "trellismx.sm120-runtime-report.v1",
        "status": "passed" if not mismatches else "failed",
        "origin": origin,
        "manifest_schema": RUNTIME_MANIFEST_SCHEMA,
        "runtime_commit": manifest.get("runtime_commit"),
        "source_files": len(sources),
        "source_mismatches": mismatches,
        "source_bytes_read": sum(
            (root / relative).stat().st_size
            for relative in sources
            if (root / relative).is_file()
        ),
        "executable": False,
        "dependencies_imported": False,
        "cuda_used": False,
        "requires_separate_pinned_image": True,
    }
