"""Metadata for the source-only TrellisMX SM120 runtime overlay."""

from typing import Any


CAPABILITIES: dict[str, Any] = {
    "package": "trellismx-runtime-sm120",
    "schema": "glm53.p8-coupled-image-sources.v11",
    "kind": "source-overlay",
    "executable": False,
    "imports_cuda": False,
    "imports_b12x": False,
    "requires_separate_pinned_image": True,
}

__version__ = "0.1.0"
