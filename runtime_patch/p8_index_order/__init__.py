"""Fail-closed opt-in loader for the P8 deterministic short-pool order."""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import hashlib
import linecache
import os
from pathlib import Path
import sys

from .patches import MODE, MODULE, ORIGINAL_SHA, transform


TRANSFORMATIONS: dict[str, dict] = {}


def _synthetic_filename(origin: str) -> str:
    return origin + ".p8logicalshortv1.py"


def _compile_transformed(source: str, origin: str):
    filename = _synthetic_filename(origin)
    lines = source.splitlines(keepends=True)
    linecache.cache[filename] = (len(source), None, lines, filename)
    return compile(source, filename, "exec", dont_inherit=True), filename


class _Loader(importlib.abc.Loader):
    def __init__(self, name: str, origin: str):
        self.name, self.origin = name, origin

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        if self.name in TRANSFORMATIONS:
            raise ImportError("duplicate P8 index-order source transformation")
        source, receipt = transform(self.name, Path(self.origin).read_bytes())
        code, filename = _compile_transformed(source, self.origin)
        # inspect.getsource(class) follows module.__file__, whereas functions
        # follow co_filename. Point both at the same linecache-backed source so
        # CuTeDSL never retraces the unmodified on-disk file under the new key.
        module.__file__ = filename
        visible = "".join(linecache.getlines(filename))
        visible_sha = hashlib.sha256(visible.encode("utf-8")).hexdigest()
        if visible_sha != receipt["emitted_sha256"]:
            linecache.cache.pop(filename, None)
            raise ImportError("transformed P8 index-order source is not inspect-visible")
        try:
            exec(code, module.__dict__)
        except BaseException:
            linecache.cache.pop(filename, None)
            raise
        TRANSFORMATIONS[self.name] = {
            **receipt,
            "origin": self.origin,
            "synthetic_filename": filename,
            "inspect_source_sha256": visible_sha,
        }


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not spec.origin or not spec.origin.endswith(".py"):
            raise ImportError(f"P8 index-order cannot resolve exact source: {fullname}")
        spec.loader = _Loader(fullname, spec.origin)
        return spec


def install() -> None:
    if os.environ.get("GLM53_P8_INDEX_ORDER") != MODE:
        raise RuntimeError(f"P8 index order requires GLM53_P8_INDEX_ORDER={MODE}")
    if MODULE in sys.modules:
        raise RuntimeError("P8 index order must install before fused-indexer import")
    if any(isinstance(finder, _Finder) for finder in sys.meta_path):
        raise RuntimeError("duplicate P8 index-order installation")
    sys.meta_path.insert(0, _Finder())
