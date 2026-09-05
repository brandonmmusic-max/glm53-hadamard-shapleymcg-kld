"""Fail-closed opt-in import observer; no GPU work during installation."""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
from pathlib import Path
import sys

from .patches import ORIGINAL_SHA, transform

TRANSFORMATIONS: dict[str, dict] = {}


class _Loader(importlib.abc.Loader):
    def __init__(self, name, origin):
        self.name, self.origin = name, origin

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        source, receipt = transform(self.name, Path(self.origin).read_bytes())
        receipt['origin'] = self.origin
        TRANSFORMATIONS[self.name] = receipt
        exec(compile(source, self.origin, 'exec'), module.__dict__)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in ORIGINAL_SHA:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not spec.origin or not spec.origin.endswith('.py'):
            raise ImportError(f'index-trace cannot resolve exact source: {fullname}')
        spec.loader = _Loader(fullname, spec.origin)
        return spec


def install():
    if os.environ.get('GLM53_P8_INDEX_TRACE') != '1':
        raise RuntimeError('index trace requires explicit value1')
    if any(name in sys.modules for name in ORIGINAL_SHA):
        raise RuntimeError('index trace must install before its target imports')
    if any(isinstance(finder, _Finder) for finder in sys.meta_path):
        raise RuntimeError('duplicate index-trace installation')
    sys.meta_path.insert(0, _Finder())
