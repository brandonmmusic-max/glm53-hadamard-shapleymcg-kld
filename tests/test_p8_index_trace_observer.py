"""CPU policy tests; synthetic kernel execution has a separate exact-image script."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('trace_patches', ROOT / 'runtime_patch/p8_index_trace/patches.py')
patches = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patches)


def test_anchor_fail_closed():
    assert patches.replace_once('aba', 'b', 'x') == 'axa'
    for text in ('aaa', 'abba'):
        with pytest.raises(ValueError):
            patches.replace_once(text, 'b', 'x')


@pytest.mark.parametrize('name', list(patches.ORIGINAL_SHA))
def test_wrong_original_identity_rejected(name):
    with pytest.raises(ValueError, match='source identity'):
        patches.transform(name, b'# substituted source\n')


def test_all_sources_parse():
    for path in (ROOT / 'runtime_patch/p8_index_trace').glob('*.py'):
        ast.parse(path.read_text())


def test_graph_copy_has_no_host_observation():
    tree = ast.parse((ROOT / 'runtime_patch/p8_index_trace/observer.py').read_text())
    names = {'_copy', '_cache', 'observe_pool', 'observe_attention_inputs'}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            calls = [child for child in ast.walk(node) if isinstance(child, ast.Call)]
            assert not any(isinstance(c.func, ast.Attribute) and c.func.attr in {'item', 'cpu', 'numpy', 'sort', 'sort_'} for c in calls)


def test_startup_installs_before_native_imports():
    source = (ROOT / 'runtime_patch/sitecustomize.py').read_text()
    assert source.index('_install_index_trace()') < source.index('import vllm.models.glm5next.nvidia.model')


def test_buffers_cannot_be_allocated_inside_capture():
    source = (ROOT / 'runtime_patch/p8_index_trace/observer.py').read_text()
    start = source.index('if entry is None:')
    assert source.index('observer buffers must be allocated', start) < source.index('torch.zeros(', start)
