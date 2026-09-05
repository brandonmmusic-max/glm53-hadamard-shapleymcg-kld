"""Structural adapter gates; integration must prove real dispatch receipts."""
import ast
from pathlib import Path


SOURCE=Path(__file__).resolve().parents[1]/'runtime_patch/sitecustomize.py'


def test_serving_adapter_passes_both_explicit_options():
    tree=ast.parse(SOURCE.read_text())
    calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)
           and isinstance(node.func,ast.Name) and node.func.id=='_P8NativeTPMoE']
    assert len(calls)==1
    keys={keyword.arg:ast.unparse(keyword.value) for keyword in calls[0].keywords}
    assert keys['fc1_tile_n']=='_P8N_FC1_TILE_N'
    assert keys['fuse_scratch_zero']=='_P8N_FUSED_SCRATCH'


def test_m1_receipt_is_after_runtime_call_and_guarded_by_actual_shape():
    tree=ast.parse(SOURCE.read_text())
    run=next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=='_p8n_run')
    assert isinstance(run.body[0],ast.Assign)
    assert isinstance(run.body[1],ast.If)
    assert 'x.shape[0] == 1' in ast.unparse(run.body[1].test)
    assert 'GLM53_P8_M1_DISPATCH' in ast.unparse(run.body[1])
