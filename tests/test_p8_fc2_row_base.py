"""Regression for the observed grouped FC2 DSL undefined row origin."""
import ast
from pathlib import Path


def test_row_base_defined_before_staged_control_flow():
    path = Path(__file__).resolve().parents[1] / "runtime_patch/b12x_h16/b12x/moe/_shared/kernels/p8_small_m.py"
    tree = ast.parse(path.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_run_task")
    assignment = next(n for n in method.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "physical_row_base" for t in n.targets))
    assert ast.unparse(assignment.value) == "source_m_tile * Int32(self.source_tile_m) + m_half * Int32(self.tile_m)"
    first_branch = next(n for n in method.body if isinstance(n, (ast.If, ast.For, ast.While)))
    assert assignment.lineno < first_branch.lineno
    for tile in range(4):
        # Grouped M64 uses m_half=0; every hrow maps to its own physical row.
        assert [tile * 64 + row for row in range(64)] == list(range(tile * 64, (tile + 1) * 64))
