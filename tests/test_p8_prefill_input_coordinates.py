"""Guard rank-2 input coordinates: M1 cannot detect token/channel mixing."""
import ast
from pathlib import Path


def test_prefill_input_loads_use_explicit_token_channel_coordinates():
    path = Path(__file__).resolve().parents[1] / "runtime_patch/b12x_h16/b12x/moe/_shared/kernels/dynamic.py"
    tree = ast.parse(path.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_store_p8_full_coupled_input_row")
    loads = [n for n in ast.walk(method) if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == "a_input"]
    assert len(loads) == 4
    for load in loads:
        assert isinstance(load.slice, ast.Tuple)
        assert len(load.slice.elts) == 2
        assert ast.unparse(load.slice.elts[0]) == "token_idx"
