"""Exercise the actual source dispatch predicate without importing GPU DSL."""
import ast
from pathlib import Path
from types import SimpleNamespace


def test_all_direct_m1_fc1_owners_receive_route_ids_not_grouped_zero_metadata():
    path = Path(__file__).resolve().parents[1] / "runtime_patch/b12x_h16/b12x/moe/_shared/kernels/dynamic.py"
    tree = ast.parse(path.read_text())
    conditions = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if any(isinstance(stmt, ast.Assign)
               and isinstance(stmt.value, ast.Name) and stmt.value.id == "topk_ids"
               and any(isinstance(t, ast.Name) and t.id == "phase1_experts" for t in stmt.targets)
               for stmt in node.body):
            conditions.append(node.test.args[0])
    assert len(conditions) == 1
    expression = compile(ast.Expression(conditions[0]), str(path), "eval")
    route_ids = [0, 1, 17, 63, 127, 191, 255, 287]
    grouped_ids = [0] * 8
    for width in (32, 64, 128):
        direct = eval(expression, {"self": SimpleNamespace(p8_small_m=True, p8_fc1_tile_n=width)})
        actual = route_ids if direct else grouped_ids
        assert actual == route_ids
    # Dense N128 prefill must keep grouped-task indexing, not direct route IDs.
    assert not eval(expression, {"self": SimpleNamespace(p8_small_m=False, p8_fc1_tile_n=128)})
