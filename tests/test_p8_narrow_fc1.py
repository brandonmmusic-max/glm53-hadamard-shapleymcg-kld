"""CPU structural/ownership gates; these do not establish CUDA arithmetic."""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "runtime_patch"
KERNEL = PATCH / "b12x_h16/b12x/moe/_shared/kernels/p8_narrow_fc1.py"


@pytest.mark.parametrize("width", [32, 64])
def test_narrow_owners_partition_payload_and_scale_bytes(width):
    # Reproduce the ABI addressing, including the shared K128 scale word.
    payload_owners, scale_owners = [], []
    rows_capacity, words_per_row = 8 * 16, 512 // 4
    for task in range(8 * 512 // width):
        tasks_per_route = 512 // width
        route, tile = divmod(task, tasks_per_route)
        output_tile, subtile = divmod(tile, 128 // width)
        for local_block in range(width // 32):
            block = local_block + subtile * (width // 32)
            payload_owners.extend(
                route * 16 * words_per_row + output_tile * 32 + block * 8 + word
                for word in range(8)
            )
            scale_owners.append(
                (rows_capacity * words_per_row + output_tile * rows_capacity
                 + route * 16) * 4 + block
            )
    assert len(payload_owners) == len(set(payload_owners)) == 8 * 512 // 4
    assert set(payload_owners) == {
        route * 16 * words_per_row + word
        for route in range(8) for word in range(words_per_row)
    }
    assert len(scale_owners) == len(set(scale_owners)) == 8 * 16
    assert set(scale_owners) == {
        (rows_capacity * words_per_row + tile * rows_capacity + route * 16) * 4 + byte
        for route in range(8) for tile in range(4) for byte in range(4)
    }


def test_candidate_syntax_and_activation_rounding_seam():
    source = KERNEL.read_text()
    tree = ast.parse(source)
    compile(tree, str(KERNEL), "exec")
    run = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
               and n.name == "_run_task")
    # Every BF16 pack in this specialized body consumes already activated
    # FP32 values, never raw gate/up accumulators from the dense donor.
    packs = [n for n in ast.walk(run) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name)
             and n.func.id == "pack_f32x2_to_bfloat2"]
    assert len(packs) == 2
    assert [[ast.unparse(a) for a in call.args] for call in packs] == [
        ["act0", "act1"], ["act2", "act3"],
    ]
    assert "if gate > cutlass.Float32(10.0):" in source
    assert "if up < cutlass.Float32(-10.0):" in source
    assert "if up > cutlass.Float32(10.0):" in source
    assert "while k64_slice < input_k64_tiles:" in source
    assert "for kb in cutlass.range_constexpr(2):" in source
    assert "scale_bytes[" in source
    assert "] = scale_word" not in source


def test_default_and_debug_capture_are_passive():
    source = (PATCH / "p8_native_kernel.py").read_text()
    tree = ast.parse(source)
    assert "fc1_tile_n: int = 128" in source
    assert "debug_capture: bool = False" in source
    debug = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                 and ast.unparse(n.test) == "self.debug_capture")
    assert not any(isinstance(n, ast.Call) for n in ast.walk(debug))
    assignment = debug.body[0]
    assert {key.value for key in assignment.value.keys} == {
        "packed_a", "scale_flat", "intermediate_u32", "route_output",
    }


def test_candidate_image_preserves_executed_fc2():
    dockerfile = (PATCH / "p8_smallm/Dockerfile").read_text()
    assert "FROM klc/glm53-p8-smallm:v1-src-7a0d202" in dockerfile
    assert "a0c398e9d412672d1138c69a5c0c3677bb29b7215379c701062d29b8b9b9265f" in dockerfile
    assert not any(line.startswith("COPY ") and "p8_small_m.py" in line
                   for line in dockerfile.splitlines())
