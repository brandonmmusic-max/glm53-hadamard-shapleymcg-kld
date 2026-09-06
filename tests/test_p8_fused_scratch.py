"""CPU layout and unchanged-fallback checks for one-fill M1 scratch."""
import ast
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "runtime_patch"
spec = importlib.util.spec_from_file_location(
    "p8_scratch_schedule_test", PATCH / "p8_smallm_schedule.py"
)
schedule = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = schedule
spec.loader.exec_module(schedule)


def frozen_allocation_nodes():
    source = subprocess.check_output([
        "git", "show",
        "4569516b9b92d0645595576aaddfffb1e0fa0e9a:runtime_patch/p8_native_kernel.py",
    ], cwd=ROOT, text=True)
    tree = ast.parse(source)
    call = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "__call__")
    start = next(i for i, n in enumerate(call.body)
                 if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "packed_a")
    end = next(i for i, n in enumerate(call.body)
               if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "kernel_output")
    return call.body[start:end]


def test_all_23_regions_exactly_match_prior_allocations():
    class TorchShapes:
        uint8, int32, float32, bfloat16 = "uint8", "int32", "float32", "bfloat16"

        @staticmethod
        def zeros(*shape, dtype, device):
            return SimpleNamespace(shape=shape, dtype=dtype)

    context = {
        "torch": TorchShapes, "self": SimpleNamespace(
            hidden=4096, intermediate=512, experts=288, topk=8, device="cpu"
        ), "rows_padded": 128, "physical_tiles": 8, "max_tasks": 32, "m": 1,
        "tile_m": 16,
    }
    exec(compile(ast.Module(body=frozen_allocation_nodes(), type_ignores=[]),
                 "<frozen-scratch>", "exec"), context)
    layout = schedule.p8_small_m_scratch_layout()
    assert len(layout.regions) == 23
    assert len({region.name for region in layout.regions}) == 23
    for region in layout.regions:
        assert region.dtype == context[region.name].dtype
        assert region.shape == context[region.name].shape
    assert "route_output" not in {r.name for r in layout.regions}


def test_regions_are_aligned_disjoint_zeroed_and_retain_shapes():
    layout = schedule.p8_small_m_scratch_layout()
    arena = bytearray(layout.nbytes)
    cursor = 0
    formats = {"uint8": "B", "int32": "i", "float32": "f", "bfloat16": "H"}
    sizes = {"uint8": 1, "int32": 4, "float32": 4, "bfloat16": 2}
    views = []
    for ordinal, region in enumerate(layout.regions, 1):
        assert region.offset % 16 == 0
        assert cursor <= region.offset
        assert region.offset + region.nbytes <= layout.nbytes
        view = memoryview(arena)[region.offset:region.offset + region.nbytes]
        assert not any(view)
        typed = view.cast(formats[region.dtype])
        elements = 1
        for dim in region.shape:
            elements *= dim
        assert len(typed) == elements
        assert region.nbytes == elements * sizes[region.dtype]
        view[:] = bytes([ordinal]) * region.nbytes
        views.append((ordinal, view))
        cursor = region.offset + region.nbytes
    # Writing every following region must not change any earlier region.
    for ordinal, view in views:
        assert all(byte == ordinal for byte in view)
    assert layout.nbytes % 16 == 0


def test_opt_in_has_one_zero_fill_and_fallback_changes_only_scale_allocation():
    tree = ast.parse((PATCH / "p8_native_kernel.py").read_text())
    branch = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.If) and ast.unparse(n.test) == "fused_scratch_zero")
    fills = [n for stmt in branch.body for n in ast.walk(stmt)
             if isinstance(n, ast.Call) and ast.unparse(n.func) == "torch.zeros"]
    assert len(fills) == 1
    current = branch.orelse
    frozen = frozen_allocation_nodes()
    current_other = [
        node for node in current
        if not (
            isinstance(node, ast.Assign)
            and any(
                ast.unparse(target) in {"scale_elements", "scale_flat"}
                for target in node.targets
            )
        )
    ]
    frozen_other = [
        node for node in frozen
        if not (
            isinstance(node, ast.Assign)
            and any(ast.unparse(target) == "scale_flat" for target in node.targets)
        )
    ]
    assert ast.dump(ast.Module(body=current_other, type_ignores=[])) == ast.dump(
        ast.Module(body=frozen_other, type_ignores=[])
    )
    source = (PATCH / "p8_native_kernel.py").read_text()
    assert "if self.full_coupled and materialized and not small_m" in source
    assert "fuse_scratch_zero: bool = False" in source
    assert "fused_scratch_zero = self.fuse_scratch_zero and small_m" in source
    assert '"fused_scratch_zero": fused_scratch_zero' in source
