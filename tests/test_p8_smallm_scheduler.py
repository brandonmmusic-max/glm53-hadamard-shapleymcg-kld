"""Structural tests only: no CUDA imports, role data, or device execution."""
import ast
import importlib.util
from pathlib import Path
import struct
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "runtime_patch"
KERNELS = PATCH / "b12x_h16/b12x/moe/_shared/kernels"
spec = importlib.util.spec_from_file_location("p8_smallm_schedule", PATCH / "p8_smallm_schedule.py")
schedule = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = schedule
spec.loader.exec_module(schedule)


def test_geometry_covers_each_projection_once():
    geometry = schedule.P8SmallMGeometry()
    assert (geometry.physical_tiles, geometry.fc1_tasks, geometry.fc2_tasks) == (8, 32, 128)
    assert {geometry.fc1_owner(i) for i in range(32)} == {
        (route, part) for route in range(8) for part in range(4)
    }
    assert {geometry.fc2_owner(i) for i in range(128)} == {
        (route, tile) for route in range(8) for tile in range(16)
    }
    # Each N256 owner writes two N128 halves. Every real output has one owner.
    owners = [(route, tile * 256 + half * 128 + column)
              for route, tile in map(geometry.fc2_owner, range(128))
              for half in range(2) for column in range(128)]
    assert len(owners) == len(set(owners)) == 8 * 4096


@pytest.mark.parametrize("tokens", [0, 2, 3, 16, 2048])
def test_only_m1_selects_candidate(tokens):
    assert not schedule.use_small_m(True, tokens)
    with pytest.raises(ValueError):
        schedule.P8SmallMGeometry(tokens=tokens)


def test_opt_in_and_shape_rejection():
    assert schedule.use_small_m(True, 1)
    assert not schedule.use_small_m(False, 1)
    for field, value in [("experts", 256), ("topk", 6), ("intermediate", 1024)]:
        with pytest.raises(ValueError):
            schedule.P8SmallMGeometry(**{field: value})
    for slot in [-1, 32]:
        with pytest.raises(ValueError):
            schedule.P8SmallMGeometry().fc1_owner(slot)
    for slot in [-1, 128]:
        with pytest.raises(ValueError):
            schedule.P8SmallMGeometry().fc2_owner(slot)


def test_sources_compile_without_importing_cuda():
    for path in [PATCH / "p8_native_kernel.py", KERNELS / "dynamic.py", KERNELS / "p8_small_m.py"]:
        compile(path.read_text(), str(path), "exec")


def test_p8_fc2_has_no_e2m1_or_atomic_math():
    source = (KERNELS / "p8_small_m.py").read_text()
    tree = ast.parse(source)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "mxfp8_mma_m16n8k32_f32_e4m3" in names
    assert "w4a8_trellis_pair_words_dispatch" in names
    assert not any("e2m1" in name or "scatter_add" in name for name in names)
    assert "self.trellis_bits = 4" in source
    assert "self.trellis_codebook = \"mcg\"" in source
    assert "self.trellis_scaled = True" in source
    assert "self.deterministic_output = True" in source


def test_p8_fc2_stages_one_a_tile_for_each_n256_pair():
    source = (KERNELS / "p8_small_m.py").read_text()
    tree = ast.parse(source)
    kernel_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "P8SmallMPhase2Kernel"
    )
    methods = {
        node.name: node
        for node in kernel_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    def called(method, name):
        return [
            node
            for node in ast.walk(methods[method])
            if isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Name) and node.func.id == name)
                or (isinstance(node.func, ast.Attribute) and node.func.attr == name)
            )
        ]

    # The route/output-pair kernel owns N256 directly; there is no outer
    # two-half loop and no second invocation that can reload the same A slice.
    assert len(called("kernel", "_run_task")) == 1
    assert "for half in cutlass.range_constexpr(2)" not in source
    assert "output_pair * Int32(256)" in source

    # One pair stage copies A once, then stages the two compressed N128 stream
    # halves and the complete physical N256 scale tile.
    assert len(called("_stage_pair_slice", "cp_async_u32_shared_global")) == 1
    assert len(called("_stage_pair_slice", "_w4a8_stage_trellis_b_tile")) == 2
    assert "b_stage_bytes = 2 * b_half_bytes" in source
    assert "sfb_stage_bytes = 2 * sfb_half_bytes" in source
    assert "range_constexpr(8)" in source

    # Four warps own disjoint N64 spans, covering one N256 output pair and its
    # sixteen N16 trellis blocks exactly once.
    columns = {
        warp * 64 + c * 2 + nt * 8 + element
        for warp in range(4)
        for c in range(4)
        for nt in range(8)
        for element in range(2)
    }
    assert columns == set(range(256))
    n16_blocks = {
        warp * 4 + half_block
        for warp in range(4)
        for half_block in range(4)
    }
    assert n16_blocks == set(range(16))
    assert {(n16 >> 3, n16 & 7) for n16 in n16_blocks} == {
        (half, local) for half in range(2) for local in range(8)
    }


def test_dynamic_calls_route_ids_and_bypasses_old_fc2():
    source = (KERNELS / "dynamic.py").read_text()
    assert "self.external_materialized_fc2 = self.w4a8_split_materialized or self.p8_small_m" in source
    assert "self.materialize_intermediate and not self.external_materialized_fc2" in source
    assert "phase2_experts = topk_ids" in source
    assert "self.materialized_phase2_kernel = P8SmallMPhase2Kernel()" in source
    assert 'CuTe exposes these tensor extents as staged values' in source
    wrapper = (PATCH / "p8_native_kernel.py").read_text()
    assert "small_m_scheduler: bool = False" in wrapper
    assert "p8_small_m=small_m" in wrapper
    assert "_launch_dynamic_topk_sum(" in wrapper
    assert '("small_m_scheduler", int(small_m))' in wrapper
    probe = (ROOT / "glm53_nvfp4/probe_p8_native_tp_runtime.py").read_text()
    assert '"--small-m-scheduler"' in probe
    assert "None if args.small_m_scheduler" in probe
    assert 'runtime_kwargs["small_m_scheduler"] = True' in probe
    assert '"--timing-repeats"' in probe
    assert 'torch.cuda.Event(enable_timing=True)' in probe
    assert 'statistics.median(samples)' in probe
    assert 'torch.cuda.CUDAGraph()' in probe
    assert '"matches_eager_output"' in probe
    assert 'torch.Generator(device="cuda").manual_seed(args.seed)' in probe
    assert 'default="monolithic"' in probe
    assert '"resolved_mode"' in probe
    sitecustomize = (PATCH / "sitecustomize.py").read_text()
    assert 'GLM53_P8_SMALL_M' in sitecustomize
    assert 'small_m_scheduler=_P8N_SMALL_M' in sitecustomize


def _bf16(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def test_rounding_boundary_matches_monolithic_serving_baseline():
    slices = [1.0, 2**-8, 2**-8, 0.0]
    value = 0.0
    for partial in slices:
        value = _bf16(value + _bf16(partial))
    assert value == 1.0
    assert _bf16(sum(slices)) == 1.0078125
    source = (KERNELS / "p8_small_m.py").read_text()
    assert "down_scale * facc[0][nt][element]" in source
    assert "weighted = weight * partial" in source
    assert "cutlass.BFloat16(previous + weighted)" in source
