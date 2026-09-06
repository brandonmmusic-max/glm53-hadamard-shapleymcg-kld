"""CPU/static binding checks against the immutable active parent-image ABI."""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "runtime_patch/b12x_h16/b12x/moe/_shared/kernels"

PARENT_IMAGE = "sha256:0336113e0fff876cccf9e6ac5347528ae59f4ad894a0ce7cb4c4e90b4651a745"
PARENT_PHASE1_SHA256 = "7222df68bb6ad7ec5ca69cf7c4ffd5c3d6b1ddc240e6375c4b1e77bcc151909c"
PARENT_PHASE2_SHA256 = "be317f7f76153ff5f60d2d1cb76f15dae14e6252a109b2d448e7dd8852f66194"

PHASE1_ARGS = [
    "packed_a_storage",
    "scale_storage",
    "w13_rp",
    "w13_sfb_rp",
    "intermediate_u32",
    "token_map",
    "task_expert",
    "task_valid_rows",
    "expert_tile_base",
    "alpha",
    "input_global_scale",
    "trellis_lut",
    "scale_component",
    "input_k128_tiles",
    "intermediate_tiles",
    "packed_w13_tiles",
    "max_active_clusters",
    "stream",
]
PHASE2_ARGS = [
    "intermediate_u32",
    "down_rp",
    "down_sfb_rp",
    "scatter_output",
    "token_map",
    "token_weights",
    "task_expert",
    "task_valid_rows",
    "expert_tile_base",
    "down_alpha",
    "global_scale",
    "trellis_lut",
    "scale_component",
    "intermediate_tiles",
    "packed_output_tiles",
    "max_active_clusters",
    "stream",
]


def _class_method(path: Path, class_name: str, method: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method
    )


def _launch(method: ast.FunctionDef) -> ast.Call:
    return next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "launch"
    )


def _kernel_call(launch: ast.Call) -> ast.Call:
    assert isinstance(launch.func, ast.Attribute)
    assert isinstance(launch.func.value, ast.Call)
    return launch.func.value


def test_phase2_explicitly_extends_parent_16_arg_abi_with_scale() -> None:
    path = KERNELS / "p8_small_m.py"
    method = _class_method(path, "P8SmallMPhase2Kernel", "__call__")
    assert [arg.arg for arg in method.args.args] == ["self", *PHASE2_ARGS]
    launch = _launch(method)
    kernel = _kernel_call(launch)
    assert [ast.unparse(arg) for arg in kernel.args] == PHASE2_ARGS[:-2]
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in launch.keywords}
    assert keywords["grid"] == "(1, 1, max_active_clusters * Int32(2))"
    assert keywords["min_blocks_per_mp"] == "2"
    assert keywords["stream"] == "stream"
    source = path.read_text()
    assert PARENT_PHASE2_SHA256 in source


def test_m64_fc1_overrides_parent_two_cta_launch_with_one_cta() -> None:
    path = KERNELS / "p8_coupled_prefill_fc1.py"
    method = _class_method(path, "P8CoupledPrefillFC1Kernel", "__call__")
    assert [arg.arg for arg in method.args.args] == ["self", *PHASE1_ARGS]
    launch = _launch(method)
    kernel = _kernel_call(launch)
    assert [ast.unparse(arg) for arg in kernel.args][1:] == PHASE1_ARGS[1:-2]
    assert ast.unparse(kernel.args[0]) == (
        "cute.recast_tensor(packed_a_storage, cutlass.Uint32)"
    )
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in launch.keywords}
    assert keywords["grid"] == "(1, 1, max_active_clusters)"
    assert keywords["min_blocks_per_mp"] == "1"
    assert keywords["stream"] == "stream"
    inherited = (KERNELS / "p8_h128_fc1.py").read_text()
    assert PARENT_PHASE1_SHA256 in inherited


def test_dynamic_call_sites_bind_the_explicit_scale_abis() -> None:
    source = (KERNELS / "dynamic.py").read_text()
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr
        in {"materialized_phase1_kernel", "materialized_phase2_kernel"}
    ]
    phase1 = next(call for call in calls if len(call.args) == len(PHASE1_ARGS))
    phase2 = next(call for call in calls if len(call.args) == len(PHASE2_ARGS))
    assert [ast.unparse(arg) for arg in phase1.args][11:14] == [
        "trellis_lut",
        "trellis_rotations",
        "Int32(a_input.shape[1]) // Int32(128)",
    ]
    assert [ast.unparse(arg) for arg in phase2.args][11:14] == [
        "trellis_lut",
        "trellis_rotations",
        "gate_tile_cnt",
    ]


def test_parent_identity_and_stale_donor_boundary_are_explicit() -> None:
    assert PARENT_IMAGE.endswith("1a745")
    report = (ROOT / "results/P8_FULL_COUPLED_PREFILL_CPU_STATIC.md").read_text()
    assert "build/lib" not in "\n".join(
        line for line in report.splitlines() if line.startswith("p8_")
    )
