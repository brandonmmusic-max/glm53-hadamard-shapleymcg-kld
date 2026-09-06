from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DYNAMIC = (
    ROOT
    / "runtime_patch/b12x_h16/b12x/moe/_shared/kernels/dynamic.py"
)


def _function_source(name: str) -> str:
    source = DYNAMIC.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"missing function {name}")


def test_input_trace_is_opt_in_and_uses_no_new_launch_operand() -> None:
    source = DYNAMIC.read_text()
    assert "self.p8_input_prequant_diagnostic = False" in source
    call_source = _function_source("__call__")
    assert "p8_input_prequant_diagnostic" not in ast.unparse(
        ast.parse(call_source).body[0].args
    )
    kernel_source = _function_source("kernel")
    assert "if cutlass.const_expr(self.p8_input_prequant_diagnostic):" in kernel_source
    assert "assert trellis_lut is not None" in kernel_source


def test_input_trace_contract_is_exactly_two_k32_blocks_and_512_bytes() -> None:
    kernel_source = _function_source("kernel")
    assert "m1_blk_idx == Int32(40)" in kernel_source
    assert "m1_blk_idx == Int32(62)" in kernel_source
    assert "trace_base = Int32(64)" in kernel_source
    assert "cutlass.range_constexpr(32)" in kernel_source
    tree = ast.parse(kernel_source)
    assert any(
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Mult)
        and ast.unparse(node.left) == "m1_values[trace_elem]"
        and ast.unparse(node.right) == "trace_inv_scale"
        for node in ast.walk(tree)
    )
    # 2 blocks * (32 raw + 32 normalized) * sizeof(float32).
    assert 2 * (32 + 32) * 4 == 512


def test_trace_occurs_after_payload_quantization_and_before_wire_store() -> None:
    kernel_source = _function_source("kernel")
    quant = kernel_source.index("m1_payload, m1_scale_byte = quantize_block_fp8_mx(")
    trace = kernel_source.index(
        "if cutlass.const_expr(self.p8_input_prequant_diagnostic):"
    )
    wire = kernel_source.index("m1_block_start = m1_blk_idx * Int32(32)", trace)
    assert quant < trace < wire
