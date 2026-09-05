"""CPU copy-address and unchanged-arithmetic gates for cropped FC1 staging."""
import ast
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
RELATIVE = "runtime_patch/b12x_h16/b12x/moe/_shared/kernels/p8_narrow_fc1.py"
FROZEN = "f8ee1caeb99325f365374e9f968d4e9f96256d88"


@pytest.mark.parametrize("width", [32, 64])
def test_owned_b_copies_partition_original_and_cover_decoder_windows(width):
    # K4: each K16/N16 window is 32 u32; every async copy transfers four.
    full_chunks, k16_rows, global_stride = 64, 4, 1024
    all_copies = []
    for subtile in range(128 // width):
        owned_chunks = width // 2
        copies = []
        for idx in range(k16_rows * owned_chunks):
            row, local_chunk = divmod(idx, owned_chunks)
            chunk = subtile * owned_chunks + local_chunk
            copies.append((row * global_stride + 4 * chunk,
                           16 * (row * full_chunks + chunk)))
        all_copies.extend(copies)
        staged_words = {dst // 4 + word for _, dst in copies for word in range(4)}
        decoder_words = {
            row * 256 + n16 * 32 + word
            for row in range(4)
            for n16 in range(subtile * (width // 16), (subtile + 1) * (width // 16))
            for word in range(32)
        }
        assert staged_words == decoder_words
    original = {(row * global_stride + 4 * chunk, 16 * (row * 64 + chunk))
                for row in range(4) for chunk in range(64)}
    assert len(all_copies) == len(set(all_copies)) == len(original)
    assert set(all_copies) == original


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("packed_half", [0, 1])
def test_owned_sfb_vectors_preserve_every_consumed_k128_word(width, packed_half):
    all_copies = []
    for subtile in range(128 // width):
        copies = [
            (packed_half * 128 + 4 * vector, 16 * vector)
            for vector in range(subtile * (width // 4), (subtile + 1) * (width // 4))
        ]
        all_copies.extend(copies)
        staged_words = {dst // 4 + word for _, dst in copies for word in range(4)}
        # Each logical warp owns four N8 fragments; q indexes eight scale words.
        consumer_words = {
            (logical_warp * 4 + nt) * 8 + q
            for logical_warp in range(subtile * (width // 32),
                                      (subtile + 1) * (width // 32))
            for nt in range(4) for q in range(8)
        }
        assert staged_words == consumer_words
    assert len(all_copies) == len(set(all_copies)) == 32
    assert set(all_copies) == {(packed_half * 128 + 4 * vector, 16 * vector)
                              for vector in range(32)}


def test_compute_and_grid_match_frozen_exact_candidate():
    old = ast.parse(subprocess.check_output(
        ["git", "show", f"{FROZEN}:{RELATIVE}"], cwd=ROOT, text=True
    ))
    new = ast.parse((ROOT / RELATIVE).read_text())
    compile(new, RELATIVE, "exec")

    def method(tree, name):
        return next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == name)

    for name in ["__init__", "_activated_value", "kernel"]:
        assert ast.dump(method(new, name)) == ast.dump(method(old, name))

    # The only allowed compute-body delta is forwarding subtile to staging.
    current_run = method(new, "_run_task")
    changed_calls = 0
    for node in ast.walk(current_run):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "_stage_slice":
                assert isinstance(node.args[-1], ast.Name)
                assert node.args[-1].id == "subtile"
                node.args.pop()
                changed_calls += 1
    assert changed_calls == 2
    assert ast.dump(current_run) == ast.dump(method(old, "_run_task"))
