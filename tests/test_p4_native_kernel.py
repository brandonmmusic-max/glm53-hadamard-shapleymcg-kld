"""Synthetic CPU and offline CUDA compiler checks. Never launch a GPU."""
import ast
import ctypes
import hashlib
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pytest
import torch

from glm53_nvfp4.p4_reference import (
    decode_projection, mcg_code, moe_reference, synthetic_payload, tensor_sha256, tile_states,
)
from glm53_nvfp4.trellis_nvfp4 import (
    pack_trellis_edges, reconstruct_trellis_states,
)
from glm53_nvfp4.p4_codec import e2m1_codes, mcg_half_values
from runtime_patch.p4_native_kernel import MMA, SOURCE, compile_library, validate_payload


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def host(tmp_path_factory):
    path = tmp_path_factory.mktemp("p4-host") / "decode.so"
    subprocess.run(["c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                    "-shared", "-fPIC", str(ROOT / "tests/p4_host_bridge.cpp"), "-o", str(path)], check=True)
    lib = ctypes.CDLL(str(path))
    ptr, integer = ctypes.c_void_p, ctypes.c_int
    lib.host_codes.argtypes = [ptr]
    lib.host_states.argtypes = [ptr, ptr]
    lib.host_decode.argtypes = [ptr, ptr, integer, integer, integer]
    lib.host_fragment_coords.argtypes = [ptr, ptr, ptr]
    lib.host_scale_offset.argtypes = [integer] * 5
    lib.host_scale_offset.restype = ctypes.c_uint64
    return lib


def host_decode(host, stream):
    e, kt, nt, _ = stream.shape
    result = torch.empty((e, nt * 16, kt * 8), dtype=torch.uint8)
    host.host_decode(stream.data_ptr(), result.data_ptr(), e, nt * 16, kt * 16)
    return result


def test_exhaustive_mcg_law_matches_independent_half_oracle_and_frozen_codec(host):
    got = np.empty(65536, dtype=np.uint8)
    host.host_codes(got.ctypes.data)
    expected = np.array([mcg_code(state) for state in range(65536)], dtype=np.uint8)
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(got, e2m1_codes(mcg_half_values(np.arange(65536, dtype="<u2"))))
    assert (got == 8).sum() == 5432  # native signed-zero ABI, not legacy canonicalization
    digest = hashlib.sha256(got.tobytes()).hexdigest()
    assert digest == "195d9e9aac6dca94828fa8f693e9bcbba6804f8e160566ce02cad7ba742b7ba1"
    print("state_census_sha256=" + digest)


@pytest.mark.parametrize("seed", [0, 17, 20260904])
def test_sliding_states_cross_words_and_wrap_tiles(host, seed):
    gen = torch.Generator().manual_seed(seed)
    edges = torch.randint(0, 16, (4, 256), generator=gen)
    packed = pack_trellis_edges(edges, 4)
    expected = reconstruct_trellis_states(edges, 4).numpy().view(np.uint16)
    for tile in range(4):
        actual = np.empty(256, dtype=np.uint16)
        host.host_states(packed[tile].data_ptr(), actual.ctypes.data)
        np.testing.assert_array_equal(actual, expected[tile])
        assert actual.tolist() == tile_states(packed[tile].numpy().tobytes())


def test_every_k4_state_is_reachable_through_a_sliding_window(host):
    # Enumerate every four-symbol window without a 65536-tile allocation.
    # The separate random test covers all positions, including cyclic wraps.
    edges = (torch.arange(65536)[:, None] >> torch.tensor([12, 8, 4, 0])) & 15
    packed = pack_trellis_edges(edges.reshape(-1, 256), 4)
    actual = np.empty(256, dtype=np.uint16)
    for tile in range(1024):
        host.host_states(packed[tile].data_ptr(), actual.ctypes.data)
        np.testing.assert_array_equal(actual[3::4], np.arange(tile * 64, (tile + 1) * 64))


def test_projection_packing_matches_independent_reference(host):
    tensors, _ = synthetic_payload()
    digests = {
        "gate": "f89f698018331a433c47634d8445d9e0db2e2f5db44235d7c0607d1128bbf106",
        "up": "0e973abda7afbea5bf95de14bba201929fd5d45585074592cdb6d5a794b4b8df",
        "down": "5032821148e89ef5b325ec1eb9b61b5d2f522905521f2f3c4835530c5d8b2298",
    }
    for name, streams in (("gate", tensors["w13_trellis"][0]),
                          ("up", tensors["w13_trellis"][1]),
                          ("down", tensors["w2_trellis"])):
        actual = host_decode(host, streams)
        assert torch.equal(actual, decode_projection(streams))
        assert tensor_sha256(actual) == digests[name]
        print(f"{name}_packed_sha256={tensor_sha256(actual)}")


def test_scale_byte_addressing_not_ue8m0_or_k32(host):
    tensors, _ = synthetic_payload()
    for sf in (tensors["w13_scale_e4m3"][0], tensors["w13_scale_e4m3"][1], tensors["w2_scale_e4m3"]):
        e, n, k16 = sf.shape
        seen = set()
        for expert in range(e):
            for row in range(n):
                for k in range(k16 * 16):
                    offset = host.host_scale_offset(expert, row, k, n, k16 * 16)
                    assert int(sf.flatten()[offset]) == int(sf[expert, row, k // 16])
                    assert offset == (expert * n + row) * k16 + k // 16
                    seen.add(offset)
        assert seen == set(range(sf.numel()))
    # Catch truncation to 32-bit addresses without a huge allocation.
    assert host.host_scale_offset(287, 4095, 4095, 4096, 4096) == 301989887
    assert host.host_scale_offset(65535, 4095, 4095, 4096, 4096) == 68719476735


def test_m16n8k64_fragments_and_four_scale_bytes_cover_exact_logical_operands(host):
    a = np.empty((32, 32), dtype=np.int32)
    b = np.empty((32, 16), dtype=np.int32)
    sf = np.empty((32, 2), dtype=np.int32)
    host.host_fragment_coords(a.ctypes.data, b.ctypes.data, sf.ctypes.data)
    assert sorted(a.flatten()) == list(range(16 * 64))
    assert sorted(b.flatten()) == list(range(8 * 64))
    # Independently enumerate PTX fragment elements (ISA 8.8 sec 9.7.14.5.11).
    for lane in range(32):
        for i in range(32):
            row = lane // 4 + (8 if 8 <= i < 16 or i >= 24 else 0)
            col = lane % 4 * 8 + i % 8 + (32 if i >= 16 else 0)
            assert a[lane, i] == row * 64 + col
        for i in range(16):
            assert b[lane, i] == lane // 4 * 64 + lane % 4 * 8 + i % 8 + (32 if i >= 8 else 0)
    # With {0,0}, only the lower A lane pair and the first B lane contribute.
    # Tag every physical scale byte distinctly to catch row or K-half swaps.
    tags_a = np.arange(64).reshape(16, 4)
    tags_b = np.arange(32).reshape(8, 4) + 64
    for group in range(8):
        np.testing.assert_array_equal(tags_a[sf[4 * group, 0]], tags_a[group])
        np.testing.assert_array_equal(tags_a[sf[4 * group + 1, 0]], tags_a[group + 8])
        np.testing.assert_array_equal(tags_b[sf[4 * group, 1]], tags_b[group])


def test_payload_geometry_and_exact_rate():
    tensors, metadata = synthetic_payload()
    assert validate_payload(tensors, metadata, layer=3, tp_rank=0,
                            expected_design_sha256=metadata["source_design_sha256"]) == (3, 128, 64)
    weights = tensors["w13_trellis"].numel() * 4 + tensors["w2_trellis"].numel() * 4
    payload = sum(t.numel() * t.element_size() for k, t in tensors.items() if "global" not in k)
    globals_bytes = sum(t.numel() * t.element_size() for k, t in tensors.items() if "global" in k)
    assert 8 * payload / weights == 4.5
    assert globals_bytes == 3 * 3 * 4
    print(f"fixture_payload_bpw=4.5 fixture_with_global_scalars_bpw={8 * (payload + globals_bytes) / weights:.10f}")


@pytest.mark.parametrize("key,value", [("bits", "3"), ("alphabet", "e4m3"), ("scale", "ue8m0-k32"),
    ("compander", "2"), ("w13_order", "up,gate"), ("boundary", "h16"),
    ("weight_rounding", "nearest-ties-low-magnitude"), ("signed_zero", "canonical-positive"),
    ("scale_layout", "modelopt-128x4"), ("schema", "glm53-p4-mcg-tp-rank.v1"),
    ("ldlq", "true"), ("source_design_sha256", "0" * 64)])
def test_rejects_incompatible_metadata(key, value):
    tensors, metadata = synthetic_payload()
    design = metadata["source_design_sha256"]
    metadata[key] = value
    with pytest.raises(ValueError, match="metadata"):
        validate_payload(tensors, metadata, layer=3, tp_rank=0, expected_design_sha256=design)


@pytest.mark.parametrize("mutation", ["k32", "nan", "negative", "zero", "global", "extra", "dtype"])
def test_rejects_invalid_physical_planes(mutation):
    tensors, metadata = synthetic_payload()
    if mutation == "k32":
        tensors["w13_scale_e4m3"] = tensors["w13_scale_e4m3"][..., ::2].contiguous()
    elif mutation in ("nan", "negative"):
        tensors["w2_scale_e4m3"].flatten()[0] = 127 if mutation == "nan" else 128
    elif mutation == "zero":
        tensors["w2_scale_e4m3"].flatten()[0] = 0
    elif mutation == "global":
        tensors["w2_global_scale"][0] = float("nan")
    elif mutation == "extra":
        tensors["dense_weights"] = torch.empty(1)
    else:
        tensors["w2_trellis"] = tensors["w2_trellis"].float()
    with pytest.raises(ValueError):
        validate_payload(tensors, metadata, layer=3, tp_rank=0,
                         expected_design_sha256=metadata["source_design_sha256"])


def test_reference_output_hashes_are_deterministic():
    tensors, _ = synthetic_payload()
    x = (torch.arange(3 * 128).reshape(3, 128).float() % 29 - 14) / 32
    ids = torch.tensor([[2, 0], [1, 2], [0, 1]])
    weights = torch.tensor([[.25, .75], [.5, .5], [.75, .25]])
    # Fixed one-thread CPU arithmetic, not a five-run GPU determinism claim.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        hashes = [tensor_sha256(moe_reference(tensors, x, ids, weights)) for _ in range(5)]
    finally:
        torch.set_num_threads(previous)
    assert len(set(hashes)) == 1
    assert hashes[0] == "da59d0a934094f58453c1963d7ad67fda391d87b8171e5726a392b660c306ca9"
    print("cpu_moe_output_sha256_runs=" + repr(hashes))


@pytest.fixture(scope="module")
def offline_cuda(tmp_path_factory):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda-13.2/bin/nvcc"
    if not Path(nvcc).is_file():
        pytest.skip("offline CUDA compiler unavailable; device ISA is NOT verified")
    path = tmp_path_factory.mktemp("p4-offline")
    ptx, cubin = path / "p4.ptx", path / "p4.cubin"
    for mode, output in (("--ptx", ptx), ("--cubin", cubin)):
        command = [nvcc, "-std=c++17", "-O3", "-lineinfo", "-arch=sm_120a", mode,
                   "-Xptxas=-v", str(SOURCE), "-o", str(output)]
        result = subprocess.run(command, capture_output=True, text=True)
        print("offline_command=" + repr(command))
        print(result.stdout + result.stderr)
        assert result.returncode == 0
        if mode == "--cubin":
            spills = re.findall(r"(\d+) bytes spill stores, (\d+) bytes spill loads", result.stderr)
            assert spills and all(pair == ("0", "0") for pair in spills)
            registers = [int(value) for value in re.findall(r"Used (\d+) registers", result.stderr)]
            assert registers and max(registers) <= 64
            assert "1728 bytes smem" in result.stderr
    dump = subprocess.check_output([str(Path(nvcc).with_name("cuobjdump")), "--dump-sass", str(cubin)], text=True)
    return ptx.read_text(), dump, path


def test_offline_isa_is_native_nvfp4_and_contains_no_float_weight_mma(offline_cuda):
    ptx, sass, _ = offline_cuda
    operations = re.findall(r"mma\.sync[^;]+;", ptx)
    assert operations and all(op.startswith(MMA) for op in operations)
    assert all(re.search(r"\{0,0\}.*\{0,0\}", op) for op in operations)
    instructions = [line.strip() for line in sass.splitlines() if "MMA" in line]
    assert instructions and all("OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X" in line for line in instructions)
    assert "HMMA" not in sass
    assert "st.local" not in ptx and "ld.local" not in ptx
    print(f"assembled_mma_sites={len(instructions)} representative=" + instructions[0])
    print("offline_ptx_sha256=" + hashlib.sha256(ptx.encode()).hexdigest())


def test_offline_full_launch_library_links_without_loading_or_gpu(offline_cuda):
    _, _, directory = offline_cuda
    path = compile_library(directory / "library")
    assert path.is_file()
    symbols = subprocess.check_output(["nm", "-D", str(path)], text=True)
    for symbol in ("p4_project", "p4_quantize", "p4_swiglu", "p4_sum", "p4_decode_probe", "p4_prepare", "p4_capture_state"):
        assert re.search(r" T " + symbol + r"$", symbols, re.M)
    nvcc = shutil.which("nvcc") or "/usr/local/cuda-13.2/bin/nvcc"
    sass = subprocess.check_output([str(Path(nvcc).with_name("cuobjdump")), "--dump-sass", str(path)], text=True)
    instructions = [line for line in sass.splitlines() if "MMA" in line]
    assert instructions and all("OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X" in line for line in instructions)
    print(f"linked_library_mma_sites={len(instructions)}")


def test_cpu_probe_never_initializes_or_queries_cuda(monkeypatch):
    from glm53_nvfp4.probe_p4_native import cpu_probe

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU probe attempted CUDA access")

    for method in ("_lazy_init", "is_available", "get_device_capability", "current_device"):
        monkeypatch.setattr(torch.cuda, method, forbidden)
    result = cpu_probe()
    assert result["pass"] and not result["gpu_executed"]
    assert result["protected_roles_opened"] == []


def test_no_dense_weight_fallback_and_p8_sources_unchanged():
    source = SOURCE.read_text()
    runtime = (ROOT / "runtime_patch/p4_native_kernel.py").read_text()
    header = SOURCE.with_name("p4_decode.cuh").read_text()
    assert "kLookupBytes = 0" in header
    assert "__half" not in header and "__half" not in source
    assert "Stage stages[2]" in source and "rendezvous();" in source
    tree = ast.parse(runtime)
    for node in ast.walk(tree):
        assert not isinstance(node, ast.MatMult)
        if isinstance(node, ast.ImportFrom):
            assert node.module not in ("glm53_nvfp4.p4_reference", "p8_native_kernel")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in ("matmul", "mm", "bmm", "linear", "dequantize")
    # Regression pins cover the complete previous endpoint and its backend.
    originals = {
        "runtime_patch/p8_native_kernel.py": "a1f04de129e9e088137b869182fc03ea5e2268f5123ba8a5abbceeb0534b43b4",
        "runtime_patch/b12x_h16/b12x/moe/_shared/kernels/dynamic.py": "2428e8a6abaec75348181e6afee37deb2edf7697c820dd762029a453c8b071e4",
    }
    for file, digest in originals.items():
        assert hashlib.sha256((ROOT / file).read_bytes()).hexdigest() == digest
