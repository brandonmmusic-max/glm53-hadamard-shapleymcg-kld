"""Experimental P4 TP-local MoE: cyclic K4 MCG -> E2M1/E4M3/16 -> mxf4nvf4.

Independent endpoint; importing this module neither compiles nor launches.
CUDA compilation/device closure and serving integration are separate gates.
Port provenance: ExLlamaV3 law/stream, KQuant/QSRT codec/lane conventions,
and B12X w4a8_trellis producer/consumer design; see THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

import ctypes
import fcntl
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import torch


SOURCE = Path(__file__).resolve().parent / "p4" / "p4_moe.cu"
MMA = "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3"


def validate_payload(tensors: dict[str, torch.Tensor], metadata: dict[str, str], *,
                     layer: int, tp_rank: int, expected_design_sha256: str) -> tuple[int, int, int]:
    """Validate the complete physical ABI on CPU before any device allocation."""
    required = {
        "schema": "glm53-p4-mcg-tp-rank.v1", "role": "physical-codec",
        "layer": str(layer), "rank": str(tp_rank), "world_size": "4", "bits": "4",
        "alphabet": "e2m1", "scale": "e4m3-k16", "law": "procedural-mcg",
        "compander": "1", "weight_rounding": "nearest-ties-low-magnitude",
        "boundary": "identity", "w13_order": "gate,up", "ldlq": "false",
        "source_design_sha256": expected_design_sha256,
    }
    if not 3 <= layer <= 44 or not 0 <= tp_rank < 4:
        raise ValueError("P4 requires GLM layer 3..44 and TP4 rank 0..3")
    if len(expected_design_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_design_sha256):
        raise ValueError("P4 requires an immutable source design hash")
    if any(metadata.get(k) != v for k, v in required.items()):
        raise ValueError("P4 sidecar metadata does not match the frozen physical ABI")
    names = {"w13_trellis", "w2_trellis", "w13_scale_e4m3", "w2_scale_e4m3",
             "w13_global_scale", "w2_global_scale"}
    if set(tensors) != names:
        raise ValueError("P4 sidecar must contain exactly the six physical tensors")
    for value in tensors.values():
        if value.device.type != "cpu" or not value.is_contiguous():
            raise ValueError("P4 payload validation requires contiguous CPU tensors")
    w13, w2 = tensors["w13_trellis"], tensors["w2_trellis"]
    if w13.dtype != torch.int16 or w13.ndim != 5 or w13.shape[0] != 2 or w13.shape[-1] != 64:
        raise ValueError("invalid P4 W13 K4 stream")
    _, experts, k16, n16, _ = w13.shape
    hidden, intermediate = k16 * 16, n16 * 16
    if experts <= 0 or hidden <= 0 or intermediate <= 0 or hidden % 64 or intermediate % 64:
        raise ValueError("P4 K dimensions must be positive multiples of 64")
    if w2.dtype != torch.int16 or tuple(w2.shape) != (experts, n16, k16, 64):
        raise ValueError("invalid P4 W2 K4 stream")
    for name, shape in (("w13_scale_e4m3", (2, experts, intermediate, hidden // 16)),
                        ("w2_scale_e4m3", (experts, hidden, intermediate // 16))):
        sf = tensors[name]
        if sf.dtype != torch.uint8 or tuple(sf.shape) != shape or bool((sf > 126).any()):
            raise ValueError(f"invalid P4 E4M3/16 scale plane: {name}")
    for name, shape in (("w13_global_scale", (2, experts)), ("w2_global_scale", (experts,))):
        gs = tensors[name]
        if (gs.dtype != torch.float32 or tuple(gs.shape) != shape
                or not bool(torch.isfinite(gs).all()) or not bool((gs > 0).all())):
            raise ValueError(f"invalid P4 global scale: {name}")
    return experts, hidden, intermediate


def compile_library(build_dir: Path, *, nvcc: str | None = None) -> Path:
    """Compile SM120a without querying or running a GPU (also the probe seam)."""
    compiler = nvcc or shutil.which("nvcc") or "/usr/local/cuda-13.2/bin/nvcc"
    flags = ["-std=c++17", "-O3", "-lineinfo", "-gencode=arch=compute_120a,code=sm_120a",
             "--shared", "-Xcompiler=-fPIC"]
    identity = hashlib.sha256(SOURCE.read_bytes() + SOURCE.with_name("p4_decode.cuh").read_bytes()
                              + repr(flags).encode()
                              + subprocess.check_output([compiler, "--version"])).hexdigest()[:20]
    target = Path(build_dir) / identity / "p4_moe.so"
    target.parent.mkdir(parents=True, exist_ok=True)
    with (target.parent / "compile.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            fd, temporary = tempfile.mkstemp(dir=target.parent, suffix=".so")
            os.close(fd)
            try:
                subprocess.run([compiler, *flags, str(SOURCE), "-o", temporary], check=True)
                os.replace(temporary, target)
            finally:
                Path(temporary).unlink(missing_ok=True)
    return target


class P4NativeTPMoE:
    """Two fused trellis projections with FP32 activation scratch and fixed-order top-k sum.

    This initial launch seam uses stable route sorting and per-call scratch.
    CUDA graph parity and throughput remain unqualified. It does not modify
    sitecustomize, production loaders, P8 dispatch, or any service.
    """

    def __init__(self, sidecar: Path, *, device: torch.device, tp_rank: int, layer: int,
                 expected_design_sha256: str, topk: int = 8, swiglu_limit: float = 10.0,
                 build_dir: Path | None = None):
        from safetensors import safe_open

        with safe_open(sidecar, framework="pt", device="cpu") as src:
            tensors = {key: src.get_tensor(key) for key in src.keys()}
            metadata = src.metadata() or {}
        self._initialize(tensors, metadata, device=device, tp_rank=tp_rank, layer=layer,
                         expected_design_sha256=expected_design_sha256, topk=topk,
                         swiglu_limit=swiglu_limit, build_dir=build_dir)

    @classmethod
    def from_tensors(cls, tensors, metadata, **kwargs):
        """Explicit synthetic/device-closure seam with the same payload gates."""
        obj = cls.__new__(cls)
        obj._initialize(tensors, metadata, **kwargs)
        return obj

    def _initialize(self, tensors, metadata, *, device, tp_rank, layer,
                    expected_design_sha256, topk=8, swiglu_limit=10.0, build_dir=None):
        self.experts, self.hidden, self.intermediate = validate_payload(
            tensors, metadata, layer=layer, tp_rank=tp_rank,
            expected_design_sha256=expected_design_sha256)
        if not isinstance(topk, int) or isinstance(topk, bool) or not 1 <= topk <= self.experts:
            raise ValueError("P4 topk must be an integer in 1..experts")
        if not math.isfinite(swiglu_limit) or swiglu_limit <= 0:
            raise ValueError("P4 SwiGLU limit must be positive and finite")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("P4 runtime requires CUDA; use p4_reference for CPU validation")
        if torch.cuda.get_device_capability(self.device) != (12, 0):
            raise ValueError("P4 runtime currently targets SM120 only")
        self.topk, self.limit = topk, float(swiglu_limit)
        self.device = torch.device("cuda", self.device.index if self.device.index is not None else torch.cuda.current_device())
        self.tensors = {name: tensor.to(self.device) for name, tensor in tensors.items()}
        self.build_dir = build_dir or SOURCE.parents[2] / "build" / "p4-native"
        self.lib = None

    def compile(self):
        if self.lib is None:
            self.lib = ctypes.CDLL(str(compile_library(self.build_dir)))
            ptr, integer = ctypes.c_void_p, ctypes.c_int
            self.lib.p4_project.argtypes = [ptr] * 8 + [integer] * 6 + [ptr]
            self.lib.p4_swiglu.argtypes = [ptr, ptr, integer, integer, ctypes.c_float, ptr]
            self.lib.p4_sum.argtypes = [ptr] * 3 + [integer] * 3 + [ptr]
            self.lib.p4_decode_probe.argtypes = [ptr, ptr] + [integer] * 3 + [ptr]
            for name in ("p4_project", "p4_swiglu", "p4_sum", "p4_decode_probe"):
                getattr(self.lib, name).restype = integer
        return self.lib

    @staticmethod
    def _check(status):
        if status:
            raise RuntimeError(f"P4 CUDA launch failed with status {status}")

    @torch.inference_mode()
    def __call__(self, x, topk_weights, topk_ids, *, return_intermediates=False):
        if x.device != self.device or x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] != self.hidden:
            raise ValueError("P4 expects device-local [M,hidden] BF16 activations")
        m = x.shape[0]
        for value in (topk_weights, topk_ids):
            if value.device != self.device or tuple(value.shape) != (m, self.topk):
                raise ValueError("P4 routing device/shape mismatch")
        if topk_ids.dtype not in (torch.int32, torch.int64) or topk_weights.dtype != torch.float32:
            raise ValueError("P4 routes require integer IDs and FP32 weights")
        if not m:
            result = x.clone()
            if return_intermediates:
                return (result, torch.empty((0, 2, self.intermediate), dtype=torch.float32, device=self.device),
                        torch.empty((0, self.intermediate), dtype=torch.float32, device=self.device),
                        torch.empty((0, self.hidden), dtype=torch.float32, device=self.device))
            return result
        routes = m * self.topk
        if (routes + 15) // 16 + self.experts > 65535:
            raise ValueError("P4 route tile count exceeds the initial launch grid")
        with torch.cuda.device(self.device):
            lib = self.compile()
            ids = topk_ids.contiguous().view(-1).long()
            torch._assert_async(((ids >= 0) & (ids < self.experts)).all(), "P4 expert ID outside payload")
            torch._assert_async(torch.isfinite(x).all(), "P4 input is non-finite")
            torch._assert_async(torch.isfinite(topk_weights).all(), "P4 route weight is non-finite")
            order = torch.argsort(ids, stable=True)
            # Fixed-size scatter histogram avoids bincount's data-dependent
            # result extent and rejects invalid IDs before index arithmetic.
            counts = torch.zeros(self.experts, dtype=torch.int64, device=self.device)
            counts.scatter_add_(0, ids, torch.ones_like(ids))
            offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
            tiles = torch.cat((counts.new_zeros(1), ((counts + 15) // 16).cumsum(0)))
            activation = x.float().contiguous()
            weights = topk_weights.contiguous()
            gu = torch.empty((routes, 2, self.intermediate), dtype=torch.float32, device=self.device)
            mid = torch.empty((routes, self.intermediate), dtype=torch.float32, device=self.device)
            routed = torch.empty((routes, self.hidden), dtype=torch.float32, device=self.device)
            output = torch.empty((m, self.hidden), dtype=torch.float32, device=self.device)
            cuda_stream = torch.cuda.current_stream(self.device).cuda_stream
            t = self.tensors

            def project(inp, name, out, n, k, projections, divisor):
                args = (inp, t[name + "_trellis"], t[name + "_scale_e4m3"],
                        t[name + "_global_scale"], order, offsets, tiles, out)
                self._check(lib.p4_project(*[v.data_ptr() for v in args], self.experts,
                                          routes, n, k, projections, divisor, cuda_stream))

            project(activation, "w13", gu, self.intermediate, self.hidden, 2, self.topk)
            self._check(lib.p4_swiglu(gu.data_ptr(), mid.data_ptr(), routes,
                                     self.intermediate, self.limit, cuda_stream))
            project(mid, "w2", routed, self.hidden, self.intermediate, 1, 1)
            self._check(lib.p4_sum(routed.data_ptr(), weights.data_ptr(), output.data_ptr(),
                                  m, self.topk, self.hidden, cuda_stream))
            # ctypes launches are external to Torch's allocator. Record all
            # referenced tensors on the actual stream, including sidecars.
            for v in (*t.values(), activation, weights, order, offsets, tiles, gu, mid, routed, output):
                v.record_stream(torch.cuda.current_stream(self.device))
            result = output.to(torch.bfloat16)
            return (result, gu, mid, routed) if return_intermediates else result
