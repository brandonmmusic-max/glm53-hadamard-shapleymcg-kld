"""Experimental P4 TP-local MoE: cyclic K4 MCG -> E2M1/E4M3/16 -> mxf4nvf4.

GLM serving selects this endpoint through the B12X P4 backend adapter.
Importing this module neither compiles nor launches. Device closure remains
a separate gate from the host/static serving integration.
Port provenance: ExLlamaV3 law/stream, KQuant/QSRT codec/lane conventions,
and B12X w4a8_trellis producer/consumer design; see THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

import ctypes
from contextlib import contextmanager
import fcntl
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading

import torch


SOURCE = Path(__file__).resolve().parent / "p4" / "p4_moe.cu"
MMA = "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3"
_LIBRARIES = {}
# A graph stores raw addresses for these external ctypes launches. The adapter
# has no graph-destruction callback, so captured resources intentionally remain
# live until worker exit. Eager-only owners can be reclaimed with their model.
_CAPTURE_PINS = {}


class P4WorkspaceOwner:
    """One model namespace/TP rank; sequential layers may share its scratch."""

    def __init__(self):
        self.workspaces = {}
        self._dispatch_lock = threading.Lock()

    @contextmanager
    def dispatch(self):
        # Torch and ctypes can release the GIL mid-enqueue. A shared stream
        # orders GPU work, but does not prevent two host calls interleaving.
        if not self._dispatch_lock.acquire(blocking=False):
            raise RuntimeError("P4 concurrent/reentrant host dispatch is unsupported; DBO must be disabled")
        try:
            yield
        finally:
            self._dispatch_lock.release()


def max_route_tiles(routes: int, experts: int) -> int:
    """Tight worst-case number of nonempty M16 tiles for arbitrary routing."""
    if type(routes) is not int or type(experts) is not int or routes < 0 or experts <= 0:
        raise ValueError("P4 route/expert counts must be nonnegative/positive integers")
    return min(routes, (routes + 15 * experts) // 16)


def _verify_file(path: Path, expected_sha256: str, expected_bytes: int | None) -> None:
    if (len(expected_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_sha256)):
        raise ValueError("P4 file hash must be 64 lowercase hexadecimal characters")
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as handle:
        for data in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(data)
            size += len(data)
    if digest.hexdigest() != expected_sha256 or (expected_bytes is not None and size != expected_bytes):
        raise ValueError(f"P4 sidecar content identity mismatch: {path}")


def validate_payload(tensors: dict[str, torch.Tensor], metadata: dict[str, str], *,
                     layer: int, tp_rank: int, expected_design_sha256: str) -> tuple[int, int, int]:
    """Validate the complete physical ABI on CPU before any device allocation."""
    required = {
        "schema": "glm53-p4-mcg-tp-rank.v2", "role": "physical-codec",
        "layer": str(layer), "rank": str(tp_rank), "world_size": "4", "bits": "4",
        "alphabet": "e2m1", "scale": "e4m3-k16", "law": "procedural-mcg-alpha1-rne-e2m1",
        "compander": "1", "weight_rounding": "nearest-even-satfinite",
        "signed_zero": "preserve", "scale_layout": "row-major-n-k16",
        "state_bits": "16", "tile_values": "256", "state_boundary": "cyclic-per-tile",
        "mcg_arithmetic": "u32-wrap-mask-xor-add-rn-f16", "byte_order": "little",
        "trellis_layout": "k16-n16-exl3-lane-pair-swapped-i16",
        "global_scale": "positive-f32-per-projection-expert",
        "expert_order": "global-contiguous-zero-based", "tp_sharding": "gate-up-rows-down-columns",
        "state_lut_bytes": "0",
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
    dimensions = {"experts": str(experts), "hidden": str(hidden),
                  "intermediate": str(intermediate * 4), "intermediate_per_rank": str(intermediate)}
    if any(metadata.get(name) != value for name, value in dimensions.items()):
        raise ValueError("P4 metadata dimensions do not match physical TP4 tensors")
    if experts <= 0 or hidden <= 0 or intermediate <= 0 or hidden % 64 or intermediate % 64:
        raise ValueError("P4 K dimensions must be positive multiples of 64")
    if w2.dtype != torch.int16 or tuple(w2.shape) != (experts, n16, k16, 64):
        raise ValueError("invalid P4 W2 K4 stream")
    for name, shape in (("w13_scale_e4m3", (2, experts, intermediate, hidden // 16)),
                        ("w2_scale_e4m3", (experts, hidden, intermediate // 16))):
        sf = tensors[name]
        if (sf.dtype != torch.uint8 or tuple(sf.shape) != shape
                or bool(((sf == 0) | (sf > 126)).any())):
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

    Stable route sorting uses model/shape/stream/capture-keyed activation scratch.
    CUDA graph parity and throughput remain unqualified. Serving calls
    prepare() during weight loading, before vLLM may capture CUDA graphs.
    """

    def __init__(self, sidecar: Path, *, device: torch.device, tp_rank: int, layer: int,
                 expected_design_sha256: str, topk: int = 8, swiglu_limit: float = 10.0,
                 build_dir: Path | None = None, expected_file_sha256: str | None = None,
                 expected_file_bytes: int | None = None,
                 expected_geometry: tuple[int, int, int] | None = None,
                 workspace_owner: P4WorkspaceOwner | None = None):
        from safetensors import safe_open

        if expected_file_sha256 is not None:
            _verify_file(sidecar, expected_file_sha256, expected_file_bytes)
        with safe_open(sidecar, framework="pt", device="cpu") as src:
            tensors = {key: src.get_tensor(key) for key in src.keys()}
            metadata = src.metadata() or {}
        if expected_file_sha256 is not None:
            _verify_file(sidecar, expected_file_sha256, expected_file_bytes)
        self._initialize(tensors, metadata, device=device, tp_rank=tp_rank, layer=layer,
                         expected_design_sha256=expected_design_sha256, topk=topk,
                         swiglu_limit=swiglu_limit, build_dir=build_dir,
                         expected_geometry=expected_geometry, workspace_owner=workspace_owner)

    @classmethod
    def from_tensors(cls, tensors, metadata, **kwargs):
        """Explicit synthetic/device-closure seam with the same payload gates."""
        obj = cls.__new__(cls)
        obj._initialize(tensors, metadata, **kwargs)
        return obj

    def _initialize(self, tensors, metadata, *, device, tp_rank, layer,
                    expected_design_sha256, topk=8, swiglu_limit=10.0, build_dir=None,
                    expected_geometry=None, workspace_owner=None):
        self.experts, self.hidden, self.intermediate = validate_payload(
            tensors, metadata, layer=layer, tp_rank=tp_rank,
            expected_design_sha256=expected_design_sha256)
        if (expected_geometry is not None
                and (self.experts, self.hidden, self.intermediate) != expected_geometry):
            raise ValueError("P4 sidecar geometry does not match the serving layer")
        if not isinstance(topk, int) or isinstance(topk, bool) or not 1 <= topk <= self.experts:
            raise ValueError("P4 topk must be an integer in 1..experts")
        if not math.isfinite(swiglu_limit) or swiglu_limit <= 0:
            raise ValueError("P4 SwiGLU limit must be positive and finite")
        if workspace_owner is not None and not isinstance(workspace_owner, P4WorkspaceOwner):
            raise TypeError("P4 workspace requires an explicit P4WorkspaceOwner")
        self.workspace_owner = workspace_owner if workspace_owner is not None else self.new_workspace_owner()
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
        self.prepared = False

    @staticmethod
    def new_workspace_owner():
        return P4WorkspaceOwner()

    def compile(self):
        if self.lib is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("P4 must compile and prepare during weight loading, before graph capture")
            key = (str(self.build_dir.resolve()), hashlib.sha256(
                SOURCE.read_bytes() + SOURCE.with_name("p4_decode.cuh").read_bytes()).hexdigest())
            if key in _LIBRARIES:
                self.lib = _LIBRARIES[key]
                return self.lib
            self.lib = ctypes.CDLL(str(compile_library(self.build_dir)))
            ptr, integer = ctypes.c_void_p, ctypes.c_int
            self.lib.p4_project.argtypes = [ptr] * 9 + [integer] * 6 + [ptr]
            self.lib.p4_quantize.argtypes = [ptr] * 3 + [integer] * 3 + [ptr]
            self.lib.p4_swiglu.argtypes = [ptr, ptr, integer, integer, ctypes.c_float, ptr]
            self.lib.p4_sum.argtypes = [ptr] * 3 + [integer] * 3 + [ptr]
            self.lib.p4_decode_probe.argtypes = [ptr, ptr] + [integer] * 3 + [ptr]
            self.lib.p4_prepare.argtypes = []
            self.lib.p4_capture_state.argtypes = [ptr, ctypes.POINTER(integer), ctypes.POINTER(ctypes.c_ulonglong)]
            for name in ("p4_project", "p4_quantize", "p4_swiglu", "p4_sum", "p4_decode_probe", "p4_prepare",
                         "p4_capture_state"):
                getattr(self.lib, name).restype = integer
            _LIBRARIES[key] = self.lib
        return self.lib

    def prepare(self):
        """Compile/load all CUDA functions outside graph capture; no kernel launch."""
        with torch.cuda.device(self.device):
            if self.prepared:
                return
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("P4 prepare must finish before CUDA graph capture")
            self._check(self.compile().p4_prepare())
            self.prepared = True

    @staticmethod
    def _check(status):
        if status:
            raise RuntimeError(f"P4 CUDA launch failed with status {status}")

    def capture_id(self, stream):
        """None for eager, otherwise CUDA's unique process-lifetime sequence ID."""
        active, sequence = ctypes.c_int(-1), ctypes.c_ulonglong()
        self._check(self.lib.p4_capture_state(stream, ctypes.byref(active), ctypes.byref(sequence)))
        if active.value not in (0, 1):
            raise RuntimeError("P4 invalid CUDA capture-state response")
        # Do not assume a valid capture ID is nonzero.
        return sequence.value if active.value else None

    def _pin_capture(self, capture_id, *inputs):
        if capture_id is not None:
            pins = _CAPTURE_PINS.setdefault((self.workspace_owner, capture_id), {"runtimes": {}, "inputs": {}})
            pins["runtimes"][id(self)] = self  # sidecars, compiled module and owner scratch
            for tensor in inputs:
                pins["inputs"][id(tensor)] = tensor

    def workspace(self, rows, stream, capture_id=None):
        """Activation/routing scratch only; never decoded weight matrices.

        Only layers with the same explicit model owner may reuse storage.
        Eager and distinct captures use disjoint storage even on one stream;
        different streams/shapes also remain disjoint. Captured resources
        are pinned until worker exit. No tensor here is a decoded weight.
        Returned model outputs never alias this pool.
        """
        key = (str(self.device), rows, int(stream), capture_id, self.experts, self.hidden,
               self.intermediate, self.topk)
        workspaces = self.workspace_owner.workspaces
        if key not in workspaces:
            routes = rows * self.topk
            def empty(shape, dtype):
                return torch.empty(shape, dtype=dtype, device=self.device)
            work = {name: empty((routes,), torch.int64) for name in ("ids", "sorted_ids", "order")}
            work.update({
                "ones": torch.ones(routes, dtype=torch.int64, device=self.device),
                "counts": empty((self.experts,), torch.int64),
                "tile_counts": empty((self.experts,), torch.int64),
                "offsets": torch.zeros(self.experts + 1, dtype=torch.int64, device=self.device),
                "tiles": torch.zeros(self.experts + 1, dtype=torch.int64, device=self.device),
                "a1": empty((rows, self.hidden // 8), torch.int32),
                "sfa1": empty((rows, self.hidden // 16), torch.uint8),
                "a2": empty((routes, self.intermediate // 8), torch.int32),
                "sfa2": empty((routes, self.intermediate // 16), torch.uint8),
                "gu": empty((routes, 2, self.intermediate), torch.float32),
                "mid": empty((routes, self.intermediate), torch.float32),
                "routed": empty((routes, self.hidden), torch.float32),
                "output": empty((rows, self.hidden), torch.float32),
            })
            workspaces[key] = work
        self._pin_capture(capture_id)
        return workspaces[key]

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
        if max_route_tiles(routes, self.experts) > 65535:
            raise ValueError("P4 route tile count exceeds the initial launch grid")
        with self.workspace_owner.dispatch(), torch.cuda.device(self.device):
            self.prepare()
            lib = self.lib
            current_stream = torch.cuda.current_stream(self.device)
            cuda_stream = current_stream.cuda_stream
            capture_id = self.capture_id(cuda_stream)
            work = self.workspace(m, cuda_stream, capture_id)
            ids = work["ids"]
            ids.view(m, self.topk).copy_(topk_ids)
            torch._assert_async(((ids >= 0) & (ids < self.experts)).all(), "P4 expert ID outside payload")
            torch._assert_async(torch.isfinite(x).all(), "P4 input is non-finite")
            torch._assert_async(torch.isfinite(topk_weights).all(), "P4 route weight is non-finite")
            order = work["order"]
            torch.sort(ids, stable=True, out=(work["sorted_ids"], order))
            # Fixed-size scatter histogram avoids bincount's data-dependent
            # result extent and rejects invalid IDs before index arithmetic.
            counts = work["counts"].zero_()
            counts.scatter_add_(0, ids, work["ones"])
            offsets, tiles = work["offsets"], work["tiles"]
            torch.cumsum(counts, 0, out=offsets[1:])
            torch.add(counts, 15, out=work["tile_counts"])
            torch.div(work["tile_counts"], 16, rounding_mode="floor", out=work["tile_counts"])
            torch.cumsum(work["tile_counts"], 0, out=tiles[1:])
            activation = x.contiguous()
            weights = topk_weights.contiguous()
            self._pin_capture(capture_id, activation, weights)
            gu, mid, routed, output = (work[name] for name in ("gu", "mid", "routed", "output"))
            t = self.tensors

            def project(inp, sf, name, out, n, k, projections, divisor):
                args = (inp, sf, t[name + "_trellis"], t[name + "_scale_e4m3"],
                        t[name + "_global_scale"], order, offsets, tiles, out)
                self._check(lib.p4_project(*[v.data_ptr() for v in args], self.experts,
                                          routes, n, k, projections, divisor, cuda_stream))

            self._check(lib.p4_quantize(activation.data_ptr(), work["a1"].data_ptr(),
                         work["sfa1"].data_ptr(), m, self.hidden, 1, cuda_stream))
            project(work["a1"], work["sfa1"], "w13", gu, self.intermediate, self.hidden, 2, self.topk)
            self._check(lib.p4_swiglu(gu.data_ptr(), mid.data_ptr(), routes,
                                     self.intermediate, self.limit, cuda_stream))
            self._check(lib.p4_quantize(mid.data_ptr(), work["a2"].data_ptr(),
                         work["sfa2"].data_ptr(), routes, self.intermediate, 0, cuda_stream))
            project(work["a2"], work["sfa2"], "w2", routed, self.hidden, self.intermediate, 1, 1)
            self._check(lib.p4_sum(routed.data_ptr(), weights.data_ptr(), output.data_ptr(),
                                  m, self.topk, self.hidden, cuda_stream))
            # ctypes launches are external to Torch's allocator. Record all
            # referenced tensors on the actual stream, including sidecars.
            for v in (*t.values(), *work.values(), activation, weights):
                v.record_stream(current_stream)
            result = output.to(torch.bfloat16)
            return (result, gu.clone(), mid.clone(), routed.clone()) if return_intermediates else result
