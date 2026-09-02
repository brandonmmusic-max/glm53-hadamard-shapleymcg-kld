"""Native-Blackwell format ladder as pseudo-quantizers (PyTorch, GPU).

Tiers (all element/scale conventions as documented in the BMXFP4 v2 plan):
  T1  NVFP4 : E2M1 elements, UE4M3 block scale per 16 (along the input dim), per-tensor FP32 scale.
  T0  2:4-sparse NVFP4 : same element grid, 2 of every 4 kept; metadata 2 bit per stored nonzero.
  T2  MXFP6 : E2M3 (or E3M2) elements, UE8M0 block scale per 32.
  T3  FP8   : E4M3 elements, per-output-channel FP32 scale.
  T4  BF16  : passthrough.

Every quantizer returns the DEQUANTIZED weight (same shape/dtype as input, in the caller's basis)
plus exact payload-byte accounting.  Nothing here is an idealized grid: the E4M3/E8M0 rounding of
scales and the per-tensor scale are inside the forward, and a straight-through estimator is used
for the roundings so the same functions can sit inside rotation learning.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

E2M1_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M3_LEVELS = None  # built lazily
E4M3_MAX = 448.0
E4M3_MIN_NORMAL = 2.0 ** -6  # 0.015625


# ----------------------------------------------------------------------------- helpers
def _ste_round(x: torch.Tensor) -> torch.Tensor:
    """Round with straight-through gradient."""
    return x + (torch.round(x) - x).detach()


def _round_to_levels(x: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Round |x| to the nearest level (levels sorted ascending, non-negative) keeping sign; STE."""
    lv = levels.to(x.device, x.dtype)
    ax = x.abs()
    # nearest level via midpoints (levels small, so broadcasting is cheap)
    mids = (lv[1:] + lv[:-1]) / 2
    idx = torch.bucketize(ax, mids)  # 0..len(lv)-1
    q = lv[idx] * torch.sign(x)
    return x + (q - x).detach()


def _e4m3_round(s: torch.Tensor) -> torch.Tensor:
    """Round a positive FP32 scale to E4M3 (unsigned, 3 mantissa bits, max 448, subnormals to 2^-9)."""
    s = s.clamp(min=0.0, max=E4M3_MAX)
    s8 = s.to(torch.float8_e4m3fn).to(torch.float32)
    return s + (s8 - s).detach()


def _e8m0_round(s: torch.Tensor) -> torch.Tensor:
    """Round a positive scale to a power of two (UE8M0), rounding the log2 to nearest."""
    e = torch.log2(s.clamp(min=2.0 ** -126))
    e = e + (torch.round(e) - e).detach()
    return torch.exp2(e)


def _blocks(w: torch.Tensor, g: int) -> torch.Tensor:
    """View [..., K] as [..., K/g, g] (K must be divisible by g)."""
    *lead, k = w.shape
    assert k % g == 0, f"input dim {k} not divisible by group {g}"
    return w.reshape(*lead, k // g, g)


# ----------------------------------------------------------------------------- NVFP4 (T1)
@dataclass
class NVFP4Config:
    group: int = 16
    scale_search: bool = True        # MSE-optimal block scale search (vs plain absmax/6)
    search_grid: int = 12            # candidate shrink factors between 0.65 and 1.0 (plus 4-vs-6 max)
    range_max_search: bool = True    # try mapping block max to 6 or to 4 (MR-GPTQ / 4over6 style)
    tensor_scale_iters: int = 2      # alternating per-tensor / per-block scale refinement


def nvfp4_quantize(w: torch.Tensor, cfg: NVFP4Config = NVFP4Config(), return_parts: bool = False):
    """Fake-quantize W [..., K] with NVFP4 along the last dim.  Returns dequantized W (same dtype).

    Convention (ModelOpt-style): per-tensor scale  S = amax(W) / (6 * 448)   (so block scales fit E4M3)
    block scale  s_b = amax(block) / 6 / S  -> E4M3 ;  element q = round(w / (s_b * S)) on the E2M1 grid.
    """
    orig_dtype = w.dtype
    if w.numel() == 0:
        return (w, {"tensor_scale": None, "block_scale": None}) if return_parts else w
    x = w.to(torch.float32)
    g = cfg.group
    xb = _blocks(x, g)                                         # [..., B, g]
    amax_t = x.abs().amax().clamp(min=1e-12)
    S = amax_t / (6.0 * E4M3_MAX)                              # per-tensor FP32 scale
    bmax = xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)  # [..., B, 1]

    def quant_with(scale_b):  # scale_b: [..., B, 1] real block scale (already E4M3-rounded * S)
        q = _round_to_levels(xb / scale_b, E2M1_LEVELS)
        return q * scale_b

    def block_scale(range_max, shrink):
        raw = bmax / range_max * shrink / S
        return _e4m3_round(raw) * S

    best_err = None
    best_scale = None
    candidates = []
    range_maxes = [6.0, 4.0] if cfg.range_max_search else [6.0]
    shrinks = torch.linspace(0.65, 1.0, cfg.search_grid).tolist() if cfg.scale_search else [1.0]
    for rm in range_maxes:
        for sh in shrinks:
            candidates.append((rm, sh))
    for rm, sh in candidates:
        sb = block_scale(rm, sh)
        deq = quant_with(sb)
        err = ((deq - xb) ** 2).sum(dim=-1, keepdim=True)     # per block
        if best_err is None:
            best_err, best_scale = err, sb
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_scale = torch.where(better, sb, best_scale)
    # alternating refinement of the per-tensor scale against the chosen block grid
    for _ in range(cfg.tensor_scale_iters if cfg.scale_search else 0):
        q = _round_to_levels(xb / best_scale, E2M1_LEVELS)       # integer-grid codes (times 1)
        # least-squares refit of a global multiplier gamma: min || xb - gamma * q*scale ||^2
        num = (xb * q * best_scale).sum()
        den = ((q * best_scale) ** 2).sum().clamp(min=1e-20)
        gamma = (num / den).clamp(0.8, 1.25)
        best_scale = best_scale * gamma
    deq = quant_with(best_scale).reshape_as(x)
    out = deq.to(orig_dtype)
    if return_parts:
        return out, {"tensor_scale": S, "block_scale": best_scale.squeeze(-1)}
    return out


def nvfp4_payload_bytes(shape, group: int = 16) -> int:
    """Exact storage: 4 bits/element + 8-bit E4M3 scale per group + 4-byte per-tensor scale."""
    n = math.prod(shape)
    return n // 2 + (n // group) + 4


# ----------------------------------------------------------------------------- 2:4 sparse NVFP4 (T0)
def sparse24_mask(w: torch.Tensor, importance: torch.Tensor | None = None) -> torch.Tensor:
    """Keep the 2 largest-magnitude (or highest-importance) elements of every 4 along the last dim."""
    x = (w if importance is None else importance).to(torch.float32)
    x4 = x.reshape(*x.shape[:-1], x.shape[-1] // 4, 4)
    idx = x4.abs().topk(2, dim=-1).indices
    mask = torch.zeros_like(x4, dtype=torch.bool).scatter_(-1, idx, True)
    return mask.reshape_as(w)


def sparse48_mask(w: torch.Tensor, importance: torch.Tensor | None = None, pairs: bool = True) -> torch.Tensor:
    """4:8 structured sparsity along the last dim: keep 4 of every 8 elements.

    pairs=True (FP4 hardware pattern: elements are packed two per byte, so the kept elements come as two
    adjacent pairs out of the four pairs in each chunk of 8, selected by pair importance).
    pairs=False keeps any 4 of 8 (upper bound, not storable with 2-bit-per-pair metadata).
    """
    x = (w if importance is None else importance).to(torch.float32)
    k = x.shape[-1]
    assert k % 8 == 0
    if pairs:
        xp = x.reshape(*x.shape[:-1], k // 8, 4, 2).abs().sum(-1)          # pair importance [.., K/8, 4]
        idx = xp.topk(2, dim=-1).indices
        mp = torch.zeros_like(xp, dtype=torch.bool).scatter_(-1, idx, True)   # [.., K/8, 4]
        mask = mp.unsqueeze(-1).expand(*mp.shape, 2).reshape(*x.shape[:-1], k)
    else:
        x8 = x.reshape(*x.shape[:-1], k // 8, 8).abs()
        idx = x8.topk(4, dim=-1).indices
        mask = torch.zeros_like(x8, dtype=torch.bool).scatter_(-1, idx, True).reshape_as(x)
    return mask.reshape(w.shape)


SPARSE_PATTERN = "4:8"   # campaign default (Brandon 2026-09-02): 4:8 pair-structured for FP4


def sparse_nvfp4_payload_bytes(shape, group: int = 16, scale_per_logical: bool = True, pattern: str = SPARSE_PATTERN) -> int:
    """Sparse NVFP4 payload bytes.

    2:4 : stored nonzeros n/2 at 4 bit + 2-bit metadata per stored nonzero (1 bit/element) + scales.
    4:8 : stored nonzeros n/2 at 4 bit + pair-index metadata: 2 bits per kept pair, 2 pairs per 8
          elements = 4 bits per 8 elements (0.5 bit/element) + scales.
    scale_per_logical=True: one UE4M3 per 16 LOGICAL elements; False: per 16 STORED elements.
    """
    n = math.prod(shape)
    stored = n // 2
    scales = (n // group) if scale_per_logical else (stored // group)
    meta_bits = n if pattern == "2:4" else n // 2
    return stored // 2 + meta_bits // 8 + scales + 4


# ----------------------------------------------------------------------------- MXFP6 (T2)
def _fp_levels(exp_bits: int, man_bits: int) -> torch.Tensor:
    """Positive representable values of a small float with the given exponent/mantissa bits (with subnormals),
    bias = 2^(e-1) - 1, max = largest finite."""
    bias = 2 ** (exp_bits - 1) - 1
    vals = {0.0}
    for e in range(0, 2 ** exp_bits):
        for m in range(0, 2 ** man_bits):
            if e == 0:
                v = (m / 2 ** man_bits) * 2.0 ** (1 - bias)
            else:
                v = (1 + m / 2 ** man_bits) * 2.0 ** (e - bias)
            vals.add(v)
    return torch.tensor(sorted(vals))


_LEVEL_CACHE: dict[tuple[int, int], torch.Tensor] = {}


def fp_levels(exp_bits: int, man_bits: int) -> torch.Tensor:
    key = (exp_bits, man_bits)
    if key not in _LEVEL_CACHE:
        _LEVEL_CACHE[key] = _fp_levels(exp_bits, man_bits)
    return _LEVEL_CACHE[key]


@dataclass
class MXFP6Config:
    group: int = 32
    variant: str = "e2m3"            # or "e3m2"
    scale_search: bool = True         # exponent fitting (MR-GPTQ App. H style: try a few exponent offsets)


def mxfp6_quantize(w: torch.Tensor, cfg: MXFP6Config = MXFP6Config()) -> torch.Tensor:
    orig_dtype = w.dtype
    x = w.to(torch.float32)
    levels = fp_levels(2, 3) if cfg.variant == "e2m3" else fp_levels(3, 2)
    lmax = float(levels[-1])
    xb = _blocks(x, cfg.group)
    bmax = xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    base = _e8m0_round(bmax / lmax)
    best_err, best_scale = None, None
    offsets = [1.0, 0.5, 2.0] if cfg.scale_search else [1.0]
    for off in offsets:
        sb = base * off
        deq = _round_to_levels(xb / sb, levels) * sb
        err = ((deq - xb) ** 2).sum(dim=-1, keepdim=True)
        if best_err is None:
            best_err, best_scale = err, sb
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_scale = torch.where(better, sb, best_scale)
    deq = (_round_to_levels(xb / best_scale, levels) * best_scale).reshape_as(x)
    return deq.to(orig_dtype)


def mxfp6_payload_bytes(shape, group: int = 32) -> int:
    n = math.prod(shape)
    return (n * 6) // 8 + (n // group)


# ----------------------------------------------------------------------------- FP8 (T3)
def fp8_quantize(w: torch.Tensor) -> torch.Tensor:
    """E4M3 per-output-channel (row) scaling; W is [out, in]."""
    orig_dtype = w.dtype
    x = w.to(torch.float32)
    rmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    s = rmax / E4M3_MAX
    q = (x / s).to(torch.float8_e4m3fn).to(torch.float32)
    deq = x + (q * s - x).detach()
    return deq.to(orig_dtype)


def fp8_payload_bytes(shape) -> int:
    n = math.prod(shape)
    return n + 4 * shape[0]


def bf16_payload_bytes(shape) -> int:
    return 2 * math.prod(shape)


# ----------------------------------------------------------------------------- state / requant API (for GPTQ)
def nvfp4_state(w: torch.Tensor, cfg: NVFP4Config = NVFP4Config()) -> torch.Tensor:
    """Compute the real per-block scale [..., B, 1] (E4M3-rounded * tensor scale) once for a slab."""
    _, parts = nvfp4_quantize(w, cfg, return_parts=True)
    return parts["block_scale"].unsqueeze(-1)


def nvfp4_requant(w: torch.Tensor, scale_b: torch.Tensor, group: int = 16) -> torch.Tensor:
    """Round a slab [..., K] with FIXED block scales (from nvfp4_state); cheap, used inside GPTQ loops."""
    xb = _blocks(w.to(torch.float32), group)
    q = _round_to_levels(xb / scale_b, E2M1_LEVELS) * scale_b
    return q.reshape_as(w).to(w.dtype)


def mxfp6_state(w: torch.Tensor, cfg: MXFP6Config = MXFP6Config()) -> tuple[torch.Tensor, torch.Tensor]:
    levels = fp_levels(2, 3) if cfg.variant == "e2m3" else fp_levels(3, 2)
    lmax = float(levels[-1])
    xb = _blocks(w.to(torch.float32), cfg.group)
    bmax = xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    base = _e8m0_round(bmax / lmax)
    best_err, best_scale = None, None
    for off in ([1.0, 0.5, 2.0] if cfg.scale_search else [1.0]):
        sb = base * off
        err = ((_round_to_levels(xb / sb, levels) * sb - xb) ** 2).sum(dim=-1, keepdim=True)
        if best_err is None:
            best_err, best_scale = err, sb
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err); best_scale = torch.where(better, sb, best_scale)
    return best_scale, levels


def mxfp6_requant(w: torch.Tensor, state, group: int = 32) -> torch.Tensor:
    scale_b, levels = state
    xb = _blocks(w.to(torch.float32), group)
    return (_round_to_levels(xb / scale_b, levels) * scale_b).reshape_as(w).to(w.dtype)


class GroupQuantizer:
    """Callable pair for GPTQ: state(slab)->s ; requant(slab, s)->deq.  Slab is [..., group]."""

    def __init__(self, tier: str, nv: NVFP4Config | None = None, mx: MXFP6Config | None = None):
        self.tier = tier
        self.nv = nv or NVFP4Config()
        self.mx = mx or MXFP6Config()
        self.group = 32 if tier == "T2_mxfp6" else 16

    def state(self, slab: torch.Tensor):
        if self.tier in ("T1_nvfp4", "T0_sparse_nvfp4"):
            return nvfp4_state(slab, self.nv)
        if self.tier == "T2_mxfp6":
            return mxfp6_state(slab, self.mx)
        raise KeyError(self.tier)

    def requant(self, slab: torch.Tensor, state) -> torch.Tensor:
        if self.tier in ("T1_nvfp4", "T0_sparse_nvfp4"):
            return nvfp4_requant(slab, state, self.group)
        if self.tier == "T2_mxfp6":
            return mxfp6_requant(slab, state, self.group)
        raise KeyError(self.tier)


# ----------------------------------------------------------------------------- tier registry
TIERS = ("T0_sparse_nvfp4", "T1_nvfp4", "T2_mxfp6", "T3_fp8", "T4_bf16")


def tier_payload_bytes(tier: str, shape) -> int:
    if tier == "T0_sparse_nvfp4":
        return sparse_nvfp4_payload_bytes(shape)
    if tier == "T1_nvfp4":
        return nvfp4_payload_bytes(shape)
    if tier == "T2_mxfp6":
        return mxfp6_payload_bytes(shape)
    if tier == "T3_fp8":
        return fp8_payload_bytes(shape)
    if tier == "T4_bf16":
        return bf16_payload_bytes(shape)
    raise KeyError(tier)


def tier_bpw(tier: str, shape=(768, 2048)) -> float:
    return 8.0 * tier_payload_bytes(tier, shape) / math.prod(shape)


def make_quantizer(tier: str, nv: NVFP4Config | None = None, mx: MXFP6Config | None = None):
    """Return f(W[..., K]) -> dequantized W for a tier (T0 handled by GPTQ/SparseGPT driver, here mask+quant)."""
    nv = nv or NVFP4Config()
    mx = mx or MXFP6Config()
    if tier == "T1_nvfp4":
        return lambda w: nvfp4_quantize(w, nv)
    if tier == "T2_mxfp6":
        return lambda w: mxfp6_quantize(w, mx)
    if tier == "T3_fp8":
        return fp8_quantize
    if tier == "T4_bf16":
        return lambda w: w
    if tier == "T0_sparse_nvfp4":
        def f(w):
            m = sparse48_mask(w) if SPARSE_PATTERN == "4:8" else sparse24_mask(w)
            return nvfp4_quantize(w * m, nv) * m
        return f
    raise KeyError(tier)


if __name__ == "__main__":  # smoke test
    torch.manual_seed(0)
    w = torch.randn(256, 2048, device="cuda") * 0.02
    for t in TIERS:
        q = make_quantizer(t)(w)
        rel = ((q - w).norm() / w.norm()).item()
        print(f"{t:16s} bpw={tier_bpw(t):5.3f} relL2={rel:.5f}")
