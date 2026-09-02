"""Block-diagonal 16x16 orthogonal transforms and channel permutations for NVFP4 grouping.

W is [out, in].  A block rotation acts on the INPUT dim:  W' = W @ blockdiag(R),  H' = R^T H R (blockwise),
and the runtime applies R^T to activations (y = W x = (W R)(R^T x)).  For pseudo-quant scoring the
effective weight is  W_eff = Q(W R) R^T.

Rotation kinds: identity (control), fixed Hadamard-16, seeded random SO(16), learned Cayley
R = R0 * (I - A)(I + A)^-1 with A skew-symmetric (120 params), R0 in {I, Had16}.
"""
from __future__ import annotations

import torch


def hadamard16() -> torch.Tensor:
    h = torch.tensor([[1.0]])
    while h.shape[0] < 16:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / 4.0  # orthonormal (1/sqrt(16))


def random_so16(seed: int, n: int = 16) -> torch.Tensor:
    g = torch.Generator().manual_seed(int(seed))
    a = torch.randn(n, n, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.to(torch.float32)


def cayley(a_skew: torch.Tensor, r0: torch.Tensor | None = None) -> torch.Tensor:
    """R = R0 (I - A)(I + A)^-1 for a skew-symmetric A (any square matrix is skewed here)."""
    a = a_skew - a_skew.transpose(-1, -2)
    a = a / 2
    n = a.shape[-1]
    eye = torch.eye(n, device=a.device, dtype=a.dtype)
    r = torch.linalg.solve((eye + a).transpose(-1, -2), (eye - a).transpose(-1, -2)).transpose(-1, -2)
    if r0 is not None:
        r = r0.to(r.dtype) @ r
    return r


def apply_block_rotation(w: torch.Tensor, r: torch.Tensor, group: int = 16) -> torch.Tensor:
    """W [..., K] -> W @ blockdiag(R).  r: [g, g] shared across blocks, or [K/g, g, g] per block."""
    *lead, k = w.shape
    wb = w.reshape(*lead, k // group, group).to(torch.float32)
    if r.dim() == 2:
        out = wb @ r.to(wb.dtype)
    else:
        out = torch.einsum("...bg,bgh->...bh", wb, r.to(wb.dtype))
    return out.reshape(*lead, k).to(w.dtype)


def rotate_hessian(h: torch.Tensor, r: torch.Tensor, group: int = 16) -> torch.Tensor:
    """H [K, K] -> blockdiag(R)^T H blockdiag(R)."""
    k = h.shape[-1]
    if r.dim() == 2:
        big = torch.block_diag(*([r.to(h.dtype)] * (k // group)))
    else:
        big = torch.block_diag(*[r[i].to(h.dtype) for i in range(k // group)])
    return big.transpose(-1, -2) @ h @ big


def block_diag_matrix(r: torch.Tensor, k: int, group: int = 16) -> torch.Tensor:
    if r.dim() == 2:
        return torch.block_diag(*([r] * (k // group)))
    return torch.block_diag(*[r[i] for i in range(k // group)])


def hessian_permutation(h_diag: torch.Tensor, h: torch.Tensor | None, group: int = 16, mode: str = "diag_band") -> torch.Tensor:
    """Return a permutation of input channels so that each consecutive 16-block groups similar channels.

    mode 'diag_band': sort channels by Hessian diagonal (energy); consecutive groups then share dynamic
    range (what absmax/E4M3 block scaling wants).  mode 'corr_cluster': greedy grouping by |correlation|
    from the Hessian (memory gain).  Returns perm (LongTensor) with W[:, perm] as the new order.
    """
    k = h_diag.shape[0]
    if mode == "diag_band" or h is None:
        return torch.argsort(h_diag, descending=True)
    d = torch.sqrt(h_diag.clamp(min=1e-12))
    c = (h / (d[:, None] * d[None, :])).abs().clone()
    c.fill_diagonal_(0)
    remaining = torch.ones(k, dtype=torch.bool, device=h.device)
    order = []
    while remaining.any():
        seed = int(torch.nonzero(remaining)[0])
        remaining[seed] = False
        grp = [seed]
        while len(grp) < group and remaining.any():
            score = c[grp].sum(0)
            score[~remaining] = -1
            j = int(score.argmax())
            remaining[j] = False
            grp.append(j)
        order.extend(grp)
    return torch.tensor(order, device=h.device)


def inverse_permutation(perm: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=perm.device)
    return inv
