"""Batched GPTQ / SparseGPT with a group quantizer whose block scale is searched once per group.

W [E, N, K] (E experts, N output rows, K input columns) with per-expert Hessians H [E, K, K].

Static act-order (MR-GPTQ ingredient 2): inside every fixed group of `group` columns, columns are
re-ordered by descending Hessian diagonal (batch-mean over experts so the batch stays batched); the
whole matrix and Hessian are permuted accordingly, standard sequential GPTQ runs, and the result is
un-permuted.  Group membership (and hence the storage layout and block scales) never changes.

Inside a group the block scale is searched ONCE on the current (error-updated) group values, then
columns are quantized one at a time with that fixed scale, feeding the error forward (OBQ / GPTQ).

2:4 sparsity (SparseGPT-style): the mask is decided on the ORIGINAL column layout (groups of 4)
from the current values at group start, using saliency w^2 / [H^-1]_ii^2, then carried through the
permutation; pruned weights are compensated by the error feedback like any other rounding error.

quantizer: object with .state(slab)->s and .requant(slab, s)->deq   (see nvfp4.GroupQuantizer)
"""
from __future__ import annotations

import torch


def hessian_inverse_chol(h: torch.Tensor, percdamp: float = 0.01) -> torch.Tensor:
    """Upper Cholesky factor of (H + damp I)^-1, batched [E, K, K]."""
    e, k, _ = h.shape
    h = h.to(torch.float32).clone()
    idx = torch.arange(k, device=h.device)
    diag = h[:, idx, idx]
    dead = diag <= 0
    h[:, idx, idx] = torch.where(dead, torch.ones_like(diag), diag)
    damp = percdamp * h[:, idx, idx].mean(dim=-1, keepdim=True)
    h[:, idx, idx] += damp
    l = torch.linalg.cholesky(h)
    hinv = torch.cholesky_inverse(l)
    return torch.linalg.cholesky(hinv, upper=True)


def in_group_act_order(diag_mean: torch.Tensor, group: int) -> torch.Tensor:
    """Permutation that sorts columns by descending diagonal INSIDE each fixed group."""
    k = diag_mean.numel()
    perm = torch.empty(k, dtype=torch.long, device=diag_mean.device)
    for g0 in range(0, k, group):
        order = torch.argsort(diag_mean[g0:g0 + group], descending=True)
        perm[g0:g0 + group] = g0 + order
    return perm


@torch.no_grad()
def gptq_quantize(
    w: torch.Tensor,
    h: torch.Tensor,
    quantizer,
    group: int | None = None,
    block: int = 128,
    percdamp: float = 0.01,
    act_order_static: bool = True,
    sparse24: bool = False,
    sparse_pattern: str = "2:4",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (W_q dequantized [E,N,K] in w.dtype, per-expert accumulated GPTQ loss [E]).

    sparse24=True enables structured sparsity with the given pattern: "2:4" (2 of 4 elements) or
    "4:8" (two adjacent pairs of the four pairs in each 8-element chunk; FP4 hardware pattern)."""
    e, n, k = w.shape
    dev = w.device
    group = group or getattr(quantizer, "group", 16)
    assert k % group == 0 and block % group == 0
    h = h.to(torch.float32)
    diag_h = torch.diagonal(h, dim1=-2, dim2=-1)
    if act_order_static:
        perm = in_group_act_order(diag_h.mean(0), group)
    else:
        perm = torch.arange(k, device=dev)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(k, device=dev)

    wq = w.to(torch.float32)[:, :, perm].clone()
    hp = h[:, perm][:, :, perm]
    hinv = hessian_inverse_chol(hp, percdamp)
    losses = torch.zeros(e, device=dev)

    for c0 in range(0, k, block):
        c1 = min(c0 + block, k)
        w_blk = wq[:, :, c0:c1]                        # view; updated in place
        err_blk = torch.zeros(e, n, c1 - c0, device=dev)
        hinv_blk = hinv[:, c0:c1, c0:c1]
        hdiag = torch.diagonal(hinv_blk, dim1=-2, dim2=-1)   # [E, blk]
        for g0 in range(0, c1 - c0, group):
            g1 = g0 + group
            grp = w_blk[:, :, g0:g1]
            if sparse24:
                # saliency in ORIGINAL layout: un-permute this group's columns (they map back to the same
                # 16-block), decide 2-of-4 on original positions, re-permute the mask
                cols_perm = perm[c0 + g0:c0 + g1]                       # original column ids, in processing order
                local = cols_perm - (c0 + g0)                           # 0..group-1 original positions
                sal = grp ** 2 / (hdiag[:, None, g0:g1] ** 2 + 1e-12)   # in processing order
                sal_orig = torch.empty_like(sal)
                sal_orig[:, :, local] = sal                             # back to original order
                if sparse_pattern == "4:8":
                    salp = sal_orig.reshape(e, n, group // 8, 4, 2).sum(-1)      # pair saliency
                    keep = salp.topk(2, dim=-1).indices
                    mp = torch.zeros_like(salp, dtype=torch.bool).scatter_(-1, keep, True)
                    mask_orig = mp.unsqueeze(-1).expand(*mp.shape, 2).reshape(e, n, group)
                else:
                    sal4 = sal_orig.reshape(e, n, group // 4, 4)
                    keep = sal4.topk(2, dim=-1).indices
                    mask_orig = torch.zeros_like(sal4, dtype=torch.bool).scatter_(-1, keep, True).reshape(e, n, group)
                mask = mask_orig[:, :, local]                           # into processing order
                grp_m = grp * mask
            else:
                mask = None
                grp_m = grp
            state = quantizer.state(grp_m)
            for j in range(group):
                col = g0 + j
                cur = w_blk[:, :, g0:g1]
                if mask is not None:
                    cur = cur * mask
                qcol = quantizer.requant(cur, state)[:, :, j]
                if mask is not None:
                    qcol = qcol * mask[:, :, j]
                d = hdiag[:, col]
                err = (w_blk[:, :, col] - qcol) / d[:, None]
                w_blk[:, :, col:] -= err[:, :, None] * hinv_blk[:, col, col:][:, None, :]
                w_blk[:, :, col] = qcol
                err_blk[:, :, col] = err
                losses += (err ** 2).sum(dim=1) * d ** 2 / 2
        if c1 < k:
            wq[:, :, c1:] -= err_blk @ hinv[:, c0:c1, c1:]
    return wq[:, :, inv].to(w.dtype), losses


@torch.no_grad()
def hessian_weighted_error(w: torch.Tensor, wq: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """tr((W - Wq) H (W - Wq)^T) per expert, [E]."""
    d = (w - wq).to(torch.float32)
    return torch.einsum("enk,ekl,enl->e", d, h.to(torch.float32), d)


if __name__ == "__main__":
    import sys, time
    sys.path.insert(0, ".")
    from nvfp4 import GroupQuantizer, nvfp4_quantize, NVFP4Config
    torch.manual_seed(0)
    e, n, k, t = 8, 768, 2048, 3000
    w = torch.randn(e, n, k, device="cuda") * torch.rand(e, 1, k, device="cuda") * 0.03
    x = torch.randn(e, t, k, device="cuda") * torch.rand(e, 1, k, device="cuda") * 2
    h = (x.transpose(1, 2) @ x) / t
    q = GroupQuantizer("T1_nvfp4", NVFP4Config(search_grid=8))
    for ao in (False, True):
        t0 = time.time(); wq, loss = gptq_quantize(w, h, q, act_order_static=ao); torch.cuda.synchronize(); dt = time.time() - t0
        rtn = nvfp4_quantize(w, NVFP4Config(search_grid=8))
        eg, er = hessian_weighted_error(w, wq, h).mean(), hessian_weighted_error(w, rtn, h).mean()
        print(f"act_order={ao}: gptq {dt:.2f}s | H-err gptq={eg:.4g} rtn={er:.4g} ratio={eg/er:.3f}")
    t0 = time.time(); wq2, _ = gptq_quantize(w, h, q, sparse24=True); torch.cuda.synchronize()
    nz = (wq2 != 0).float().reshape(e, n, k // 4, 4).sum(-1)
    print(f"sparse24 gptq {time.time()-t0:.2f}s | H-err={hessian_weighted_error(w, wq2, h).mean():.4g} nnz={(wq2!=0).float().mean():.3f} max-per-4={int(nz.max())}")
    e = 128; w = torch.randn(e, 1536, k, device="cuda") * 0.02; x = torch.randn(e, 2048, k, device="cuda"); h = (x.transpose(1, 2) @ x) / 2048
    t0 = time.time(); wq, loss = gptq_quantize(w, h, q); torch.cuda.synchronize(); print(f"full-layer gate+up stack (128 x 1536 x 2048): {time.time()-t0:.1f}s")
