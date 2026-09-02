"""Batched GPTQ / SparseGPT with an arbitrary block quantizer in the loop.

Operates on a batch of weight matrices W [E, N, K] (E experts sharing a layout, N output rows,
K input columns) with per-expert Hessians H [E, K, K] (H = X^T X over the tokens routed to the
expert).  Quantization is done column-block by column-block (block = group size of the format,
16 for NVFP4, 32 for MXFP6) with error feedback to the not-yet-quantized columns (OBQ / GPTQ).

Static act-order (MR-GPTQ ingredient 2): columns are processed in descending Hessian-diagonal
order INSIDE each fixed group; the group membership / storage layout never changes, so no runtime
permutation is needed.  Because a group is quantized jointly (one block scale), the intra-group
order only affects the error feedback, which is exactly the point.

The quantizer callback receives a [E, N, g] slab and returns its dequantized version.
"""
from __future__ import annotations

import torch


def _hessian_inverse(h: torch.Tensor, percdamp: float = 0.01) -> torch.Tensor:
    """Damped Cholesky inverse, batched.  Returns upper Cholesky factor of H^-1 (GPTQ convention)."""
    e, k, _ = h.shape
    h = h.clone()
    dead = torch.diagonal(h, dim1=-2, dim2=-1) == 0
    idx = torch.arange(k, device=h.device)
    h[:, idx, idx] = torch.where(dead, torch.ones_like(h[:, idx, idx]), h[:, idx, idx])
    damp = percdamp * torch.diagonal(h, dim1=-2, dim2=-1).mean(dim=-1, keepdim=True)
    h[:, idx, idx] += damp
    l = torch.linalg.cholesky(h)
    hinv = torch.cholesky_inverse(l)
    u = torch.linalg.cholesky(hinv, upper=True)
    return u


@torch.no_grad()
def gptq_quantize(
    w: torch.Tensor,
    h: torch.Tensor,
    quantizer,
    group: int = 16,
    block: int = 128,
    percdamp: float = 0.01,
    act_order_static: bool = True,
    sparse24: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (W_q dequantized [E,N,K], per-expert Hessian-weighted loss [E]).

    sparse24=True performs SparseGPT-style 2:4 pruning inside each group of 4 (mask chosen by
    w^2 / [H^-1]_ii saliency), then quantizes the survivors; the mask is applied jointly with the
    error feedback so pruned weights are compensated.
    """
    e, n, k = w.shape
    dev = w.device
    wq = w.to(torch.float32).clone()
    hinv = _hessian_inverse(h.to(torch.float32), percdamp)          # [E, K, K] upper
    losses = torch.zeros(e, device=dev)
    diag_h = torch.diagonal(h, dim1=-2, dim2=-1)                    # [E, K]
    assert k % group == 0 and block % group == 0

    for c0 in range(0, k, block):
        c1 = min(c0 + block, k)
        w_blk = wq[:, :, c0:c1].clone()
        q_blk = torch.zeros_like(w_blk)
        err_blk = torch.zeros_like(w_blk)
        hinv_blk = hinv[:, c0:c1, c0:c1]

        for g0 in range(c0, c1, group):
            g1 = g0 + group
            # column order inside the group: descending Hessian diagonal (shared choice across experts
            # would break the batch; we use the batch-mean diagonal, which keeps the loop batched)
            if act_order_static:
                order = torch.argsort(diag_h[:, g0:g1].mean(0), descending=True)
            else:
                order = torch.arange(group, device=dev)
            # quantize the whole group jointly (one block scale), but feed error column by column
            # in `order`: emulate by iterating columns and re-quantizing the group with the current
            # (error-updated) values for the not-yet-fixed columns.
            fixed = torch.zeros(group, dtype=torch.bool, device=dev)
            grp_vals = w_blk[:, :, g0 - c0:g1 - c0]                    # view into current block state
            if sparse24:
                # SparseGPT saliency on the current values: w^2 / hinv_ii^2 per element, 2 of 4 kept
                hd = torch.diagonal(hinv_blk, dim1=-2, dim2=-1)[:, g0 - c0:g1 - c0]  # [E, group]
                sal = grp_vals ** 2 / (hd[:, None, :] ** 2 + 1e-12)
                sal4 = sal.reshape(e, n, group // 4, 4)
                keep_idx = sal4.topk(2, dim=-1).indices
                mask = torch.zeros_like(sal4, dtype=torch.bool).scatter_(-1, keep_idx, True).reshape(e, n, group)
            else:
                mask = torch.ones(e, n, group, dtype=torch.bool, device=dev)
            qgrp = torch.zeros_like(grp_vals)
            for j in order.tolist():
                col = g0 - c0 + j
                # current group values with pruning mask applied
                cur = grp_vals * mask
                qg = quantizer(cur) * mask                              # [E, N, group] dequantized
                qcol = qg[:, :, j]
                qgrp[:, :, j] = qcol
                d = hinv_blk[:, col, col]                               # [E]
                err = (w_blk[:, :, col] - qcol) / d[:, None]            # [E, N]
                # feed error to remaining (unfixed) columns of this block
                upd = err[:, :, None] * hinv_blk[:, col, col:c1 - c0][:, None, :]   # [E, N, rest]
                w_blk[:, :, col:] -= upd
                # restore the fixed value for this column (it is now quantized)
                w_blk[:, :, col] = qcol
                err_blk[:, :, col] = err
                fixed[j] = True
                losses += (err ** 2 * d[:, None]).sum(dim=1) / 2
            q_blk[:, :, g0 - c0:g1 - c0] = qgrp
        wq[:, :, c0:c1] = q_blk
        # propagate the block's accumulated error to all later columns
        if c1 < k:
            wq[:, :, c1:] -= err_blk @ hinv[:, c0:c1, c1:]
    return wq.to(w.dtype), losses


@torch.no_grad()
def hessian_weighted_error(w: torch.Tensor, wq: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """tr((W - Wq) H (W - Wq)^T) per expert, [E]."""
    d = (w - wq).to(torch.float32)
    return torch.einsum("enk,ekl,enl->e", d, h.to(torch.float32), d)


if __name__ == "__main__":
    import time
    from nvfp4 import nvfp4_quantize, NVFP4Config
    torch.manual_seed(0)
    e, n, k, t = 8, 768, 2048, 4096
    w = torch.randn(e, n, k, device="cuda") * 0.02
    x = torch.randn(e, t, k, device="cuda")
    h = x.transpose(1, 2) @ x
    q = lambda s: nvfp4_quantize(s, NVFP4Config(search_grid=6))
    t0 = time.time(); wq, loss = gptq_quantize(w, h, q); torch.cuda.synchronize()
    rtn = q(w)
    print(f"gptq {time.time()-t0:.1f}s  rel-H-err gptq={hessian_weighted_error(w,wq,h).mean():.4g} rtn={hessian_weighted_error(w,rtn,h).mean():.4g}")
