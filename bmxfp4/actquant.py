"""W4A4 emulation: quantize expert inputs to NVFP4 (per-token blocks of 16, dynamic E4M3 block scales,
per-call FP32 tensor scale) in the transform basis used for the weights.

With installed pseudo-quant weight W_eff = (Q(W_perm R) R^T)[:, inv_perm], replacing the input x by
x~ = (R Qa(R^T x[perm]))[inv_perm] yields W_eff x~ = Q(W_perm R) Qa(R^T x[perm]) = the W4A4 product.
"""
from __future__ import annotations

import torch

from nvfp4 import nvfp4_quantize, NVFP4Config
from rotation import apply_block_rotation

ACT_CFG = NVFP4Config(scale_search=False, range_max_search=False, tensor_scale_iters=0)  # plain absmax dynamic


class ActQuantHooks:
    """transforms: dict[(layer, kind)] -> {"R": [16,16] or None, "perm": LongTensor or None}, kind in {"in","mid"}."""

    def __init__(self, model, transforms: dict, cfg: NVFP4Config = ACT_CFG):
        self.handles = []
        self.cfg = cfg
        from model_io import moe_block, num_layers
        for l in range(num_layers(model)):
            blk = moe_block(model, l)
            t_in = transforms.get((l, "in"), {})
            t_mid = transforms.get((l, "mid"), {})
            for ex in blk.experts:
                self.handles.append(ex.gate_proj.register_forward_pre_hook(self._mk(t_in)))
                self.handles.append(ex.up_proj.register_forward_pre_hook(self._mk(t_in)))
                self.handles.append(ex.down_proj.register_forward_pre_hook(self._mk(t_mid)))

    def _mk(self, t):
        R = t.get("R"); perm = t.get("perm")
        inv = None
        if perm is not None:
            inv = torch.empty_like(perm); inv[perm] = torch.arange(perm.numel(), device=perm.device)

        def hook(module, args):
            x = args[0]
            if x.numel() == 0:
                return None
            shp = x.shape
            xf = x.reshape(-1, shp[-1]).float()
            if perm is not None:
                xf = xf[:, perm]
            if R is not None:
                xf = apply_block_rotation(xf, R.T.to(xf.device), group=R.shape[0])
            xq = nvfp4_quantize(xf, self.cfg)
            if R is not None:
                xq = apply_block_rotation(xq, R.to(xq.device), group=R.shape[0])
            if inv is not None:
                xq = xq[:, inv]
            return (xq.reshape(shp).to(x.dtype),) + tuple(args[1:])
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
