"""Teacher logit capture and student KLD scoring on sealed 2048-token windows.

Metric (same manner as the prior Qwen ShapleyMCG campaigns): per-token KL(teacher || student) over
positions 0..T-2 (predicting token t+1), full vocabulary, computed in float32 from stored fp32 (or
bf16-safe) teacher logits; per-window mean; window-weighted mean (all windows have 2048 tokens so
this equals the token mean); p90/p99 of per-token KLD; top-1 agreement; paired bootstrap 95% CI of
the difference in per-window mean KLD between two arms (windows are the resampling unit).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


def _windows_from_seal(seal: dict, role: str) -> tuple[np.ndarray, list[dict]]:
    ws = seal["windows"][role]
    ids = np.array([w["token_ids"] for w in ws], dtype=np.int64)
    meta = [{"document_id": w.get("document_id"), "domain": w.get("domain"), "offset": w.get("offset"), "token_sha256": w.get("token_sha256")} for w in ws]
    return ids, meta


@torch.no_grad()
def capture_logits(model, ids: np.ndarray, out_dir: Path, batch: int = 1, dtype=torch.float32, device="cuda", tag="teacher") -> list[Path]:
    """Run the model on each window; store logits [T, V] per window as float32 safetensors."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(0, len(ids), batch):
        chunk = torch.from_numpy(ids[i:i + batch]).to(device)
        out = model(input_ids=chunk, use_cache=False)
        logits = out.logits.to(dtype)
        for j in range(chunk.shape[0]):
            p = out_dir / f"row-{i + j:03d}.safetensors"
            save_file({"logits": logits[j].contiguous().cpu()}, str(p))
            paths.append(p)
    return paths


@torch.no_grad()
def score_student(model, ids: np.ndarray, teacher_dir: Path, device="cuda", batch: int = 1) -> dict:
    """Per-window KLD of the (already patched) student model against stored teacher logits."""
    per_window, per_token_all, top1 = [], [], []
    t0 = time.time()
    for i in range(0, len(ids), batch):
        chunk = torch.from_numpy(ids[i:i + batch]).to(device)
        s_logits = model(input_ids=chunk, use_cache=False).logits.float()
        for j in range(chunk.shape[0]):
            t_logits = load_file(str(teacher_dir / f"row-{i + j:03d}.safetensors"))["logits"].to(device).float()
            tl = F.log_softmax(t_logits[:-1], dim=-1)
            sl = F.log_softmax(s_logits[j, :-1], dim=-1)
            kl = (tl.exp() * (tl - sl)).sum(dim=-1)              # [T-1]
            per_token_all.append(kl.cpu().numpy())
            per_window.append(float(kl.mean()))
            top1.append(float((tl.argmax(-1) == sl.argmax(-1)).float().mean()))
    tok = np.concatenate(per_token_all)
    return {
        "per_window_mean_kld": per_window,
        "per_window_top1": top1,
        "mean_kld": float(np.mean(per_window)),
        "token_mean_kld": float(tok.mean()),
        "p90_token_kld": float(np.percentile(tok, 90)),
        "p99_token_kld": float(np.percentile(tok, 99)),
        "top1_agreement": float(np.mean(top1)),
        "windows": len(per_window),
        "tokens": int(tok.size),
        "seconds": time.time() - t0,
        "per_token": tok.astype(np.float32),
    }


def paired_bootstrap(a: list[float], b: list[float], draws: int = 20000, seed: int = 20260902) -> dict:
    """95% CI of mean(a) - mean(b) resampling windows with replacement (paired)."""
    a = np.asarray(a); b = np.asarray(b); n = len(a)
    assert n == len(b)
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, n, size=(draws, n))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff_mean": float(d.mean()), "ci95": [float(lo), float(hi)], "excludes_zero": bool(lo > 0 or hi < 0), "draws": draws, "windows": n}


def save_report(report: dict, path: Path, per_token_path: Path | None = None):
    r = dict(report)
    tok = r.pop("per_token", None)
    if tok is not None and per_token_path is not None:
        np.save(per_token_path, tok)
        r["per_token_file"] = str(per_token_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(r, indent=1))
    return r
