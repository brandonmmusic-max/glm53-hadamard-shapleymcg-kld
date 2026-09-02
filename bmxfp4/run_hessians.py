"""Hessian capture on the fit role, one process per GPU handling a slice of layers.

usage: run_hessians.py --device cuda:1 --layers 0-11
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from campaign import BF16, HESS, LOGS, load_seal, role_ids, receipt, write_json, log_line
from hessian import collect_hessians
from model_io import load_model


def parse_layers(s: str) -> list[int]:
    a, b = s.split("-")
    return list(range(int(a), int(b) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--layers", required=True)
    ap.add_argument("--role", default="fit")
    a = ap.parse_args()
    layers = parse_layers(a.layers)
    log = LOGS / f"hessians-{a.layers}.log"
    seal = load_seal()
    ids = role_ids(seal, a.role)
    t0 = time.time()
    model = load_model(BF16, a.device)
    log_line(log, f"model loaded {time.time()-t0:.0f}s; capturing layers {layers[0]}-{layers[-1]} on {len(ids)} {a.role} windows")
    stats = collect_hessians(model, ids, layers, HESS, a.device, log=lambda m: log_line(log, m))
    receipt(HESS / f"receipt-{a.layers}.json", "hessians", role=a.role, windows=int(ids.shape[0]), layers=layers, token_stats=stats, seal_sha256=seal["_sha256"])
    log_line(log, f"done in {time.time()-t0:.0f}s; token stats {stats}")


if __name__ == "__main__":
    main()
