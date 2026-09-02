"""Stage 0: capture BF16 teacher logits (fp32) for the sealed panels, and run the B0 instrument check.

usage: run_teacher.py --device cuda:0 --panels selection,final,confirmation,wikitext [--b0-runs 3]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from campaign import BF16, TEACHER, LOGS, MODEL_ID, MODEL_REVISION, load_seal, panel_ids, receipt, sha256_file, write_json, log_line
from kld import capture_logits
from model_io import load_model, describe
from safetensors.torch import load_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--panels", default="selection,final,confirmation,wikitext")
    ap.add_argument("--b0-runs", type=int, default=3)
    a = ap.parse_args()
    log = LOGS / "teacher.log"
    seal = load_seal()
    log_line(log, f"seal sha256={seal['_sha256']} roles={seal['role_counts']}")
    t0 = time.time()
    model = load_model(BF16, a.device)
    log_line(log, f"model loaded in {time.time()-t0:.0f}s: {describe(model)}")
    for panel in a.panels.split(","):
        ids = panel_ids(panel, seal)
        out = TEACHER / panel
        t1 = time.time()
        paths = capture_logits(model, ids, out, device=a.device)
        log_line(log, f"teacher {panel}: {len(paths)} windows x {ids.shape[1]} tokens in {time.time()-t1:.0f}s")
        receipt(out / "receipt.json", "teacher-logits", panel=panel, windows=int(ids.shape[0]), tokens=int(ids.shape[1]),
                model_id=MODEL_ID, revision=MODEL_REVISION, dtype="float32", attention="sdpa",
                token_ids_sha256=__import__("hashlib").sha256(ids.tobytes()).hexdigest(),
                files={p.name: sha256_file(p) for p in paths[:3]}, seal_sha256=seal["_sha256"])
    # B0: instrument determinism — re-run the first selection window N times and compare bitwise
    ids = panel_ids("selection", seal)[:1]
    ref = load_file(str(TEACHER / "selection" / "row-000.safetensors"))["logits"]
    bitwise = []
    for r in range(a.b0_runs):
        with torch.no_grad():
            lg = model(input_ids=torch.from_numpy(ids).to(a.device), use_cache=False).logits[0].float().cpu()
        bitwise.append(bool(torch.equal(lg, ref)))
        maxdiff = float((lg - ref).abs().max())
        log_line(log, f"B0 run {r}: bitwise_identical={bitwise[-1]} max|diff|={maxdiff:.3e}")
    receipt(TEACHER / "b0-receipt.json", "b0-instrument", runs=a.b0_runs, bitwise_identical=bitwise)
    log_line(log, "teacher stage done")


if __name__ == "__main__":
    main()
