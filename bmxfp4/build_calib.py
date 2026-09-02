"""Build the calibration windows from Brandon's REAP calibration corpus (the calibration data of record):
  /home/brandonmusic/klc-linux/reap_recall_build/work/calibration/reap_recall_calib.jsonl  (12,228 samples, 4 balanced axes)

Convention (same as tools/blockwise_nvfp4_saliency.py in the REAP build): the `text` field is tokenized RAW
(add_special_tokens=True, no chat-template re-rendering, no truncation).  Because the Hessian/attribution code
works on fixed 2048-token windows, samples are packed per axis into a token stream (EOS between samples) and
the stream is cut into 2048-token windows.  Per axis: shuffle with a fixed seed, consume samples until
`per_axis` windows are filled.  Windows are interleaved so that the first 4*k windows are balanced for any k
(the first 32 form the attribution subset, role "calib-attrib"; all 64 are the Hessian/rotation role "calib").

usage: build_calib.py --per-axis 16 --seed 20260902
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from campaign import BF16, CALIB, receipt, sha256_file, write_json

SRC = Path("/home/brandonmusic/klc-linux/reap_recall_build/work/calibration/reap_recall_calib.jsonl")
AXES = ["axis1_general", "axis2_legal", "axis3_code_agentic", "axis4_reasoning_termination"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-axis", type=int, default=16)
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--src", type=Path, default=SRC)
    a = ap.parse_args()
    t0 = time.time()
    CALIB.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(a.src)]
    by_axis = {ax: [i for i, r in enumerate(rows) if r["axis"] == ax] for ax in AXES}
    tok = AutoTokenizer.from_pretrained(str(BF16))
    eos = tok.eos_token_id
    windows_by_axis = {}
    manifest = {}
    for ax in AXES:
        idx = by_axis[ax][:]
        random.Random(a.seed + AXES.index(ax)).shuffle(idx)
        stream, used, wins = [], [], []
        for i in idx:
            ids = tok(rows[i]["text"], add_special_tokens=True, truncation=False, padding=False)["input_ids"]
            if not ids:
                continue
            stream.extend(ids + [eos])
            used.append({"row": i, "source": rows[i]["source"], "tokens": len(ids), "sha256": hashlib.sha256(rows[i]["text"].encode()).hexdigest()[:16]})
            while len(stream) >= a.window and len(wins) < a.per_axis:
                wins.append(stream[:a.window]); stream = stream[a.window:]
            if len(wins) >= a.per_axis:
                break
        assert len(wins) == a.per_axis, (ax, len(wins))
        windows_by_axis[ax] = wins
        manifest[ax] = {"samples_used": len(used), "tokens_used": sum(u["tokens"] for u in used), "samples": used}
    # interleave axes: window j*4 + k = axis k's j-th window
    ids = np.zeros((a.per_axis * len(AXES), a.window), dtype=np.int64)
    axis_of = []
    for j in range(a.per_axis):
        for k, ax in enumerate(AXES):
            ids[j * len(AXES) + k] = windows_by_axis[ax][j]; axis_of.append(ax)
    out = CALIB / "reap-calib-windows.npz"
    np.savez(out, input_ids=ids, axis=np.array(axis_of))
    write_json(CALIB / "manifest.json", {"source": str(a.src), "source_sha256": sha256_file(a.src), "windows": int(ids.shape[0]), "window_tokens": a.window,
                                         "per_axis": a.per_axis, "seed": a.seed, "tokenizer": str(BF16), "packing": "raw text field, add_special_tokens=True, EOS between samples, no truncation, per-axis stream cut into windows",
                                         "axes": manifest})
    receipt(CALIB / "receipt.json", "calibration-windows", source_sha256=sha256_file(a.src), npz_sha256=sha256_file(out), windows=int(ids.shape[0]), window_tokens=a.window,
            per_axis=a.per_axis, seed=a.seed, samples_used={ax: manifest[ax]["samples_used"] for ax in AXES}, tokens_used={ax: manifest[ax]["tokens_used"] for ax in AXES}, seconds=time.time() - t0)
    print(json.dumps({ax: (manifest[ax]["samples_used"], manifest[ax]["tokens_used"]) for ax in AXES}), "->", out, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
