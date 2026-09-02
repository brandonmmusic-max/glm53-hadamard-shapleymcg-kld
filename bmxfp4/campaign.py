"""Campaign paths, sealed inputs, receipts."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("BMXFP4_ROOT", "/media/brandonmusic/klcstore/bmxfp4"))
MODELS = ROOT / "models"
BF16 = Path(os.environ.get("BMXFP4_MODEL", str(MODELS / "Qwen3-30B-A3B")))
SEAL_OVERRIDE = os.environ.get("BMXFP4_SEAL_JSON")  # local seal file (tiny smoke tests)
SEALS = ROOT / "seals"
WORK = ROOT / "work"
LOGS = ROOT / "logs"
TEACHER = ROOT / "teacher"
HESS = ROOT / "hessians"
ARMS = ROOT / "arms"
CODE = Path("/home/brandonmusic/KLC_SANDBOXES/bmxfp4-qwen3-30b-a3b")

MODEL_ID = "Qwen/Qwen3-30B-A3B"
MODEL_REVISION = "4c446470ba0aec43e22ac1128f9ffd915f338ba3"  # same weights as main (16/16 shard SHAs identical)
SEAL_DATASET = "brandonmusic/shapleymcg-qwen3-30b-a3b-reproducibility"
SEAL_FILE = "controls/fixed-hadamard-k34-v1/calibration/qwen-sealed-corpus.json"
WIKI_FILE = "causal-arm-v3/turboderp-wiki2-sdpa-teacher/input-token-ids.npz"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=False))


def receipt(path: Path, kind: str, **fields):
    rec = {"kind": kind, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **fields}
    write_json(path, rec)
    return rec


def load_seal() -> dict:
    if SEAL_OVERRIDE:
        p = SEAL_OVERRIDE
    else:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(SEAL_DATASET, SEAL_FILE, repo_type="dataset")
    d = json.load(open(p))
    d["_local_path"] = p
    d["_sha256"] = sha256_file(Path(p))
    return d


def role_ids(seal: dict, role: str) -> np.ndarray:
    return np.array([w["token_ids"] for w in seal["windows"][role]], dtype=np.int64)


def wiki_ids() -> np.ndarray:
    if SEAL_OVERRIDE:
        d = json.load(open(SEAL_OVERRIDE))
        return np.array([w["token_ids"] for w in d["windows"]["selection"][:2]], dtype=np.int64)
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(SEAL_DATASET, WIKI_FILE, repo_type="dataset")
    return np.load(p)["input_ids"].astype(np.int64)


def panel_ids(name: str, seal: dict) -> np.ndarray:
    if name == "wikitext":
        return wiki_ids()
    return role_ids(seal, name)


def log_line(path: Path, msg: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(path, "a") as f:
        f.write(line + "\n")
