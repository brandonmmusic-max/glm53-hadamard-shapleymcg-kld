#!/usr/bin/env python3
"""Preserve terminal P8 tail-V2 receipts without raw logits or private logs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ROOT = Path("/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6")
DEST = REPO / "evidence/opened/codec-v2/p8-tail-omission-repair-v2"
FILES = {
    "image/receipt.json": (ROOT / "p8-tail-repair-image-v2a/receipt.json", "96f2389d872620ee084d2c9b51d7a06767545d1e396cb4d46517df5872e408cc"),
    "device-row-closure.json": (ROOT / "p8-tail-repair-image-v2a/device-row-closure.json", "03d4994d05fcb979491ba435f2960092917e96d6097c9c9048c5dca2c79ffbe7"),
    "execution.json": (ROOT / "p8-tail-omission-repair-v2/execution.json", "76f7ab31ed838b79b38537e5cf8623022e5607977f26080124363b27ac9a89df"),
    "candidate-execution.json": (ROOT / "p8-tail-omission-repair-v2/candidate/execution.json", "ea2dc36c56a0f6643b541105f45a04471892d85c3a5af8d0c5e77b389fde920b"),
    "analysis.json": (ROOT / "p8-tail-omission-repair-v2-kld/analysis.json", "4a2df9e5fe8656469456094066f1d7bf39071b4b2ff5cf0d3edf353c26cdf7a9"),
}


def save(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"immutable snapshot differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def main() -> None:
    inventory = {}
    values = {}
    for name, (source, expected) in FILES.items():
        data = source.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if source != source.resolve() or source.is_symlink() or actual != expected:
            raise ValueError(f"terminal source differs: {source}")
        save(DEST / name, data)
        inventory[name] = {"source": str(source), "bytes": len(data), "sha256": actual}
        values[name] = json.loads(data)
    analysis = values["analysis.json"]
    execution = values["execution.json"]
    closure = values["device-row-closure.json"]
    if (
        analysis.get("verdict") != "material-tail-cause-supported"
        or analysis.get("physical_bpw") != 4.25
        or analysis.get("windows") != 4
        or analysis.get("window_wins") != 4
        or analysis.get("protected_roles_opened") != []
        or closure.get("diff_count_by_L_mod_4") != {"0": 0, "1": 512, "2": 512, "3": 512}
        or closure.get("differing_rows") != 1536
        or closure.get("unchanged_rows") != 511
        or execution.get("exit_code") != 0
        or execution.get("capture_protocol_complete") is not True
        or execution.get("restoration_safety", {}).get("ok") is not True
    ):
        raise ValueError("terminal tail-V2 result differs")
    snapshot = {
        "schema": "glm53.p8-tail-omission-repair-snapshot.v2",
        "status": "complete",
        "files": inventory,
        "runtime": {
            "attention_backend": "B12X_MLA_SPARSE",
            "kv_cache_dtype": "nvfp4_ds_mla",
            "moe_backend": "native-p8-mxf8f6f4-n64-fused-scratch",
            "activation_precision": "E4M3 K32",
            "physical_bpw": 4.25,
        },
        "omitted": [
            "5,072,629,760 raw-logit bytes retained on NVMe and hash-bound by candidate execution",
            "private launch records, logs, requests, hardware XML, and score NPZs",
        ],
        "claim_boundary": analysis["claim_boundary"],
    }
    save(DEST / "snapshot.json", (json.dumps(snapshot, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps({"status": "pass", "files": len(inventory), "destination": str(DEST)}))


if __name__ == "__main__":
    main()
