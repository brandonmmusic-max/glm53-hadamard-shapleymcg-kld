#!/usr/bin/env python3
"""Preserve terminal P8 tail-repair receipts without raw logits or private logs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ROOT = Path("/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6")
DEST = REPO / "evidence/opened/codec-v2/p8-tail-omission-repair-v1"
FILES = {
    "image/receipt.json": (ROOT / "p8-tail-repair-image-v1a/receipt.json", "abb82eb3c19285bf050695bc8fde0a6eab9250efa1369457cf96936bbeba0181"),
    "failures/v1-execution.json": (ROOT / "p8-tail-omission-repair-v1/execution.json", "d2964c716b649e731d74833fad849f3c3341c8900c16b49bae308965a300f536"),
    "failures/v1a-execution.json": (ROOT / "p8-tail-omission-repair-v1a/execution.json", "0d4318d349a788b40418ec8972df1d3513f0fed91af8b051973107482cb3ddf1"),
    "execution.json": (ROOT / "p8-tail-omission-repair-v1b/execution.json", "31c9cf5200f76d211d1e20ee54083ecd6ebd0955f52a49047da30b930ae45e7a"),
    "candidate-execution.json": (ROOT / "p8-tail-omission-repair-v1b/candidate/execution.json", "e1488b987b31e9dae2403e3991999ef8fc6f9735b148897d8519aa670eecce7b"),
    "analysis.json": (ROOT / "p8-tail-omission-repair-v1b-kld/analysis.json", "44d3f9037600436ceecac449f00b2348dccf7f463f35d50ae3d36d9059d0efb5"),
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
    if (
        analysis.get("verdict") != "material-tail-cause-rejected"
        or analysis.get("physical_bpw") != 4.25
        or analysis.get("windows") != 4
        or analysis.get("protected_roles_opened") != []
        or execution.get("exit_code") != 0
        or execution.get("capture_protocol_complete") is not True
        or execution.get("restoration_safety", {}).get("ok") is not True
    ):
        raise ValueError("terminal tail-repair verdict differs")
    for failed in ("failures/v1-execution.json", "failures/v1a-execution.json"):
        if values[failed].get("exit_code") != 1:
            raise ValueError("preflight failure receipt differs")
    snapshot = {
        "schema": "glm53.p8-tail-omission-repair-snapshot.v1",
        "status": "complete",
        "files": inventory,
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
