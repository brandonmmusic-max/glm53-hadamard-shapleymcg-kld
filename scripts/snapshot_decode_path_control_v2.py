#!/usr/bin/env python3
"""Preserve the terminal matched decode-control receipts without raw logits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CAPTURE = Path(
    "/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/"
    "decode-path-matched-control-v2"
)
SCORED = Path(
    "/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/"
    "decode-path-matched-control-v2-kld"
)
DEST = REPO / "evidence/opened/codec-v2/decode-path-matched-control-v2"
EXECUTION_SHA256 = "7b9be8c0faf1190608f2342e1721f8fb67e780611e5c10ea9c35496014991396"
ANALYSIS_SHA256 = "50a70045f018609261b1d4f82f2a79ca46441f1a26576e4f3a5d19e71c5f0b46"
WINDOWS = (
    "conditional-fit-0032",
    "conditional-fit-0021",
    "conditional-fit-0090",
    "conditional-fit-0003",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"immutable snapshot differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def checked(path: Path, expected: str | None = None) -> tuple[bytes, dict]:
    if path != path.resolve() or path.is_symlink() or not path.is_file():
        raise ValueError(f"canonical regular file required: {path}")
    data = path.read_bytes()
    receipt = {"bytes": len(data), "sha256": digest(data), "source": str(path)}
    if expected is not None and receipt["sha256"] != expected:
        raise ValueError(f"source identity differs: {path}")
    return data, receipt


def main() -> None:
    execution_data, execution_receipt = checked(
        CAPTURE / "execution.json", EXECUTION_SHA256
    )
    analysis_data, analysis_receipt = checked(
        SCORED / "analysis.json", ANALYSIS_SHA256
    )
    execution = json.loads(execution_data)
    analysis = json.loads(analysis_data)
    if (
        execution.get("exit_code") != 0
        or execution.get("capture_protocol_complete") is not True
        or execution.get("mtp") is not False
        or execution.get("protected_roles_opened") != []
        or execution.get("restoration_safety", {}).get("ok") is not True
        or analysis.get("status") != "complete"
        or analysis.get("windows") != 4
        or analysis.get("codec_exonerated_for_0_11_endpoint") is not True
        or analysis.get("shared_production_serving_bug_supported") is not True
        or analysis.get("protected_roles_opened") != []
    ):
        raise ValueError("terminal control result differs from the frozen verdict")

    inventory = {
        "execution.json": execution_receipt,
        "analysis.json": analysis_receipt,
    }
    save(DEST / "execution.json", execution_data)
    save(DEST / "analysis.json", analysis_data)
    for window in WINDOWS:
        for arm in ("stock", "exl3"):
            name = f"{window}.{arm}.json"
            data, receipt = checked(SCORED / name)
            row = json.loads(data)
            if row.get("window_id") != window or row.get("arm") != arm:
                raise ValueError(f"window identity differs: {name}")
            save(DEST / "windows" / name, data)
            inventory[f"windows/{name}"] = receipt

    snapshot = {
        "schema": "glm53.decode-path-matched-control-snapshot.v1",
        "status": "complete",
        "files": inventory,
        "omitted": [
            "10,145,259,520 raw-logit bytes retained on NVMe and hash-bound by execution receipts",
            "private container launch records, server logs, and hardware XML",
            "score NPZ payloads; their hashes remain in analysis.json",
        ],
        "claim_boundary": analysis["claim_boundary"],
    }
    save(
        DEST / "snapshot.json",
        (json.dumps(snapshot, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps({"status": "pass", "files": len(inventory), "destination": str(DEST)}))


if __name__ == "__main__":
    main()
