#!/usr/bin/env python3
"""Preserve terminal P8/EXL3 FP8-path receipts without raw logits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CAPTURE = Path("/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/decode-path-fp8-ds-mla-p8-exl3-v2")
SCORED = Path("/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/decode-path-fp8-ds-mla-p8-exl3-v2-kld")
DEST = REPO / "evidence/opened/codec-v2/p8-exl3-fp8-ds-mla-control-v2"
EXECUTION_SHA256 = "73f77a5c3b6c5c6dbf5e17685d061feb842c5d6d73be514a4962404e4d1d0d8b"
ANALYSIS_SHA256 = "a628a5e415c0eae370c19983c0428f30cfcbfdde42511cc3dd6af6b50465ec88"
WINDOWS = ("conditional-fit-0032", "conditional-fit-0021", "conditional-fit-0090", "conditional-fit-0003")


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
    execution_data, execution_receipt = checked(CAPTURE / "execution.json", EXECUTION_SHA256)
    analysis_data, analysis_receipt = checked(SCORED / "analysis.json", ANALYSIS_SHA256)
    execution, analysis = json.loads(execution_data), json.loads(analysis_data)
    if (
        execution.get("exit_code") != 0
        or execution.get("capture_protocol_complete") is not True
        or execution.get("restoration_safety", {}).get("ok") is not True
        or execution.get("mtp") is not False
        or analysis.get("status") != "complete"
        or analysis.get("verdict") != "nvfp4-production-mla-stack-implicated"
        or analysis.get("windows") != 4
        or analysis.get("protected_roles_opened") != []
        or any(analysis["arms"][arm]["window_wins"] != 4 for arm in ("p8", "exl3"))
    ):
        raise ValueError("terminal FP8-path result differs from the sealed verdict")
    inventory = {"execution.json": execution_receipt, "analysis.json": analysis_receipt}
    save(DEST / "execution.json", execution_data)
    save(DEST / "analysis.json", analysis_data)
    for window in WINDOWS:
        for arm in ("p8", "exl3"):
            name = f"{window}.{arm}.json"
            data, receipt = checked(SCORED / name)
            row = json.loads(data)
            if row.get("window_id") != window or row.get("arm") != arm:
                raise ValueError(f"window identity differs: {name}")
            save(DEST / "windows" / name, data)
            inventory[f"windows/{name}"] = receipt
    snapshot = {
        "schema": "glm53.p8-exl3-fp8-ds-mla-control-snapshot.v2",
        "status": "complete",
        "files": inventory,
        "omitted": [
            "10,145,259,520 raw-logit bytes retained on NVMe and hash-bound by execution receipts",
            "private container launch records, server logs, hardware XML, and score NPZ payloads",
            "failed-v1 private startup log; its exact hashes are frozen in the public amendment",
        ],
        "claim_boundary": analysis["claim_boundary"],
    }
    save(DEST / "snapshot.json", (json.dumps(snapshot, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps({"status": "pass", "files": len(inventory), "destination": str(DEST)}))


if __name__ == "__main__":
    main()
