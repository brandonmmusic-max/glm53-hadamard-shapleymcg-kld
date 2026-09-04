"""Build a fresh V4 rotation-selection role after V2/V3 exhaustion.

The published panel's 64 source-``selection`` windows are all opened. V4
therefore promotes never-opened source-``conditional-fit`` windows into one
32-window protected selection role. Their source role remains explicit so the
teacher-manifest verifier can distinguish storage provenance from experimental
use. The V3 confirmation role is carried forward unopened for the eventual
frozen 6-bpw endpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .roles import sha256_file


SALT = "glm53-nvfp4-v4-rtn-router-stability"
DOMAINS = (
    "axis1_general",
    "axis2_legal",
    "axis3_code_agentic",
    "axis4_reasoning_termination",
)


def _rank(item: dict) -> str:
    identity = f"{SALT}:{item['window_id']}:{item['token_ids_sha256']}"
    return hashlib.sha256(identity.encode()).hexdigest()


def _entry(item: dict, arrays: Path, *, experimental_role: str) -> dict:
    token_path = arrays / f"{item['window_id']}.tokens.npy"
    if not token_path.is_file():
        raise FileNotFoundError(token_path)
    tokens = np.load(token_path, mmap_mode="r", allow_pickle=False)
    if tokens.shape != (2048,):
        raise ValueError(f"{token_path}: expected 2048 tokens")
    actual = sha256_file(token_path)
    if actual != item["token_ids_sha256"]:
        raise ValueError(f"{item['window_id']}: token hash mismatch")
    return {
        "id": item["window_id"],
        "domain": item["domain"],
        "input_sha256": actual,
        "token_path": str(token_path),
        "teacher_path": (
            f"logits/full-panel/{item['role']}/{item['window_id']}.safetensors"
        ),
        "teacher_source_role": item["role"],
        "experimental_role": experimental_role,
        "prediction_positions": item["prediction_positions"],
        "rank_sha256": _rank(item),
    }


def build(
    panel_path: Path,
    arrays: Path,
    v2_roles_path: Path,
    v3_roles_path: Path,
) -> dict:
    panel = json.loads(panel_path.read_text())
    v2 = json.loads(v2_roles_path.read_text())
    v3 = json.loads(v3_roles_path.read_text())
    by_id = {item["window_id"]: item for item in panel["windows"]}
    by_role_domain: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in panel["windows"]:
        by_role_domain[(item["role"], item["domain"])].append(item)

    used_eval = set()
    for prior in (v2, v3):
        for role, entries in prior["roles"].items():
            if role != "fit":
                used_eval.update(item["id"] for item in entries)

    roles: dict[str, list[dict]] = {
        "fit": [],
        "conditional-fit": [],
        "selection": [],
        "confirmation": [],
        "final": [],
    }

    # Calibration-only inputs remain the same immutable 64-window REAP subset.
    for prior in v3["roles"]["fit"]:
        source = by_id[prior["id"]]
        roles["fit"].append(_entry(source, arrays, experimental_role="fit"))

    # One and only one fresh V4 selection wave: eight windows per domain.
    for domain in DOMAINS:
        pool = sorted(
            (
                item
                for item in by_role_domain[("conditional-fit", domain)]
                if item["window_id"] not in used_eval
            ),
            key=_rank,
        )
        if len(pool) < 8:
            raise ValueError(f"conditional-fit/{domain}: only {len(pool)} unused")
        roles["selection"].extend(
            _entry(item, arrays, experimental_role="selection")
            for item in pool[:8]
        )

    # Preserve, do not open, the existing 6-bpw confirmation allocation.
    for prior in v3["roles"]["confirmation"]:
        source = by_id[prior["id"]]
        roles["confirmation"].append(
            _entry(source, arrays, experimental_role="confirmation")
        )

    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for role, entries in roles.items():
        for item in entries:
            if item["id"] in seen_ids or item["input_sha256"] in seen_hashes:
                raise ValueError(f"V4 role overlap at {role}/{item['id']}")
            seen_ids.add(item["id"])
            seen_hashes.add(item["input_sha256"])
    selection_ids = {item["id"] for item in roles["selection"]}
    if used_eval & selection_ids:
        raise ValueError("V4 selection reused a V2/V3 declared evaluation row")
    if Counter(item["domain"] for item in roles["selection"]) != Counter(
        {domain: 8 for domain in DOMAINS}
    ):
        raise ValueError("V4 selection is not domain balanced")

    return {
        "schema": "glm53-nvfp4-v4.roles.v1",
        "selection_salt": SALT,
        "selection_rule": (
            "lowest salted SHA-256 per domain among source conditional-fit "
            "windows absent from every V2/V3 declared non-fit role"
        ),
        "role_remap": {
            "source_role": "conditional-fit",
            "experimental_role": "selection",
            "reason": "all 64 source selection windows were opened by V2/V3",
        },
        "source_panel": str(panel_path),
        "source_panel_sha256": sha256_file(panel_path),
        "prior_roles": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in (v2_roles_path, v3_roles_path)
        ],
        "fit_reuse": "V3 fit only; calibration inputs have no teacher-KLD scores",
        "confirmation_reuse": "V3 confirmation allocation; still unopened",
        "roles": roles,
        "counts": {role: len(items) for role, items in roles.items()},
        "domains": {
            role: dict(Counter(item["domain"] for item in items))
            for role, items in roles.items()
        },
        "selection_waves": {
            "1": [item["id"] for item in roles["selection"]]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--v2-roles", type=Path, required=True)
    parser.add_argument("--v3-roles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        args.panel, args.arrays, args.v2_roles, args.v3_roles
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "counts": result["counts"],
                "selection_windows": len(result["selection_waves"]["1"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
