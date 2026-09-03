"""Build the V3 continuation roles without reusing opened V2 evaluation rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .roles import sha256_file


SALT = "glm53-nvfp4-v3-rotation-shapley6"


def _rank(item: dict) -> str:
    return hashlib.sha256(f"{SALT}:{item['window_id']}:{item['token_ids_sha256']}".encode()).hexdigest()


def _entry(item: dict, arrays: Path, wave: int | None = None) -> dict:
    token_path = arrays / f"{item['window_id']}.tokens.npy"
    if not token_path.is_file():
        raise FileNotFoundError(token_path)
    tokens = np.load(token_path, mmap_mode="r", allow_pickle=False)
    if tokens.shape != (2048,):
        raise ValueError(f"{token_path}: expected 2048 tokens")
    actual = sha256_file(token_path)
    if actual != item["token_ids_sha256"]:
        raise ValueError(f"{item['window_id']}: token hash mismatch")
    result = {
        "id": item["window_id"],
        "domain": item["domain"],
        "input_sha256": actual,
        "token_path": str(token_path),
        "teacher_path": f"logits/full-panel/{item['role']}/{item['window_id']}.safetensors",
        "prediction_positions": item["prediction_positions"],
        "rank_sha256": _rank(item),
    }
    if wave is not None:
        result["selection_wave"] = wave
    return result


def build(panel_path: Path, arrays: Path, prior_roles_path: Path) -> dict:
    panel = json.loads(panel_path.read_text())
    prior = json.loads(prior_roles_path.read_text())["roles"]
    used_eval = {x["id"] for role in ("conditional-fit", "selection", "confirmation") for x in prior[role]}
    prior_fit = {x["id"] for x in prior["fit"]}
    by_role_domain: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in panel["windows"]:
        by_role_domain[(item["role"], item["domain"])].append(item)
    domains = sorted({x["domain"] for x in panel["windows"]})
    if len(domains) != 4:
        raise ValueError(f"expected four domains, found {domains}")

    roles: dict[str, list[dict]] = {name: [] for name in ("fit", "conditional-fit", "selection", "confirmation", "final")}
    # Reuse the prior 64 calibration-only fit windows.  This is deliberate:
    # they were never scored, and axis4 has no unused fit windows.
    panel_by_id = {x["window_id"]: x for x in panel["windows"]}
    for wid in sorted(prior_fit, key=lambda x: _rank(panel_by_id[x])):
        roles["fit"].append(_entry(panel_by_id[wid], arrays))

    # Fresh conditional-fit: four per domain, excluding every opened V2 row.
    for domain in domains:
        pool = sorted((x for x in by_role_domain[("conditional-fit", domain)] if x["window_id"] not in used_eval), key=_rank)
        if len(pool) < 4:
            raise ValueError(f"conditional-fit/{domain}: only {len(pool)} unused")
        roles["conditional-fit"].extend(_entry(x, arrays) for x in pool[:4])

    # Three fresh, balanced 16-window selection waves.  Wave n+1 may be opened
    # only if the predeclared wave-n candidate family fails its gate.
    for domain in domains:
        pool = sorted((x for x in by_role_domain[("selection", domain)] if x["window_id"] not in used_eval), key=_rank)
        if len(pool) < 12:
            raise ValueError(f"selection/{domain}: only {len(pool)} unused")
        for index, item in enumerate(pool[:12]):
            roles["selection"].append(_entry(item, arrays, wave=index // 4 + 1))

    # Seven per domain remain untouched for one final confirmation of the
    # frozen 6-bpw allocation; this is 28 independent windows.
    for domain in domains:
        pool = sorted((x for x in by_role_domain[("confirmation", domain)] if x["window_id"] not in used_eval), key=_rank)
        if len(pool) < 7:
            raise ValueError(f"confirmation/{domain}: only {len(pool)} unused")
        roles["confirmation"].extend(_entry(x, arrays) for x in pool[:7])

    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for role, entries in roles.items():
        for item in entries:
            if item["id"] in seen_ids or item["input_sha256"] in seen_hashes:
                raise ValueError(f"V3 role overlap at {role}/{item['id']}")
            seen_ids.add(item["id"])
            seen_hashes.add(item["input_sha256"])
    if used_eval & seen_ids:
        raise ValueError("V3 reused opened V2 evaluation data")

    return {
        "schema": "glm53-nvfp4-v3.roles.v1",
        "selection_salt": SALT,
        "selection_rule": "lowest salted SHA-256 per source role/domain after excluding V2 evaluation IDs",
        "source_panel": str(panel_path),
        "source_panel_sha256": sha256_file(panel_path),
        "prior_roles": str(prior_roles_path),
        "prior_roles_sha256": sha256_file(prior_roles_path),
        "fit_reuse": "V2 fit only; calibration inputs were never evaluation-scored",
        "roles": roles,
        "counts": {role: len(items) for role, items in roles.items()},
        "domains": dict(Counter(x["domain"] for entries in roles.values() for x in entries)),
        "selection_waves": {str(w): [x["id"] for x in roles["selection"] if x["selection_wave"] == w] for w in (1, 2, 3)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--prior-roles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.panel, args.arrays, args.prior_roles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "counts": result["counts"], "waves": {k: len(v) for k, v in result["selection_waves"].items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
