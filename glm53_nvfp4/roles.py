"""Build deterministic, disjoint role manifests from the local sealed token panel."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROLE_COUNTS_PER_DOMAIN = {
    "fit": 16,
    "conditional-fit": 4,
    "selection": 4,
    "confirmation": 8,
}
SELECTION_SALT = "glm53-nvfp4-v2"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rank_key(item: dict) -> str:
    payload = f"{SELECTION_SALT}:{item['window_id']}:{item['token_ids_sha256']}"
    return hashlib.sha256(payload.encode()).hexdigest()


def build(panel_path: Path, arrays_dir: Path) -> dict:
    panel = json.loads(panel_path.read_text())
    by_role_domain: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in panel["windows"]:
        if item["role"] in ROLE_COUNTS_PER_DOMAIN:
            by_role_domain[(item["role"], item["domain"])].append(item)

    domains = sorted({item["domain"] for item in panel["windows"]})
    if len(domains) != 4:
        raise ValueError(f"expected four domains, found {domains}")

    result = {
        "schema": "glm53-nvfp4-v2.roles.v1",
        "selection_rule": "lowest sha256(salt:window_id:token_ids_sha256) per source role and domain",
        "selection_salt": SELECTION_SALT,
        "source_panel": str(panel_path),
        "source_panel_sha256": sha256_file(panel_path),
        "sealed_corpus_sha256": panel["sealed_corpus_sha256"],
        "roles": {role: [] for role in (*ROLE_COUNTS_PER_DOMAIN, "final")},
    }
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for role, per_domain in ROLE_COUNTS_PER_DOMAIN.items():
        for domain in domains:
            pool = sorted(by_role_domain[(role, domain)], key=rank_key)
            if len(pool) < per_domain:
                raise ValueError(f"{role}/{domain}: need {per_domain}, found {len(pool)}")
            for item in pool[:per_domain]:
                wid = item["window_id"]
                token_path = arrays_dir / f"{wid}.tokens.npy"
                if not token_path.is_file():
                    raise FileNotFoundError(token_path)
                tokens = np.load(token_path, mmap_mode="r")
                if tokens.shape != (2048,):
                    raise ValueError(f"{token_path}: expected (2048,), got {tokens.shape}")
                actual = sha256_file(token_path)
                # The panel commits to the complete NPY container. Also record the raw payload
                # independently so dtype/shape changes cannot hide behind a container hash.
                payload_hash = hashlib.sha256(np.asarray(tokens).tobytes(order="C")).hexdigest()
                if actual != item["token_ids_sha256"]:
                    raise ValueError(f"{wid}: token NPY hash mismatch")
                if wid in seen_ids or actual in seen_hashes:
                    raise ValueError(f"role overlap: {wid}")
                seen_ids.add(wid)
                seen_hashes.add(actual)
                result["roles"][role].append({
                    "id": wid,
                    "domain": domain,
                    "input_sha256": actual,
                    "payload_sha256": payload_hash,
                    "token_path": str(token_path),
                    "teacher_path": f"logits/full-panel/{role}/{wid}.safetensors",
                    "prediction_positions": item["prediction_positions"],
                    "rank_sha256": rank_key(item),
                })
    result["counts"] = {role: len(items) for role, items in result["roles"].items()}
    result["domains"] = dict(Counter(x["domain"] for role in ROLE_COUNTS_PER_DOMAIN for x in result["roles"][role]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.panel, args.arrays)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "counts": manifest["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
