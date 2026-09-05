"""Create disjoint domain-balanced fit/tune views of the existing fit64 role."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .shard_index import sha256_file


ROLE_NAMES = ("fit", "conditional-fit", "selection", "confirmation", "final")


def build(source: Path) -> tuple[dict, dict]:
    payload = json.loads(source.read_text())
    fit = payload["roles"]["fit"]
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for item in fit:
        by_domain[item["domain"]].append(item)
    if len(by_domain) != 4 or any(len(items) != 16 for items in by_domain.values()):
        raise ValueError("source fit role must contain 16 windows in each of four domains")

    train: list[dict] = []
    tune: list[dict] = []
    for domain in sorted(by_domain):
        ordered = sorted(by_domain[domain], key=lambda item: item["rank_sha256"])
        train.extend(dict(item, teacher_source_role="fit") for item in ordered[:8])
        tune.extend(dict(item, teacher_source_role="fit") for item in ordered[8:])
    if {item["id"] for item in train} & {item["id"] for item in tune}:
        raise RuntimeError("fit/tune split overlaps")

    def view(name: str, rows: list[dict]) -> dict:
        roles = {role: [] for role in ROLE_NAMES}
        roles["fit"] = rows
        return {
            "schema": "glm53-rotation-v8.fit-split.v1",
            "split": name,
            "selection_rule": (
                "within each domain, sort source fit64 by rank_sha256; first eight "
                "are train and last eight are tune"
            ),
            "source_roles": str(source.resolve()),
            "source_roles_sha256": sha256_file(source),
            "source_panel_sha256": payload["source_panel_sha256"],
            "roles": roles,
            "counts": {role: len(items) for role, items in roles.items()},
            "domains": {
                domain: sum(item["domain"] == domain for item in rows)
                for domain in sorted(by_domain)
            },
            "protected_roles_copied": [],
        }

    return view("train32", train), view("tune32", tune)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--tune-output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.train_output, args.tune_output):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    train, tune = build(args.source)
    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    args.tune_output.parent.mkdir(parents=True, exist_ok=True)
    args.train_output.write_text(json.dumps(train, indent=2, sort_keys=True) + "\n")
    args.tune_output.write_text(json.dumps(tune, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "train": {"path": str(args.train_output), "sha256": sha256_file(args.train_output)},
                "tune": {"path": str(args.tune_output), "sha256": sha256_file(args.tune_output)},
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
