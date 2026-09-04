"""Materialize and hash-verify the 32-window development role for codec gates."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from safetensors import safe_open


ADDITIONAL_IDS = (
    "conditional-fit-0003",
    "conditional-fit-0009",
    "conditional-fit-0010",
    "conditional-fit-0011",
    "conditional-fit-0030",
    "conditional-fit-0032",
    "conditional-fit-0035",
    "conditional-fit-0040",
    "conditional-fit-0041",
    "conditional-fit-0046",
    "conditional-fit-0060",
    "conditional-fit-0063",
    "conditional-fit-0074",
    "conditional-fit-0099",
    "conditional-fit-0113",
    "conditional-fit-0118",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(
    prior_roles: Path,
    panel: Path,
    teacher_root: Path,
    teacher_manifest_path: Path,
    output: Path,
) -> dict[str, object]:
    prior = json.loads(prior_roles.read_text())
    panel_payload = json.loads(panel.read_text())
    panel_by_id = {item["window_id"]: item for item in panel_payload["windows"]}
    manifest = json.loads(teacher_manifest_path.read_text())
    teacher_by_id = {item["window_id"]: item for item in manifest["logit_files"]}
    ids = [item["id"] for item in prior["roles"]["conditional-fit"]]
    ids.extend(ADDITIONAL_IDS)
    if len(ids) != 32 or len(set(ids)) != 32:
        raise RuntimeError("conditional-fit32 must contain exactly 32 unique windows")

    windows = []
    for window_id in ids:
        source = panel_by_id[window_id]
        teacher = teacher_by_id[window_id]
        relative_teacher = Path(teacher["path"])
        teacher_path = teacher_root / relative_teacher
        token_path = panel.parent / "arrays" / f"{window_id}.tokens.npy"
        metadata_path = (
            teacher_root / ".cache/huggingface/download" / f"{relative_teacher}.metadata"
        )
        metadata_lines = metadata_path.read_text().splitlines()
        if len(metadata_lines) < 2 or metadata_lines[1] != teacher["sha256"]:
            raise RuntimeError(f"Hub metadata mismatch for {window_id}")
        if teacher_path.stat().st_size != teacher["bytes"]:
            raise RuntimeError(f"teacher byte count mismatch for {window_id}")
        if _sha256(teacher_path) != teacher["sha256"]:
            raise RuntimeError(f"teacher hash mismatch for {window_id}")
        if _sha256(token_path) != source["token_ids_sha256"]:
            raise RuntimeError(f"token hash mismatch for {window_id}")
        with safe_open(str(teacher_path), framework="pt", device="cpu") as handle:
            tensor = handle.get_slice("logits")
            st_metadata = handle.metadata() or {}
            if (
                tuple(tensor.get_shape())
                != (source["prediction_positions"], manifest["vocab_size"])
                or tensor.get_dtype() != "F32"
                or st_metadata.get("window_id") != window_id
                or st_metadata.get("token_ids_sha256")
                != source["token_ids_sha256"]
            ):
                raise RuntimeError(f"teacher tensor metadata mismatch for {window_id}")
        windows.append(
            {
                "id": window_id,
                "domain": source["domain"],
                "input_sha256": source["token_ids_sha256"],
                "prediction_positions": source["prediction_positions"],
                "teacher_path": teacher["path"],
                "teacher_sha256": teacher["sha256"],
                "teacher_source_role": "conditional-fit",
                "token_path": str(token_path),
                "source_partition": (
                    "previously-used-conditional-fit"
                    if window_id not in ADDITIONAL_IDS
                    else "previously-unscored-conditional-fit"
                ),
            }
        )

    domain_counts = Counter(item["domain"] for item in windows)
    if set(domain_counts.values()) != {8}:
        raise RuntimeError(f"conditional-fit32 is not domain balanced: {domain_counts}")
    payload = {
        "schema": "glm53-codec.conditional-fit32-development-role.v1",
        "source_panel": str(panel),
        "source_panel_sha256": _sha256(panel),
        "teacher_manifest": str(teacher_manifest_path),
        "teacher_manifest_sha256": _sha256(teacher_manifest_path),
        "teacher_repo_id": manifest["repo_id"],
        "teacher_revision": manifest["source_hub_revision"],
        "prior_roles": str(prior_roles),
        "prior_roles_sha256": _sha256(prior_roles),
        "selection_rule": "fixed 16 used plus the 16 IDs named before scoring; no selection or confirmation windows",
        "counts": {
            "fit": 0,
            "conditional-fit": 32,
            "selection": 0,
            "confirmation": 0,
            "final": 0,
        },
        "domains": dict(sorted(domain_counts.items())),
        "selection_waves": {},
        "roles": {
            "fit": [],
            "conditional-fit": windows,
            "selection": [],
            "confirmation": [],
            "final": [],
        },
        "protection_boundary": "development role only; the 28 confirmation logits remain unopened",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-roles", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build(
        args.prior_roles,
        args.panel,
        args.teacher_root,
        args.teacher_manifest,
        args.output,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": _sha256(args.output),
                "counts": payload["counts"],
                "domains": payload["domains"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
