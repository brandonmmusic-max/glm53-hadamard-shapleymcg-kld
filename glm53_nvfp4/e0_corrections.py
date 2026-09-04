"""Write append-only corrections for the audited trellis development records."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_corrections(hybrid_root: Path, output_root: Path, code_path: Path) -> None:
    code_hash = _sha256(code_path)
    for tag in ("reg025", "reg1", "reg4"):
        source = hybrid_root / f"l3-e0-fit512-balanced-{tag}-s2p16-v1.json"
        payload = json.loads(source.read_text())
        _write(
            output_root / f"{source.stem}-e0-correction.json",
            {
                "schema": "glm53-trellis-e0-objective-correction.v1",
                "original": {
                    "path": str(source),
                    "sha256": _sha256(source),
                },
                "corrected_code": {
                    "path": str(code_path),
                    "sha256": code_hash,
                },
                "selector_isolated_regularization": payload[
                    "selector_isolated_regularization"
                ],
                "correction": (
                    "local_initial_objective and joint_objective_reduction_percent "
                    "are invalidated because the original initial objective omitted "
                    "lambda times isolated error while the final objective included it"
                ),
                "artifact_and_endpoint_status": (
                    "unchanged; the optimizer used the penalized current objective, "
                    "so this defect affected the reported objective comparison"
                ),
                "corrected_objective_value": "Not recoverable from the stored summary; no value imputed",
                "status": "diagnostic-invalidated",
            },
        )
        corrected = payload
        for receipt in corrected.get("tile_receipts", []):
            for diagnostics in receipt.get("diagnostics", {}).values():
                diagnostics["local_initial_objective"] = None
                diagnostics["joint_objective_reduction_percent"] = None
                diagnostics["objective_comparison_status"] = (
                    "invalidated: original initial objective omitted the regularizer"
                )
        corrected["e0_correction"] = {
            "schema": "glm53-trellis-e0-objective-correction.v1",
            "original_path": str(source),
            "original_sha256": _sha256(source),
            "corrected_code_sha256": code_hash,
            "status": "restamped-with-invalid-diagnostic-removed",
            "note": "No corrected value was imputed and the stopped tri-law screen was not rerun.",
        }
        _write(output_root / f"{source.stem}-e0-corrected.json", corrected)

    for stem in (
        "l3-e0-fit-cv-offset128-s2p16-v1",
        "l3-e0-fit-cv-offset512-fit512-s2p16-v1",
    ):
        source = hybrid_root / f"{stem}.json"
        payload = json.loads(source.read_text())
        effect = payload["aggregate"]["hybrid"][
            "relative_error_reduction_vs_matched_rtn_percent"
        ]
        payload["practical_gate_percent"] = 10.0
        payload["decision"] = "pass" if effect >= 10.0 else "fail"
        payload["uncertainty"] = {
            "status": "not-computed",
            "unit": "expert",
            "expert_count": 1,
            "reason": "expert-bootstrap inference requires at least 16 experts",
        }
        payload["e0_correction"] = {
            "schema": "glm53-trellis-e0-gate-correction.v1",
            "original_path": str(source),
            "original_sha256": _sha256(source),
            "correction": "restamped at the preregistered 10 percent gate; gate 0 is forbidden",
            "inferential_status": "one-expert development result; no bootstrap CI",
        }
        _write(output_root / f"{stem}-e0-corrected.json", payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--code-path",
        type=Path,
        default=Path(__file__).with_name("trellis_nvfp4.py"),
    )
    args = parser.parse_args()
    write_corrections(args.hybrid_root, args.output_root, args.code_path)


if __name__ == "__main__":
    main()
