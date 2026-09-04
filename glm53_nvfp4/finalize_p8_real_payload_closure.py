"""Validate and seal a real layer-3 P8 device arithmetic closure."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repeat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    repeat = json.loads(args.repeat.read_text())
    rule = plan["decision_rule"]
    cells = repeat.get("cells", [])
    expected_tokens = plan["geometry"]["tokens"]
    cell_pass = len(cells) == len(expected_tokens) and {
        row.get("tokens") for row in cells
    } == set(expected_tokens) and all(
        row.get("pass")
        and row.get("finite") == rule["finite"]
        and row.get("bits") == 4
        and row.get("tokens") in expected_tokens
        and row.get("cosine") > rule["cosine_min_exclusive"]
        and row.get("relative_l2") < rule["relative_l2_max_exclusive"]
        for row in cells
    )
    runtime_match = (
        repeat["runtime_dynamic"]["sha256"] == plan["sources"]["runtime_dynamic"]
    )
    real = repeat.get("real_layer_payload") or {}
    payload_match = (
        real.get("codec", {}).get("sha256") == plan["codec"]["sha256"]
        and real.get("dense_reference", {}).get("sha256")
        == plan["dense_reference"]["sha256"]
        and real.get("experts") == plan["geometry"]["experts"]
    )
    contract_match = (
        repeat.get("decision") == "pass"
        and repeat.get("boundary") == "identity"
        and repeat.get("codebook_selection")
        == "explicit constructor argument: mcg"
        and repeat.get("ldlq") is False
        and repeat.get("geometry", {}).get("hidden") == 4096
        and repeat.get("geometry", {}).get("intermediate") == 2048
        and repeat.get("geometry", {}).get("materialize_intermediate")
        == (plan["geometry"]["kernel"] == "split-prefill")
    )
    passed = cell_pass and runtime_match and payload_match and contract_match
    result = {
        "schema": "glm53-p8-real-layer3-device-closure-result.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision": "pass-real-payload-device-closure" if passed else "fail",
        "checks": {
            "registered_tolerance": cell_pass,
            "runtime_source_identity": runtime_match,
            "real_payload_identity": payload_match,
            "codec_contract": contract_match,
        },
        "metrics": cells,
        "inputs": {
            "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)},
            "repeat": {"path": str(args.repeat), "sha256": sha256_file(args.repeat)},
        },
        "stored_bpw": 4.25,
        "table_bytes": 0,
        "ldlq": False,
        "strict_kld_gate_relaxed": False,
        "engineering_window_gate_relaxed": True,
        "scope": (
            f"real layer-3 experts 0:8 {plan['geometry']['kernel']} FC1/FC2 device "
            "arithmetic only; not end-to-end kernel KLD, speed, or determinism qualification"
        ),
        "isa_cost": "mxf8f6f4 uses twice the MMA issue count of NVFP4",
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
