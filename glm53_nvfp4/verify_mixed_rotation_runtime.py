"""Verify H16 execution across mixed Humming-NVFP4 and B12X-MXFP6 layers."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .verify_rotation_runtime import parse_layers, verify


MX_ACTIVE = re.compile(
    r"GLM53_MXFP6_H16_ALL_PROJECTION_PATCH_ACTIVE "
    r"prefix=.*?layers\.(?P<layer>\d+)\.mlp\.experts "
    r"rank=(?P<rank>\d+) rotation_sha256=(?P<sha>[0-9a-f]{64}) "
    r"input=python-block16 mid=fused-w6a8-block16"
)
MX_FORWARD = re.compile(
    r"GLM53_MXFP6_H16_INPUT_FORWARD "
    r"prefix=.*?layers\.(?P<layer>\d+)\.mlp\.experts "
    r"rank=(?P<rank>\d+) rotation_sha256=(?P<sha>[0-9a-f]{64}) "
    r"hidden_width=(?P<width>\d+) "
    r"dtype=(?P<dtype>torch\.[a-z0-9_]+)"
)


def verify_mixed(text: str, rotated: set[int], mxfp6: set[int], ranks: set[int]) -> dict:
    if not mxfp6 or not mxfp6.issubset(rotated):
        raise RuntimeError("every MXFP6 layer must be in the rotated layer set")
    nvfp4 = rotated - mxfp6
    nv = verify(text, nvfp4, ranks, require_mid=True) if nvfp4 else None
    expected = {(layer, rank) for layer in mxfp6 for rank in ranks}
    active_rows = [m.groupdict() for m in MX_ACTIVE.finditer(text)]
    forward_rows = [m.groupdict() for m in MX_FORWARD.finditer(text)]
    active = {(int(row["layer"]), int(row["rank"])) for row in active_rows}
    forward = {(int(row["layer"]), int(row["rank"])) for row in forward_rows}
    expected_sha = "c0cce70ab9288f764401571e81431a5af51cd595f2beb6f864ee517e8bdf89be"
    bad_sha = sorted(
        (int(row["layer"]), int(row["rank"]), row["sha"])
        for row in active_rows + forward_rows
        if row["sha"] != expected_sha
    )
    bad_geometry = sorted(
        (int(row["layer"]), int(row["rank"]), int(row["width"]), row["dtype"])
        for row in forward_rows
        if int(row["width"]) != 4096 or row["dtype"] != "torch.bfloat16"
    )
    if active != expected or forward != expected or bad_geometry or bad_sha:
        raise RuntimeError(
            "MXFP6 H16 evidence mismatch "
            f"active_missing={sorted(expected-active)} active_unexpected={sorted(active-expected)} "
            f"forward_missing={sorted(expected-forward)} forward_unexpected={sorted(forward-expected)} "
            f"bad_geometry={bad_geometry} bad_sha={bad_sha}"
        )
    return {
        "schema": "glm53-nvfp4-v10.mixed-h16-runtime-proof.v1",
        "status": "pass",
        "rotated_layers": sorted(rotated),
        "nvfp4_layers": sorted(nvfp4),
        "mxfp6_layers": sorted(mxfp6),
        "ranks": sorted(ranks),
        "nvfp4_proof": nv,
        "mxfp6_active_pairs": len(active),
        "mxfp6_forward_pairs": len(forward),
        "mxfp6_input_transform": "normalized H16 before fused FC1",
        "mxfp6_mid_transform": "normalized H16 on BF16 post-SwiGLU blocks before MXFP8 FC2 quantization",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--mixed-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.mixed_receipt.read_text())
    result = verify_mixed(
        args.log.read_text(errors="replace"),
        parse_layers(args.layers),
        set(map(int, receipt["mxfp6_layers"])),
        set(range(args.ranks)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
