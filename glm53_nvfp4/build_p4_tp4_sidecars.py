"""Export immutable native-RNE P4 matrix manifests for GLM TP4 serving."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .p4_serving_codec import export_tp4, file_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-manifest", type=Path, action="append", required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = export_tp4(args.matrix_manifest, args.output_dir,
                        expected_design_sha256=file_sha256(args.design), resume=args.resume)
    print(json.dumps({"manifest": str(args.output_dir / "manifest.json"),
                      "layers": result["layers"], "total": result["total"]}, sort_keys=True))


if __name__ == "__main__":
    main()
