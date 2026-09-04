#!/usr/bin/env python3
"""Write a replayable structural receipt; never touches GPU/model data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glm53_nvfp4.p4_encoder import encoder_lut_sha256, encoder_state_lut
from glm53_nvfp4.p4_fixture import scalar_mcg_code
from glm53_nvfp4.p4_codec import e2m1_codes
from glm53_nvfp4.p4_serving_codec import file_sha256
from glm53_nvfp4.p4_serving_fixture import verify_fixture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--pytest-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite structural receipt")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("receipt replay must explicitly hide all CUDA devices")
    test_text = args.pytest_log.read_text()
    match = re.search(r"(\d+) passed(?:, (\d+) skipped)? in [\d.]+s\s*$", test_text)
    if match is None or re.search(r"\d+ failed|\d+ error", test_text):
        raise ValueError("test log does not record an all-pass/explicit-skip run")
    root = Path(__file__).resolve().parents[1]
    files = ["experiments/p4-serving-bridge-v2.json", "glm53_nvfp4/p4_codec.py",
             "glm53_nvfp4/p4_encoder.py", "glm53_nvfp4/p4_serving_codec.py",
             "glm53_nvfp4/p4_serving_fixture.py", "glm53_nvfp4/build_p4_tp4_sidecars.py",
             "glm53_nvfp4/quantize_p4_layer.py", "glm53_nvfp4/trellis_nvfp4.py",
             "glm53_nvfp4/block_gptq.py", "glm53_nvfp4/capture.py",
             "tests/test_p4_serving_bridge.py", "scripts/verify_p4_bridge.py",
             "docs/P4_SERVING_BRIDGE.md"]
    codes = e2m1_codes(encoder_state_lut().view(torch.float8_e4m3fn).float().numpy())
    expected = np.array([scalar_mcg_code(state) for state in range(65536)], dtype=np.uint8)
    differences = int(np.count_nonzero(codes != expected))
    if differences:
        raise ValueError(f"native law differs at {differences} states")
    result = {"schema": "glm53-p4-serving-bridge-structural-receipt.v2",
              "evidence_level": "structural", "base_commit": "d456831b77c5b418e348a53a0bdb4d75b86d0b91",
              "plan_commit": "45771db", "plan_sha256": file_sha256(root / files[0]),
              "source_hashes": {name: file_sha256(root / name) for name in files},
              "source_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "numpy": np.__version__, "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
                              "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                              "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS")},
              "tests": {"path": str(args.pytest_log), "sha256": file_sha256(args.pytest_log),
                        "passed": int(match[1]), "skipped": int(match[2] or 0),
                        "skip_reason": "CUDA Viterbi encoder required; no GPU authorized for this stage"},
              "state_census": {"states": 65536, "byte_mismatches": differences,
                               "negative_zero_states": int(np.count_nonzero(codes == 8)),
                               "encoder_lut_sha256": encoder_lut_sha256(),
                               "encoder_only_lut_bytes": 65536, "runtime_state_lut_bytes": 0},
              "fixture": verify_fixture(args.fixture),
              "stage_timings": [{"path": str(path), "sha256": file_sha256(path),
                                 "counters": json.loads(path.read_text())}
                                for path in sorted((args.fixture / "ranks").glob("export-timings-*.json"))],
              "proven": ["native-RNE state labels including signed zero", "six-tensor TP4 export and exact charged globals",
                         "synthetic full FC1/FC2 TP4 CPU equality", "off-diagonal Hessian materially reaches selection",
                         "optimized export equals serial bytes", "plan-bound verified prefix/full resume"],
              "not_tested": ["real CUDA Viterbi calibration", "GLM P4 kernel launch", "GPU bit-exact closure",
                             "integrated GLM serving", "KLD", "prefill/decode throughput"],
              "protected_roles_opened": [], "ldlq": False,
              "authorization": "Parent assigned structural bridge work; all GPUs reserved by the ongoing P8 build."}
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"receipt": str(args.output), "sha256": file_sha256(args.output),
                      "tests_passed": result["tests"]["passed"], "evidence_level": "structural"}, sort_keys=True))


if __name__ == "__main__":
    main()
