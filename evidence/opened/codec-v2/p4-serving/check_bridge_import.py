"""Replay the real mounted-backend import and six-tensor bridge compatibility.

Usage: CUDA_VISIBLE_DEVICES='' PYTHONPATH=runtime_patch python3 <this-file> RANK_DIR
CPU-only; every CUDA discovery/initialization entry below is forbidden.
"""
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open


def forbidden(*args, **kwargs):
    raise AssertionError("CUDA access is forbidden in this host check")


for name in ("_lazy_init", "is_available", "get_device_capability", "current_device"):
    setattr(torch.cuda, name, forbidden)

import p4_glm_serving  # noqa: E402,F401
from b12x_h16.b12x.moe._shared.kernels.p4_native import P4TrellisMoEBackend  # noqa: E402
from p4_native_kernel import _verify_file, validate_payload  # noqa: E402

print("actual_backend_import", P4TrellisMoEBackend.__module__, P4TrellisMoEBackend.backend_name)
root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text())
for entry in manifest["entries"]:
    path = root / entry["path"]
    _verify_file(path, entry["sha256"], entry["bytes"])
    with safe_open(path, framework="pt", device="cpu") as handle:
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        metadata = handle.metadata()
    geometry = validate_payload(tensors, metadata, layer=entry["layer"], tp_rank=entry["rank"],
                                expected_design_sha256=manifest["source_design_sha256"])
    print("bridge_runtime_compatible", entry["rank"], geometry, entry["sha256"])
print("gpu_executed=false")
