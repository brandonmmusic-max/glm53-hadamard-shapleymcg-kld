"""Exact payload bytes of an EXL3 checkpoint from safetensors headers (no tensor data read).

Routed-expert payload = every tensor under model.layers.*.mlp.experts.* (trellis + suh/svh + scale +
mcg/mul1 flags).  Total = all tensors.  Also returns the per-projection breakdown and bits per weight
against the BF16 element counts.
"""
from __future__ import annotations

import json
import re
import struct
from pathlib import Path

DTYPE_BYTES = {"F16": 2, "BF16": 2, "F32": 4, "I16": 2, "I8": 1, "U8": 1, "I32": 4, "I64": 8, "F8_E4M3": 1, "BOOL": 1}
EXPERT_RE = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(.+)")


def read_header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def checkpoint_bytes(model_dir: Path, expert_elements: dict[str, int] | None = None) -> dict:
    total = 0
    routed = 0
    per_proj = {"gate_proj": 0, "up_proj": 0, "down_proj": 0}
    per_kind = {}
    n_expert_tensors = 0
    layers = set()
    for shard in sorted(model_dir.glob("*.safetensors")):
        hdr = read_header(shard)
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            shape = meta["shape"]
            nbytes = DTYPE_BYTES[meta["dtype"]]
            for s in shape:
                nbytes *= s
            total += nbytes
            m = EXPERT_RE.match(name)
            if m:
                routed += nbytes
                per_proj[m.group(3)] += nbytes
                per_kind[m.group(4)] = per_kind.get(m.group(4), 0) + nbytes
                n_expert_tensors += 1
                layers.add(int(m.group(1)))
    out = {"total_bytes": total, "routed_expert_payload_bytes": routed, "per_projection_bytes": per_proj, "per_tensor_kind_bytes": per_kind,
           "expert_tensors": n_expert_tensors, "layers": len(layers)}
    if expert_elements:
        elems = sum(expert_elements.values())
        out["routed_expert_bpw"] = 8.0 * routed / elems
        out["per_projection_bpw"] = {p: 8.0 * per_proj[p] / expert_elements[p] for p in per_proj}
    return out


if __name__ == "__main__":
    import sys
    d = Path(sys.argv[1])
    # Qwen3-30B-A3B: 48 layers x 128 experts; gate/up 768x2048, down 2048x768
    elems = {p: 48 * 128 * 768 * 2048 for p in ("gate_proj", "up_proj", "down_proj")}
    print(json.dumps(checkpoint_bytes(d, elems), indent=1))
