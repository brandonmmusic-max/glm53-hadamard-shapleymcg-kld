"""Deterministic CPU FC1+FC2 fixtures for the six-tensor P4 serving bridge."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .p4_codec import (
    P4Payload, e2m1_codes, e2m1_values, e4m3_scale_values,
    pack_k4_edges, serialize_p4, unpack_e2m1,
)
from .p4_fixture import scalar_operands
from .p4_serving_codec import (
    LAW, MATRIX_MANIFEST_SCHEMA, PROJECTIONS, assemble_rank, export_tp4,
    file_sha256, rank_projection, read_rank,
)


DESIGN_SHA = hashlib.sha256(b"glm53-p4-native-rne-fc1-fc2-fixture-v2").hexdigest()


def make_experts(*, experts=3, hidden=128, intermediate=256):
    rng = np.random.default_rng(2026090402)
    result = []
    for expert in range(experts):
        projections = {}
        for p, name in enumerate(PROJECTIONS):
            n, k = (hidden, intermediate) if name == "down_proj" else (intermediate, hidden)
            edges = rng.integers(0, 16, (k // 16, n // 16, 256), dtype=np.uint8)
            scales = rng.integers(30, 65, (n, k // 16), dtype=np.uint8)
            projections[name] = P4Payload(n, k, pack_k4_edges(edges), scales,
                                          np.array([.0625, .125, .25][(expert + p) % 3], dtype="<f4"))
        result.append(projections)
    return result


def matrix_values(payload):
    # Independent scalar stream addressing and binary16 arithmetic oracle.
    operands = scalar_operands(payload)
    values = e2m1_values(unpack_e2m1(operands.weight)).astype(np.float64)
    return values * np.repeat(e4m3_scale_values(operands.scale_e4m3), 16, axis=1) * float(operands.global_scale)


def quantize_activations(x):
    """Independent float64 CPU implementation of native E2M1/E4M3 SFA."""
    blocks = x.reshape(*x.shape[:-1], -1, 16)
    maximum = np.abs(blocks).max(-1)
    targets = np.clip(maximum / 6, 2**-9, 448)
    levels = e4m3_scale_values(np.arange(1, 127, dtype=np.uint8))
    upper = np.searchsorted(levels, targets).clip(0, 125)
    lower = (upper - 1).clip(0, 125)
    dl, du = abs(targets - levels[lower]), abs(targets - levels[upper])
    chosen = np.where((du < dl) | ((du == dl) & (((upper + 1) & 1) == 0)), upper, lower)
    scales = np.where(maximum == 0, 0, levels[chosen])
    normalized = blocks / np.where(scales == 0, 1, scales)[..., None]
    return (e2m1_values(e2m1_codes(normalized)) * scales[..., None]).reshape(x.shape)


def inputs(*, hidden=128, experts=3):
    x = (((np.arange(6 * hidden).reshape(6, hidden) * 11) % 41) - 20).astype(np.float64) / 32
    ids = np.array([[token % experts, (token + 1) % experts] for token in range(6)], dtype=np.int64)
    weights = np.tile(np.array([.75, .25], dtype=np.float64), (6, 1))
    return x, ids, weights


def moe_outputs(matrices, x, ids, weights):
    qx = quantize_activations(x)
    result = np.zeros_like(x, dtype=np.float64)
    stages = {"gate_up": [], "middle": [], "down": []}
    for token, row in enumerate(ids):
        for slot, expert in enumerate(row):
            expert_matrices = matrices[int(expert)]
            gate = expert_matrices["gate_proj"] @ qx[token]
            up = expert_matrices["up_proj"] @ qx[token]
            g, u = np.minimum(gate, 10), np.clip(up, -10, 10)
            middle = (g / (1 + np.exp(-g))) * u
            down = expert_matrices["down_proj"] @ quantize_activations(middle)
            result[token] += down * weights[token, slot]
            stages["gate_up"].append(np.stack((gate, up)))
            stages["middle"].append(middle)
            stages["down"].append(down)
    return result, {key: np.stack(value) for key, value in stages.items()}


def write_matrix_fixture(output_dir: Path, experts=None):
    output_dir.mkdir(parents=True, exist_ok=False)
    experts = make_experts() if experts is None else experts
    hidden, intermediate = experts[0]["gate_proj"].width, experts[0]["gate_proj"].rows
    manifest = {"schema": MATRIX_MANIFEST_SCHEMA, "source_design_sha256": DESIGN_SHA,
                "law": LAW, "ldlq": False, "layer": 3, "experts": len(experts),
                "hidden": hidden, "intermediate": intermediate, "matrices": []}
    for e, projections in enumerate(experts):
        for name, payload in projections.items():
            filename = f"expert-{e:03d}-{name}.p4.safetensors"
            raw = serialize_p4(payload)
            with (output_dir / filename).open("xb") as stream:
                stream.write(raw)
            manifest["matrices"].append({"expert": e, "projection": name, "path": filename,
                                         "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
    path = output_dir / "matrices.json"
    with path.open("x") as stream:
        stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def fixture_expectations(experts):
    matrices = [{name: matrix_values(payload) for name, payload in item.items()} for item in experts]
    x, ids, weights = inputs(hidden=experts[0]["gate_proj"].width, experts=len(experts))
    output, stages = moe_outputs(matrices, x, ids, weights)
    rank_outputs = []
    for rank in range(4):
        tensors, metadata = assemble_rank(experts, layer=3, rank=rank, source_design_sha256=DESIGN_SHA)
        local = [{name: matrix_values(rank_projection(tensors, metadata, expert=e, projection=name))
                  for name in PROJECTIONS} for e in range(len(experts))]
        local_output, _ = moe_outputs(local, x, ids, weights)
        rank_outputs.append(local_output)
    combined = np.stack(rank_outputs).sum(0)
    if not np.array_equal(output, combined):
        raise ValueError(f"synthetic TP4 output mismatch: {abs(output - combined).max()}")
    return {"schema": "glm53-p4-serving-fixture-expected.v2", "source_design_sha256": DESIGN_SHA,
            "evidence_level": "structural", "law": LAW, "experts": len(experts),
            "hidden": experts[0]["gate_proj"].width, "intermediate": experts[0]["gate_proj"].rows,
            "input": x.tolist(), "route_ids": ids.tolist(), "route_weights": weights.tolist(),
            "output_f64le_hex": output.astype("<f8").tobytes().hex(),
            "output_sha256": hashlib.sha256(output.astype("<f8").tobytes()).hexdigest(),
            "stage_sha256": {key: hashlib.sha256(value.astype("<f8").tobytes()).hexdigest() for key, value in stages.items()},
            "tp4_sum_max_abs": float(abs(output - combined).max()),
            "gpu_execution": "Not tested", "kld": "Not tested", "speed": "Not tested"}


def verify_fixture(root: Path):
    expected = json.loads((root / "fixture.expected.json").read_text())
    experts = make_experts()
    actual = fixture_expectations(experts)
    if actual != expected:
        raise ValueError("P4 fixture expected CPU output mismatch")
    manifest = json.loads((root / "ranks" / "manifest.json").read_text())
    if len(manifest["entries"]) != 4:
        raise ValueError("fixture is missing a rank")
    for entry in manifest["entries"]:
        tensors, metadata = read_rank(root / "ranks" / entry["path"], expected_sha256=entry["sha256"],
            expected_layer=3, expected_rank=entry["rank"], expected_design_sha256=DESIGN_SHA, expected_experts=3)
        serial, serial_metadata = assemble_rank(experts, layer=3, rank=entry["rank"], source_design_sha256=DESIGN_SHA)
        if any(tensors[name].tobytes() != serial[name].tobytes() for name in serial):
            raise ValueError("fixture rank differs from independent matrix assembly")
        if any(metadata[key] != value for key, value in serial_metadata.items()):
            raise ValueError("fixture rank metadata mismatch")
    return {"evidence_level": "structural", "ranks_verified": 4, "projections_verified": 9,
            "tp4_sum_max_abs": actual["tp4_sum_max_abs"], "output_sha256": actual["output_sha256"],
            "manifest_sha256": file_sha256(root / "ranks" / "manifest.json")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        matrix_manifest = write_matrix_fixture(args.output_dir / "matrices")
        export_tp4(matrix_manifest, args.output_dir / "ranks", expected_design_sha256=DESIGN_SHA)
        expected = fixture_expectations(make_experts())
        with (args.output_dir / "fixture.expected.json").open("x") as stream:
            stream.write(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verify_fixture(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
