"""Resumable BF16 -> native-RNE P4 matrix encoding for GLM experts.

Writes one matrix immediately after its checked encode. All source shard hashes
must be pinned by the design; each is hashed once per process. This is a real
CUDA Viterbi entry point and is not launched by CPU bridge verification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from .capture import LayerCapture
from .output_aware import route_weighted_hessian
from .p4_codec import read_p4, serialize_p4, storage_accounting
from .p4_encoder import encode_p4_matrix, payload_reconstruction, qdq_p4_activations
from .p4_serving_codec import LAW, MATRIX_MANIFEST_SCHEMA, PROJECTIONS, file_sha256
from .shard_index import IndexedCheckpoint


def verify_capture_files(paths: dict[str, Path], expected: dict) -> dict:
    """Verify the exact materialized fit files, not just their manifest/length.

    Sparse local captures may contain only the fit windows; their hashes must
    therefore be pinned independently of any remote/full capture manifest.
    Each file is hashed once before CUDA transfer or encoder selection.
    """
    names = {"hidden_bf16", "topk_ids_u16le", "topk_weights_f32le"}
    if set(paths) != names or not isinstance(expected, dict) or set(expected) != names:
        raise ValueError("P4 requires exact pins for all three materialized capture files")
    actual = {}
    for name in sorted(names):
        path = Path(paths[name]).resolve()
        pin = expected[name]
        if (not isinstance(pin, dict) or set(pin) != {"path", "bytes", "sha256"}
                or pin["path"] != str(path) or type(pin["bytes"]) is not int
                or not isinstance(pin["sha256"], str) or len(pin["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in pin["sha256"])):
            raise ValueError(f"invalid exact capture path/byte/hash pin: {name}")
        if not path.is_file() or path.stat().st_size != pin["bytes"]:
            raise ValueError(f"materialized capture byte count mismatch: {name}")
        digest = file_sha256(path)
        if digest != pin["sha256"]:
            raise ValueError(f"materialized capture SHA-256 mismatch: {name}")
        actual[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}
    return actual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--expert-start", type=int, default=0)
    parser.add_argument("--expert-end", type=int, default=288)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    design = json.loads(args.design.read_text())
    if (design.get("schema") != "glm53-p4-viterbi-rne-encoder-design.v1"
            or design.get("law") != LAW or design.get("ldlq") is not False
            or design.get("objective") != "viterbi-with-full-hessian-gptq-feedback"
            or design.get("source_precision") != "BF16"
            or design.get("source_index_sha256") != file_sha256(args.source_index)
            or args.layer not in design.get("layers", [])):
        raise ValueError("P4 encoder requires a pinned BF16 native-RNE no-LDLQ design")
    if (design.get("fit_roles_sha256") != file_sha256(args.roles)
            or design.get("capture_manifest_sha256") != file_sha256(args.capture_root / "capture-manifest.json")
            or design.get("data_role") != "fit" or design.get("sampling_strategy") != "domain-balanced"):
        raise ValueError("P4 encoder requires pinned fit-only domain-balanced REAP calibration")
    experts, hidden, intermediate = design["experts"], design["hidden"], design["intermediate"]
    if experts != 288 or not 3 <= args.layer <= 44 or not 0 <= args.expert_start < args.expert_end <= experts:
        raise ValueError("invalid GLM layer/expert range")
    if hidden % 64 or intermediate % 256:
        raise ValueError("GLM dimensions cannot reach TP4 P4 K64")
    design_sha = file_sha256(args.design)
    source = IndexedCheckpoint(args.source, args.source_index)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "matrices.json"
    if manifest_path.exists() and not args.resume:
        raise FileExistsError("refusing to overwrite completed P4 matrix manifest")
    manifest = {"schema": MATRIX_MANIFEST_SCHEMA, "law": LAW, "ldlq": False,
                "source_design_sha256": design_sha, "layer": args.layer,
                "experts": experts, "hidden": hidden, "intermediate": intermediate,
                "expert_range": [args.expert_start, args.expert_end], "matrices": []}
    verified_shards = set()
    device = torch.device(args.device)
    capture = LayerCapture(args.capture_root, args.layer, args.roles,
                           max_samples=design["samples"], sample_offset=0,
                           data_role="fit", sampling_strategy="domain-balanced")
    capture_files = verify_capture_files({"hidden_bf16": capture.hidden_path,
        "topk_ids_u16le": capture.ids_path, "topk_weights_f32le": capture.weights_path},
        design.get("capture_files", {}))
    started = time.perf_counter()
    for expert in range(args.expert_start, args.expert_end):
        base = source.expert_prefix(args.layer, expert)
        hidden_cpu, route_cpu = capture.samples(expert)
        calibrated_hidden = qdq_p4_activations(hidden_cpu.to(device))
        route = route_cpu.to(device)
        hidden_hessian = route_weighted_hessian(calibrated_hidden, route)
        expert_payloads = {}
        for projection in PROJECTIONS:
            stage_start = time.perf_counter()
            name = base + projection + ".weight"
            shard = source.weight_map[name]
            if shard not in verified_shards:
                actual = file_sha256(args.source / shard)
                if actual != design.get("source_file_sha256", {}).get(shard):
                    raise ValueError(f"BF16 shard not pinned by P4 design: {shard}")
                verified_shards.add(shard)
            filename = f"layer-{args.layer:03d}-expert-{expert:03d}-{projection}.p4.safetensors"
            target, receipt_path = args.output_dir / filename, args.output_dir / (filename + ".json")
            shape = (hidden, intermediate) if projection == "down_proj" else (intermediate, hidden)
            if target.exists() or receipt_path.exists():
                if not args.resume or not target.exists() or not receipt_path.exists():
                    raise FileExistsError("existing matrix requires complete resume receipt")
                receipt = json.loads(receipt_path.read_text())
                if (receipt.get("source_design_sha256") != design_sha or receipt.get("source_tensor") != name
                        or receipt.get("source_shard_sha256") != design["source_file_sha256"][shard]
                        or receipt.get("capture_files") != capture_files):
                    raise ValueError("P4 matrix resume source mismatch")
                entry = receipt["entry"]
                restored = read_p4(target, expected_sha256=entry["sha256"], expected_shape=shape)
                expected = {"expert": expert, "projection": projection, "path": filename,
                            "bytes": target.stat().st_size, "sha256": file_sha256(target)}
                if entry != expected or receipt["storage"] != storage_accounting(restored):
                    raise ValueError("P4 matrix resume receipt mismatch")
                manifest["matrices"].append(entry)
                expert_payloads[projection] = restored
                print(json.dumps({"expert": expert, "projection": projection, "status": "verified-reuse"}), flush=True)
                continue
            raw_weight = source.get(name)
            if raw_weight.dtype != torch.bfloat16 or tuple(raw_weight.shape) != shape:
                raise ValueError(f"source must be actual BF16 with shape {shape}: {name}")
            source_read_seconds = time.perf_counter() - stage_start
            encode_start = time.perf_counter()
            if projection == "down_proj":
                gate = F.linear(calibrated_hidden, payload_reconstruction(expert_payloads["gate_proj"], device=device)).clamp(max=10.)
                up = F.linear(calibrated_hidden, payload_reconstruction(expert_payloads["up_proj"], device=device)).clamp(-10., 10.)
                middle = qdq_p4_activations(F.silu(gate) * up)
                hessian = route_weighted_hessian(middle, route)
            else:
                hessian = hidden_hessian
            encoded = encode_p4_matrix(raw_weight.to(device), hessian,
                scale_refinement_iterations=design["scale_refinement_iterations"],
                search_grid=design["search_grid"], tailbite_context=design["tailbite_context"],
                percdamp=design["percdamp"], column_block=design["column_block"])
            torch.cuda.synchronize(device)
            encode_seconds = time.perf_counter() - encode_start
            data = serialize_p4(encoded.payload)
            with target.open("xb") as stream:
                stream.write(data)
            entry = {"expert": expert, "projection": projection, "path": filename,
                     "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            receipt = {"schema": "glm53-p4-encoded-matrix-receipt.v1", "source_design_sha256": design_sha,
                       "source_tensor": name, "source_shard_sha256": design["source_file_sha256"][shard],
                       "capture_files": capture_files,
                       "entry": entry, "storage": storage_accounting(encoded.payload), "encoder": encoded.report,
                       "timing": {"source_read_verify_seconds": source_read_seconds, "encode_seconds": encode_seconds,
                                  "total_seconds": time.perf_counter() - stage_start}}
            with receipt_path.open("x") as stream:
                stream.write(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
            manifest["matrices"].append(entry)
            expert_payloads[projection] = encoded.payload
            print(json.dumps({"expert": expert, "projection": projection, "status": "encoded",
                              "weight_nmse": encoded.report["history"][-1]["weight_nmse"],
                              "seconds": time.perf_counter() - stage_start}), flush=True)
            del raw_weight, encoded, data
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if manifest_path.exists():
        if manifest_path.read_text() != text:
            raise ValueError("resumed P4 manifest differs from completed artifact")
    else:
        with manifest_path.open("x") as stream:
            stream.write(text)
    print(json.dumps({"manifest": str(manifest_path), "elapsed_seconds": time.perf_counter() - started,
                      "verified_source_shards": len(verified_shards)}), flush=True)


if __name__ == "__main__":
    main()
