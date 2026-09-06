"""Bits-aware gates and manifest for the mixed-rate (K3/K4/K5 per layer) coupled P8 checkpoint.

Sibling of ``full_coupled_build_support`` for the Phase 1 assembly.  Every layer
carries one stored trellis rate; the rate is read from each rank's safetensors
metadata, checked against the allocation, and the weight payload must equal
exactly ``bits + 0.25`` bpw.  The manifest records exact bytes per layer and in
total, the allocation it installs, and whether the total stays inside the
identity K4 byte budget.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .full_coupled_build_support import (
    COUPLED_BOUNDARY,
    COUPLED_RANK_SCHEMA,
    COUPLED_SIDECAR_SCHEMA,
    LAYERS,
    METADATA_TENSORS,
    POSTWRITE_SCHEMA_PREFIX,
    WEIGHT_TENSORS,
    WORLD_SIZE,
    rank_path,
    safetensors_header,
    sha256_file,
    utc_now,
)

MANIFEST_SCHEMA = "glm53-p8-mixed-rate-coupled-all42-tp4-checkpoint.v1"
RATES = (3, 4, 5)
IDENTITY_FILE_BYTES = 161_715_707_328


def inspect_rank(path: Path, *, layer: int, rank: int, bits: int, allowed_design_sha256: set[str]) -> dict:
    metadata, tensors, header_bytes = safetensors_header(path)
    if int(bits) not in RATES:
        raise ValueError("rate must be K3, K4 or K5")
    required = {
        "schema": COUPLED_RANK_SCHEMA, "layer": str(layer), "rank": str(rank), "world_size": str(WORLD_SIZE),
        "bits": str(bits), "alphabet": "e4m3", "scale": "ue8m0-k32", "law": "procedural-mcg-alpha2",
        "boundary": COUPLED_BOUNDARY, "full_coupled": "true", "ldlq": "false",
    }
    mismatched = {key: metadata.get(key) for key, value in required.items() if metadata.get(key) != value}
    if mismatched:
        raise RuntimeError(f"rank metadata mismatch {path}: {mismatched}")
    if float(metadata.get("weight_payload_bpw", "nan")) != bits + 0.25:
        raise RuntimeError(f"rank {path} declares weight_payload_bpw {metadata.get('weight_payload_bpw')} for K{bits}")
    design = metadata.get("source_design_sha256", "")
    if design not in allowed_design_sha256:
        raise RuntimeError(f"rank {path} carries design {design!r} outside the allowlist")
    if set(tensors) != set(WEIGHT_TENSORS) | set(METADATA_TENSORS):
        raise RuntimeError(f"rank {path} tensor set differs")
    words = 16 * bits
    if tensors["w13_trellis"]["shape"][-1] != words or tensors["w2_trellis"]["shape"][-1] != words:
        raise RuntimeError(f"rank {path} stream words differ from K{bits}")

    def nbytes(name: str) -> int:
        start, stop = tensors[name]["data_offsets"]
        return int(stop) - int(start)

    def numel(name: str) -> int:
        count = 1
        for dim in tensors[name]["shape"]:
            count *= int(dim)
        return count

    weight_bytes = sum(nbytes(name) for name in WEIGHT_TENSORS)
    metadata_bytes = sum(nbytes(name) for name in METADATA_TENSORS)
    logical = 32 * (numel("w13_scale_ue8m0") + numel("w2_scale_ue8m0"))
    if weight_bytes * 32 != (4 * bits + 1) * logical:
        raise RuntimeError(f"rank {path} weight payload is not exactly K{bits}+0.25 bpw")
    file_bytes = path.stat().st_size
    if file_bytes != header_bytes + weight_bytes + metadata_bytes:
        raise RuntimeError(f"rank {path} file size differs from header + tensors")
    return {"path": str(path.resolve()), "rank": rank, "bits": bits, "bytes": file_bytes, "header_bytes": header_bytes,
            "weight_payload_bytes": weight_bytes, "metadata_bytes": metadata_bytes, "logical_elements": logical,
            "source_design_sha256": design, "exl3_scale_source_sha256": metadata.get("exl3_scale_source_sha256")}


def verify_layer(*, layer: int, bits: int, sidecars: Path, packer_receipt: Path, postwrite_receipt: Path,
                 allowed_design_sha256: set[str], rehash: bool = True) -> dict:
    packer = json.loads(Path(packer_receipt).read_text())
    if (packer.get("schema") != COUPLED_SIDECAR_SCHEMA or packer.get("layer") != layer or packer.get("world_size") != WORLD_SIZE
            or packer.get("boundary") != COUPLED_BOUNDARY or packer.get("ldlq") is not False
            or packer.get("weight_payload_bpw") != bits + 0.25 or packer.get("bits", 4) != bits):
        raise RuntimeError(f"packer receipt differs from K{bits} for layer {layer}: {packer_receipt}")
    ranks = packer.get("ranks")
    if not isinstance(ranks, list) or [row.get("rank") for row in ranks] != list(range(WORLD_SIZE)):
        raise RuntimeError(f"packer receipt lacks ranks 0..3 exactly once: {packer_receipt}")
    postwrite = json.loads(Path(postwrite_receipt).read_text())
    if (not str(postwrite.get("schema", "")).startswith(POSTWRITE_SCHEMA_PREFIX) or postwrite.get("layer") != layer
            or postwrite.get("status") != "pass"):
        raise RuntimeError(f"postwrite receipt is not a layer-{layer} pass: {postwrite_receipt}")
    rows, designs = [], set()
    for entry in ranks:
        rank = int(entry["rank"])
        path = rank_path(sidecars, layer, rank)
        if not path.is_file():
            raise RuntimeError(f"missing rank sidecar {path}")
        info = inspect_rank(path, layer=layer, rank=rank, bits=bits, allowed_design_sha256=allowed_design_sha256)
        if info["bytes"] != int(entry["bytes"]):
            raise RuntimeError(f"rank {path} bytes differ from packer receipt")
        digest = sha256_file(path) if rehash else entry["sha256"]
        if digest != entry["sha256"]:
            raise RuntimeError(f"rank {path} sha256 differs from packer receipt")
        info["sha256"] = digest
        if info["source_design_sha256"] != entry.get("source_design_sha256"):
            raise RuntimeError(f"rank {path} design hash differs from packer receipt")
        designs.add(info["source_design_sha256"])
        rows.append(info)
    if len(designs) != 1:
        raise RuntimeError(f"layer {layer} ranks carry different design hashes")
    return {"layer": layer, "bits": bits, "status": "pass", "ranks": rows, "source_design_sha256": designs.pop(),
            "packer_receipt": {"path": str(Path(packer_receipt).resolve()), "sha256": sha256_file(packer_receipt)},
            "postwrite_receipt": {"path": str(Path(postwrite_receipt).resolve()), "sha256": sha256_file(postwrite_receipt)},
            "bytes": sum(r["bytes"] for r in rows), "weight_payload_bytes": sum(r["weight_payload_bytes"] for r in rows),
            "metadata_bytes": sum(r["metadata_bytes"] for r in rows), "logical_elements": sum(r["logical_elements"] for r in rows)}


def load_allocation(path: Path) -> dict[int, int]:
    value = json.loads(Path(path).read_text())
    if value.get("schema") != "glm53.p8-layer-rate-allocation.v1":
        raise ValueError("allocation schema differs")
    assignment = {int(k): int(v) for k, v in value["assignment"].items()}
    if set(assignment) != set(LAYERS) or any(v not in RATES for v in assignment.values()):
        raise ValueError("allocation does not assign K3/K4/K5 to exactly layers 3..44")
    return assignment


def cmd_layer_complete(args: argparse.Namespace) -> dict:
    assignment = load_allocation(args.allocation)
    return verify_layer(layer=args.layer, bits=assignment[args.layer], sidecars=args.sidecars,
                        packer_receipt=args.packer_receipt, postwrite_receipt=args.postwrite_receipt,
                        allowed_design_sha256=set(args.allow_design_sha256), rehash=not args.trust_receipt_hashes)


def cmd_manifest(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    assignment = load_allocation(args.allocation)
    allowed = set(args.allow_design_sha256)
    layers, totals = [], {"bytes": 0, "weight_payload_bytes": 0, "metadata_bytes": 0, "logical_elements": 0}
    for layer in LAYERS:
        receipts = Path(args.receipts)
        row = verify_layer(layer=layer, bits=assignment[layer], sidecars=args.sidecars,
                           packer_receipt=receipts / f"layer-{layer:03d}-sidecars.json",
                           postwrite_receipt=receipts / f"layer-{layer:03d}-postwrite.json",
                           allowed_design_sha256=allowed, rehash=not args.trust_receipt_hashes)
        reuse = receipts / f"layer-{layer:03d}-reuse.json"
        if reuse.is_file():
            row["reuse_receipt"] = {"path": str(reuse.resolve()), "sha256": sha256_file(reuse)}
        for key in totals:
            totals[key] += row[key]
        layers.append(row)
    designs: dict[str, list[int]] = {}
    for row in layers:
        designs.setdefault(row["source_design_sha256"], []).append(row["layer"])
    counts = {str(rate): sum(1 for layer in LAYERS if assignment[layer] == rate) for rate in RATES}
    stored_bpw = (totals["weight_payload_bytes"] + totals["metadata_bytes"]) * 8 / totals["logical_elements"]
    manifest = {
        "schema": MANIFEST_SCHEMA, "created_at": utc_now(), "layer_range": [LAYERS[0], LAYERS[-1]], "world_size": WORLD_SIZE,
        "codec": {"alphabet": "e4m3", "bits": "per-layer 3/4/5", "law": "procedural-mcg-alpha2", "ldlq": False,
                  "scale": "ue8m0-k32", "boundary": COUPLED_BOUNDARY},
        "allocation": {"path": str(Path(args.allocation).resolve()), "sha256": sha256_file(args.allocation),
                       "assignment": {str(layer): assignment[layer] for layer in LAYERS}, "counts": counts},
        "designs": {digest: sorted(rows) for digest, rows in designs.items()},
        "transform": {"path": str(Path(args.transform).resolve()), "sha256": sha256_file(args.transform)},
        "layers": layers,
        "payload": {"weight_payload_bytes": totals["weight_payload_bytes"],
                    "weight_payload_bpw": totals["weight_payload_bytes"] * 8 / totals["logical_elements"],
                    "coupled_metadata_bytes": totals["metadata_bytes"],
                    "safetensors_header_bytes": totals["bytes"] - totals["weight_payload_bytes"] - totals["metadata_bytes"],
                    "file_bytes": totals["bytes"], "stored_bpw_including_metadata": stored_bpw,
                    "logical_elements": totals["logical_elements"],
                    "identity_k4_file_bytes": IDENTITY_FILE_BYTES,
                    "bytes_under_identity_files": IDENTITY_FILE_BYTES - totals["bytes"],
                    "same_size_or_smaller": totals["bytes"] <= IDENTITY_FILE_BYTES},
        "protected_boundary": "fit capture only; no selection, confirmation, final, or reserved confirmation logits opened",
        "isa_cost": "P8 E4M3 mxf8f6f4 has twice the NVFP4 MMA issue count; no speed claim",
    }
    if not manifest["payload"]["same_size_or_smaller"] and not args.allow_larger:
        raise RuntimeError("mixed-rate checkpoint exceeds the identity K4 file bytes")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"manifest": str(output.resolve()), "sha256": sha256_file(output), "payload": manifest["payload"], "counts": counts}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    complete = sub.add_parser("layer-complete")
    complete.add_argument("--layer", type=int, required=True, choices=LAYERS)
    complete.add_argument("--allocation", type=Path, required=True)
    complete.add_argument("--sidecars", type=Path, required=True)
    complete.add_argument("--packer-receipt", type=Path, required=True)
    complete.add_argument("--postwrite-receipt", type=Path, required=True)
    complete.add_argument("--allow-design-sha256", action="append", required=True)
    complete.add_argument("--trust-receipt-hashes", action="store_true")
    complete.set_defaults(func=cmd_layer_complete)
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--allocation", type=Path, required=True)
    manifest.add_argument("--sidecars", type=Path, required=True)
    manifest.add_argument("--receipts", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--transform", type=Path, required=True)
    manifest.add_argument("--allow-design-sha256", action="append", required=True)
    manifest.add_argument("--trust-receipt-hashes", action="store_true")
    manifest.add_argument("--allow-larger", action="store_true")
    manifest.set_defaults(func=cmd_manifest)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = args.func(args)
    except (RuntimeError, FileExistsError, FileNotFoundError, ValueError) as error:
        print(json.dumps({"status": "fail", "command": args.command, "error": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
