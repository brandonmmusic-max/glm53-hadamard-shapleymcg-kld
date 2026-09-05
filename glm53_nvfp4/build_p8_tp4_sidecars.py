"""Shard one physical P8 layer stream for tensor-parallel serving."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .shard_index import sha256_file


def _expert_bases(layer: int, start: int, stop: int) -> list[str]:
    return [
        f"model.language_model.layers.{layer}.mlp.experts.{expert}"
        for expert in range(start, stop)
    ]


def _load_rank(
    chunks: list[Path], *, layer: int, rank: int, world_size: int
) -> tuple[
    dict[str, torch.Tensor],
    list[dict[str, object]],
    str,
    str | None,
    str,
    float | None,
]:
    gate_payload: list[torch.Tensor] = []
    up_payload: list[torch.Tensor] = []
    down_payload: list[torch.Tensor] = []
    gate_scales: list[torch.Tensor] = []
    up_scales: list[torch.Tensor] = []
    down_scales: list[torch.Tensor] = []
    sources: list[dict[str, object]] = []
    source_schemas: set[str] = set()
    source_designs: set[str] = set()
    source_boundaries: set[str] = set()
    source_angles: set[float] = set()
    expected_start = 0
    for chunk in chunks:
        with safe_open(chunk, framework="pt", device="cpu") as src:
            metadata = src.metadata() or {}
            schema = metadata.get("schema", "")
            source_schemas.add(schema)
            boundary = metadata.get("boundary", "")
            source_boundaries.add(boundary)
            if metadata.get("angle_pi") is not None:
                source_angles.add(float(metadata["angle_pi"]))
            if metadata.get("design_sha256"):
                source_designs.add(metadata["design_sha256"])
            rotated = schema == "glm53-rotated-hessian-trellis-p8-layer-chunk.v1"
            if (
                schema not in {
                    "glm53-p8-identity-mcg-layer-chunk.v1",
                    "glm53-hessian-trellis-p8-layer-chunk.v2",
                    "glm53-rotated-hessian-trellis-p8-layer-chunk.v1",
                }
                or metadata.get("role") != "physical-codec"
                or metadata.get("bits") != "4"
                or metadata.get("alphabet") != "e4m3"
                or metadata.get("block_size") != "32"
                or metadata.get("scale") not in {"ue8m0", "ue8m0-k32"}
                or metadata.get("law") != "procedural-mcg-alpha2"
                or boundary
                != ("shared-mid-butterfly-p00625" if rotated else "identity")
                or (
                    rotated
                    and (
                        float(metadata.get("angle_pi", "nan")) != 0.0625
                        or metadata.get("rotation_arithmetic")
                        != "bf16-input-fp32-four-stage-final-bf16"
                    )
                )
                or metadata.get("ldlq") != "false"
                or int(metadata.get("layer", -1)) != layer
            ):
                raise RuntimeError(f"physical P8 metadata mismatch: {chunk}")
            start_text, stop_text = metadata["expert_range"].split(":")
            start, stop = int(start_text), int(stop_text)
            if start != expected_start:
                raise RuntimeError(
                    f"non-contiguous expert chunks: expected {expected_start}, got {start}"
                )
            expected_start = stop
            for base in _expert_bases(layer, start, stop):
                gate = src.get_tensor(f"{base}.gate_proj.trellis")
                up = src.get_tensor(f"{base}.up_proj.trellis")
                down = src.get_tensor(f"{base}.down_proj.trellis")
                gate_sf = src.get_tensor(f"{base}.gate_proj.scale_ue8m0")
                up_sf = src.get_tensor(f"{base}.up_proj.scale_ue8m0")
                down_sf = src.get_tensor(f"{base}.down_proj.scale_ue8m0")
                if gate.shape[1] % world_size or down.shape[0] % world_size:
                    raise RuntimeError("trellis tile geometry is not TP divisible")
                if gate_sf.shape[0] % world_size or down_sf.shape[1] % world_size:
                    raise RuntimeError("scale geometry is not TP divisible")
                gate_n16 = gate.shape[1] // world_size
                down_k16 = down.shape[0] // world_size
                gate_rows = gate_sf.shape[0] // world_size
                down_k32 = down_sf.shape[1] // world_size
                gate_payload.append(
                    gate[:, rank * gate_n16 : (rank + 1) * gate_n16].contiguous()
                )
                up_payload.append(
                    up[:, rank * gate_n16 : (rank + 1) * gate_n16].contiguous()
                )
                down_payload.append(
                    down[rank * down_k16 : (rank + 1) * down_k16].contiguous()
                )
                gate_scales.append(
                    gate_sf[rank * gate_rows : (rank + 1) * gate_rows].contiguous()
                )
                up_scales.append(
                    up_sf[rank * gate_rows : (rank + 1) * gate_rows].contiguous()
                )
                down_scales.append(
                    down_sf[:, rank * down_k32 : (rank + 1) * down_k32].contiguous()
                )
        sources.append(
            {"path": str(chunk.resolve()), "bytes": chunk.stat().st_size, "sha256": sha256_file(chunk)}
        )
    if expected_start != 288:
        raise RuntimeError(f"expected 288 experts, found {expected_start}")
    if len(source_schemas) != 1:
        raise RuntimeError(f"mixed P8 chunk schemas are forbidden: {sorted(source_schemas)}")
    if len(source_boundaries) != 1 or len(source_angles) > 1:
        raise RuntimeError("mixed P8 boundary/angle chunks are forbidden")
    source_schema = next(iter(source_schemas))
    if source_schema in {
        "glm53-hessian-trellis-p8-layer-chunk.v2",
        "glm53-rotated-hessian-trellis-p8-layer-chunk.v1",
    }:
        if len(source_designs) != 1:
            raise RuntimeError(
                f"v2 P8 chunks require one immutable design: {sorted(source_designs)}"
            )
        source_design = next(iter(source_designs))
    else:
        if source_designs:
            raise RuntimeError("historical v1 chunks may not mix an unbound design field")
        source_design = None
    tensors = {
        "w13_trellis": torch.stack(
            (torch.stack(gate_payload), torch.stack(up_payload)), dim=0
        ).contiguous(),
        "w2_trellis": torch.stack(down_payload).contiguous(),
        # The B12X FC1 SFB repack consumes runtime row order [up; gate].
        "w13_scale_ue8m0": torch.cat(
            (torch.stack(up_scales), torch.stack(gate_scales)), dim=1
        ).contiguous(),
        "w2_scale_ue8m0": torch.stack(down_scales).contiguous(),
    }
    source_boundary = next(iter(source_boundaries))
    source_angle = next(iter(source_angles)) if source_angles else None
    return tensors, sources, source_schema, source_design, source_boundary, source_angle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--world-size", type=int, default=4)
    args = parser.parse_args()
    if args.world_size != 4:
        raise ValueError("this frozen product currently targets TP4")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = args.receipt or args.output_dir / "receipt.json"
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite {receipt_path}")
    receipt: dict[str, object] = {
        "schema": "glm53-p8-mcg-tp4-sidecars.v2",
        "layer": args.layer,
        "world_size": args.world_size,
        "bits": 4,
        "physical_bpw": 4.25,
        "law": "procedural-mcg-alpha2",
        "alphabet": "e4m3",
        "scale": "ue8m0-k32",
        "boundary": None,
        "angle_pi": None,
        "ldlq": False,
        "ranks": [],
    }
    for rank in range(args.world_size):
        tensors, sources, source_schema, source_design, source_boundary, source_angle = _load_rank(
            args.chunk, layer=args.layer, rank=rank, world_size=args.world_size
        )
        if receipt["boundary"] is None:
            receipt["boundary"] = source_boundary
            receipt["angle_pi"] = source_angle
        elif receipt["boundary"] != source_boundary or receipt["angle_pi"] != source_angle:
            raise RuntimeError("TP ranks disagree on P8 boundary/angle")
        output = args.output_dir / f"p8-layer-{args.layer:03d}-tp4-rank-{rank}.safetensors"
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        logical_elements = 32 * (
            tensors["w13_scale_ue8m0"].numel()
            + tensors["w2_scale_ue8m0"].numel()
        )
        payload_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in tensors.values()
        )
        payload_bpw = payload_bytes * 8.0 / logical_elements
        if payload_bpw != 4.25:
            raise RuntimeError(f"rank {rank} payload is {payload_bpw} bpw, expected 4.25")
        save_file(
            tensors,
            output,
            metadata={
                "schema": "glm53-p8-mcg-tp4-rank.v2",
                "source_schema": source_schema,
                "source_design_sha256": source_design or "historical-unbound-v1",
                "layer": str(args.layer),
                "rank": str(rank),
                "world_size": str(args.world_size),
                "bits": "4",
                "alphabet": "e4m3",
                "scale": "ue8m0-k32",
                "law": "procedural-mcg-alpha2",
                "boundary": source_boundary,
                "angle_pi": "none" if source_angle is None else format(source_angle, ".17g"),
                "ldlq": "false",
            },
        )
        receipt["ranks"].append(
            {
                "rank": rank,
                "path": str(output.resolve()),
                "bytes": output.stat().st_size,
                "sha256": sha256_file(output),
                "logical_elements": logical_elements,
                "payload_bytes": payload_bytes,
                "payload_bpw": payload_bpw,
                "file_bpw_including_container": output.stat().st_size * 8.0 / logical_elements,
                "shapes": {name: list(value.shape) for name, value in tensors.items()},
                "source_schema": source_schema,
                "source_design_sha256": source_design,
                "sources": sources,
            }
        )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
