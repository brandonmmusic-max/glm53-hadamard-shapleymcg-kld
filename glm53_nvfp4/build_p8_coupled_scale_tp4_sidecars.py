"""Shard coupled-scale K4 P8 chunks into fail-closed TP4 sidecars."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .p8_coupled_scale import (
    COUPLED_BOUNDARY,
    COUPLED_CHUNK_SCHEMA,
    COUPLED_RANK_SCHEMA,
    COUPLED_SIDECAR_SCHEMA,
    SUPPORTED_LAYERS,
    _tensor_sha256,
)
from .shard_index import sha256_file


def _base(layer: int, expert: int) -> str:
    return f"model.language_model.layers.{layer}.mlp.experts.{expert}"


def _load_rank_coupled(
    chunks: list[Path],
    *,
    layer: int,
    rank: int,
    world_size: int,
    expected_hidden: int | None = None,
    expected_intermediate: int | None = None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, object]], str, str]:
    if layer not in SUPPORTED_LAYERS or world_size != 4 or not 0 <= rank < world_size:
        raise ValueError("coupled sidecars are frozen to layers 3/20/22 and TP4")
    gate_payload: list[torch.Tensor] = []
    up_payload: list[torch.Tensor] = []
    down_payload: list[torch.Tensor] = []
    gate_sf: list[torch.Tensor] = []
    up_sf: list[torch.Tensor] = []
    down_sf: list[torch.Tensor] = []
    gate_svh: list[torch.Tensor] = []
    up_svh: list[torch.Tensor] = []
    down_suh: list[torch.Tensor] = []
    draws: list[torch.Tensor] = []
    shared_gate: torch.Tensor | None = None
    shared_down: torch.Tensor | None = None
    design_hashes: set[str] = set()
    scale_source_hashes: set[str] = set()
    sources: list[dict[str, object]] = []
    expected_start = 0
    hidden = intermediate = None

    for chunk in chunks:
        with safe_open(chunk, framework="pt", device="cpu") as src:
            metadata = src.metadata() or {}
            required = {
                "schema": COUPLED_CHUNK_SCHEMA,
                "role": "physical-codec",
                "layer": str(layer),
                "bits": "4",
                "alphabet": "e4m3",
                "block_size": "32",
                "scale": "ue8m0-k32",
                "law": "procedural-mcg-alpha2",
                "boundary": COUPLED_BOUNDARY,
                "activation": "clipped-silu10",
                "cast_order": "bf16-fp16-h512-fp16-suh-h128-e4m3",
                "ldlq": "false",
                "encoder": "gptq-feedback-static-in-group-act-order",
                "fc1_trellis_slot_order": "gate-up",
                "fc1_scale_plane_order": "up-gate",
                "coupled_scale_order": "gate_svh-up_svh-down_suh",
            }
            if any(metadata.get(key) != value for key, value in required.items()):
                raise RuntimeError(f"invalid coupled chunk metadata: {chunk}")
            design_hashes.add(metadata.get("design_sha256", ""))
            scale_source_hashes.add(metadata.get("exl3_scale_source_sha256", ""))
            start_text, stop_text = metadata.get("expert_range", ":").split(":")
            start, stop = int(start_text), int(stop_text)
            if start != expected_start or not start < stop <= 288:
                raise RuntimeError("coupled chunks are not contiguous from expert zero")
            expected_start = stop
            chunk_gate = src.get_tensor("coupled.gate_up_suh_fp16").contiguous()
            chunk_down = src.get_tensor("coupled.down_svh_fp16").contiguous()
            chunk_gate_svh = src.get_tensor("coupled.gate_svh_fp16").contiguous()
            chunk_up_svh = src.get_tensor("coupled.up_svh_fp16").contiguous()
            chunk_down_suh = src.get_tensor("coupled.down_suh_fp16").contiguous()
            chunk_draws = src.get_tensor("coupled.intermediate_draw_u8").contiguous()
            if hidden is None:
                hidden = int(chunk_gate.numel())
                intermediate = int(chunk_gate_svh.shape[1])
                if expected_hidden is not None and hidden != expected_hidden:
                    raise RuntimeError(
                        f"coupled hidden size {hidden}, expected {expected_hidden}"
                    )
                if (
                    expected_intermediate is not None
                    and intermediate != expected_intermediate
                ):
                    raise RuntimeError(
                        "coupled intermediate size "
                        f"{intermediate}, expected {expected_intermediate}"
                    )
            if (
                chunk_gate.dtype != torch.float16
                or tuple(chunk_gate.shape) != (hidden,)
                or chunk_down.dtype != torch.float16
                or tuple(chunk_down.shape) != (hidden,)
                or chunk_gate_svh.dtype != torch.float16
                or tuple(chunk_gate_svh.shape) != (stop - start, intermediate)
                or chunk_up_svh.dtype != torch.float16
                or tuple(chunk_up_svh.shape) != (stop - start, intermediate)
                or chunk_down_suh.dtype != torch.float16
                or tuple(chunk_down_suh.shape) != (stop - start, intermediate)
                or chunk_draws.dtype != torch.uint8
                or tuple(chunk_draws.shape) != (stop - start,)
                or bool((chunk_draws > 7).any())
            ):
                raise RuntimeError("coupled scale tensor geometry/dtype mismatch")
            for value in (chunk_gate, chunk_down, chunk_gate_svh, chunk_up_svh, chunk_down_suh):
                if not bool(torch.isfinite(value).all()) or bool((value == 0).any()):
                    raise RuntimeError("coupled scale tensors must be finite and nonzero")
            if shared_gate is None:
                shared_gate, shared_down = chunk_gate, chunk_down
            elif not torch.equal(shared_gate, chunk_gate) or not torch.equal(shared_down, chunk_down):
                raise RuntimeError("shared coupled scales changed across chunks")
            if intermediate % world_size:
                raise RuntimeError("intermediate scale geometry is not TP divisible")
            local_i = intermediate // world_size
            lo, hi = rank * local_i, (rank + 1) * local_i
            gate_svh.extend(chunk_gate_svh[:, lo:hi].unbind())
            up_svh.extend(chunk_up_svh[:, lo:hi].unbind())
            down_suh.extend(chunk_down_suh[:, lo:hi].unbind())
            draws.extend(chunk_draws.unbind())

            for expert in range(start, stop):
                base = _base(layer, expert)
                gate = src.get_tensor(f"{base}.gate_proj.trellis")
                up = src.get_tensor(f"{base}.up_proj.trellis")
                down = src.get_tensor(f"{base}.down_proj.trellis")
                gsf = src.get_tensor(f"{base}.gate_proj.scale_ue8m0")
                usf = src.get_tensor(f"{base}.up_proj.scale_ue8m0")
                dsf = src.get_tensor(f"{base}.down_proj.scale_ue8m0")
                if (
                    gate.dtype != torch.int16
                    or up.dtype != torch.int16
                    or down.dtype != torch.int16
                    or gate.shape != up.shape
                    or gate.shape[-1] != 64
                    or down.shape[-1] != 64
                    or gsf.dtype != torch.uint8
                    or usf.dtype != torch.uint8
                    or dsf.dtype != torch.uint8
                    or gsf.shape != usf.shape
                    or gate.shape[1] % world_size
                    or down.shape[0] % world_size
                    or gsf.shape[0] % world_size
                    or dsf.shape[1] % world_size
                ):
                    raise RuntimeError(f"invalid K4 coupled weight geometry for expert {expert}")
                gate_n16 = gate.shape[1] // world_size
                down_k16 = down.shape[0] // world_size
                gate_rows = gsf.shape[0] // world_size
                down_k32 = dsf.shape[1] // world_size
                gate_payload.append(gate[:, rank * gate_n16 : (rank + 1) * gate_n16].contiguous())
                up_payload.append(up[:, rank * gate_n16 : (rank + 1) * gate_n16].contiguous())
                down_payload.append(down[rank * down_k16 : (rank + 1) * down_k16].contiguous())
                gate_sf.append(gsf[rank * gate_rows : (rank + 1) * gate_rows].contiguous())
                up_sf.append(usf[rank * gate_rows : (rank + 1) * gate_rows].contiguous())
                down_sf.append(dsf[:, rank * down_k32 : (rank + 1) * down_k32].contiguous())
        sources.append(
            {"path": str(chunk.resolve()), "bytes": chunk.stat().st_size, "sha256": sha256_file(chunk)}
        )

    if expected_start != 288 or len(design_hashes) != 1 or len(scale_source_hashes) != 1:
        raise RuntimeError("coupled chunks do not form one complete immutable layer")
    design_sha256 = next(iter(design_hashes))
    scale_source_sha256 = next(iter(scale_source_hashes))
    if any(
        len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
        for value in (design_sha256, scale_source_sha256)
    ):
        raise RuntimeError("coupled chunks lack valid design/scale-source hashes")
    assert shared_gate is not None and shared_down is not None
    tensors = {
        "w13_trellis": torch.stack((torch.stack(gate_payload), torch.stack(up_payload))).contiguous(),
        "w2_trellis": torch.stack(down_payload).contiguous(),
        "w13_scale_ue8m0": torch.cat((torch.stack(up_sf), torch.stack(gate_sf)), dim=1).contiguous(),
        "w2_scale_ue8m0": torch.stack(down_sf).contiguous(),
        "gate_up_suh_fp16": shared_gate,
        "intermediate_scales_fp16": torch.cat(
            (torch.stack(gate_svh), torch.stack(up_svh), torch.stack(down_suh)), dim=1
        ).contiguous(),
        "down_svh_fp16": shared_down,
        "coupled_sign_draw_u8": torch.stack(draws).contiguous(),
    }
    return tensors, sources, design_sha256, scale_source_sha256


def _payload_rates(tensors: dict[str, torch.Tensor]) -> tuple[int, int, float, float]:
    logical_elements = 32 * (
        tensors["w13_scale_ue8m0"].numel() + tensors["w2_scale_ue8m0"].numel()
    )
    weight_names = ("w13_trellis", "w2_trellis", "w13_scale_ue8m0", "w2_scale_ue8m0")
    weight_bytes = sum(tensors[name].numel() * tensors[name].element_size() for name in weight_names)
    metadata_bytes = sum(
        value.numel() * value.element_size()
        for name, value in tensors.items()
        if name not in weight_names
    )
    return (
        weight_bytes,
        metadata_bytes,
        8.0 * weight_bytes / logical_elements,
        8.0 * metadata_bytes / logical_elements,
    )


def expected_metadata_bpw(
    *, hidden: int, intermediate: int, experts: int = 288, world_size: int = 4
) -> float:
    """Return the exact TP-local FP16-scale plus one-byte/draw overhead."""
    if min(hidden, intermediate, experts, world_size) <= 0 or intermediate % world_size:
        raise ValueError("invalid coupled metadata geometry")
    logical_elements = 3 * experts * hidden * intermediate // world_size
    metadata_bytes = 4 * hidden + 6 * experts * intermediate // world_size + experts
    return 8.0 * metadata_bytes / logical_elements


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--layer", type=int, required=True, choices=SUPPORTED_LAYERS)
    parser.add_argument("--world-size", type=int, default=4)
    args = parser.parse_args()
    if args.world_size != 4:
        raise ValueError("coupled scale product is TP4-only")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = args.receipt or args.output_dir / "receipt.json"
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite {receipt_path}")
    receipt: dict[str, object] = {
        "schema": COUPLED_SIDECAR_SCHEMA,
        "layer": args.layer,
        "world_size": 4,
        "boundary": COUPLED_BOUNDARY,
        "weight_payload_bpw": 4.25,
        "metadata_bpw": None,
        "stored_bpw": None,
        "ldlq": False,
        "ranks": [],
    }
    for rank in range(4):
        tensors, sources, design_sha256, scale_source_sha256 = _load_rank_coupled(
            args.chunk,
            layer=args.layer,
            rank=rank,
            world_size=4,
            expected_hidden=4096,
            expected_intermediate=2048,
        )
        weight_bytes, metadata_bytes, weight_bpw, metadata_bpw = _payload_rates(tensors)
        if not math.isclose(weight_bpw, 4.25, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"rank {rank} weight payload is {weight_bpw}, expected 4.25")
        if receipt["metadata_bpw"] is None:
            expected_rate = expected_metadata_bpw(hidden=4096, intermediate=2048)
            if not math.isclose(metadata_bpw, expected_rate, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(
                    f"rank {rank} metadata payload is {metadata_bpw}, "
                    f"expected {expected_rate}"
                )
            receipt["metadata_bpw"] = metadata_bpw
            receipt["stored_bpw"] = weight_bpw + metadata_bpw
        elif receipt["metadata_bpw"] != metadata_bpw:
            raise RuntimeError("TP ranks disagree on coupled metadata rate")
        output = args.output_dir / f"p8-layer-{args.layer:03d}-tp4-rank-{rank}.safetensors"
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        tensor_hashes = {name: _tensor_sha256(value) for name, value in tensors.items()}
        save_file(
            tensors,
            output,
            metadata={
                "schema": COUPLED_RANK_SCHEMA,
                "layer": str(args.layer),
                "rank": str(rank),
                "world_size": "4",
                "bits": "4",
                "alphabet": "e4m3",
                "scale": "ue8m0-k32",
                "law": "procedural-mcg-alpha2",
                "boundary": COUPLED_BOUNDARY,
                "activation": "clipped-silu10",
                "cast_order": "bf16-fp16-h512-fp16-suh-h128-e4m3",
                "ldlq": "false",
                "fc1_trellis_slot_order": "gate-up",
                "fc1_scale_plane_order": "up-gate",
                "coupled_scale_order": "gate_svh-up_svh-down_suh",
                "source_design_sha256": design_sha256,
                "exl3_scale_source_sha256": scale_source_sha256,
                "weight_payload_bpw": "4.25",
                "metadata_bpw": format(metadata_bpw, ".17g"),
            },
        )
        receipt["ranks"].append(
            {
                "rank": rank,
                "path": str(output.resolve()),
                "bytes": output.stat().st_size,
                "sha256": sha256_file(output),
                "weight_payload_bytes": weight_bytes,
                "metadata_bytes": metadata_bytes,
                "weight_payload_bpw": weight_bpw,
                "metadata_bpw": metadata_bpw,
                "stored_bpw": weight_bpw + metadata_bpw,
                "shapes": {name: list(value.shape) for name, value in tensors.items()},
                "tensor_sha256": tensor_hashes,
                "source_design_sha256": design_sha256,
                "exl3_scale_source_sha256": scale_source_sha256,
                "sources": sources,
            }
        )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
