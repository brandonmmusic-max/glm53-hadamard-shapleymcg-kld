"""CPU-only exact-contract audit between the coupled P8 encoder and runtime."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

from glm53_nvfp4 import p8_coupled_scale as encoder
from glm53_nvfp4.build_p8_coupled_scale_tp4_sidecars import (
    _payload_rates,
    _rank_metadata,
)
from glm53_nvfp4.shard_index import sha256_file


PINNED_RUNTIME_SHA256 = (
    "5e9220740c30bfbcb47ab0ea0857f820562b95e6a0797ff0fb0868eb6f84a0f2"
)
PINNED_RUNTIME_COMMIT = "fe1df7a2695a975bce632a613a12756405ed2c84"


def _load_runtime(path: Path):
    if sha256_file(path) != PINNED_RUNTIME_SHA256:
        raise RuntimeError("runtime CPU contract differs from pinned fe1df7a")
    spec = importlib.util.spec_from_file_location("pinned_p8_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load runtime CPU contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _signed(length: int, offset: int = 0) -> torch.Tensor:
    values = torch.linspace(0.625, 1.375, length)
    signs = torch.where((torch.arange(length) + offset) % 3 == 0, -1.0, 1.0)
    return (values * signs).to(torch.float16).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-module", type=Path, required=True)
    args = parser.parse_args()
    runtime = _load_runtime(args.runtime_module)

    constant_pairs = {
        "schema": (encoder.COUPLED_RANK_SCHEMA, runtime.COUPLED_SCHEMA),
        "boundary": (encoder.COUPLED_BOUNDARY, runtime.COMPOSITION_TARGET),
        "component": (encoder.COUPLED_COMPONENT, runtime.COUPLED_COMPONENT),
        "cast_order": (encoder.COUPLED_CAST_ORDER, runtime.COUPLED_CAST_ORDER),
        "transform_id": (encoder.COUPLED_TRANSFORM_ID, runtime.TRANSFORM_ID),
        "sign_generator": (
            encoder.COUPLED_SIGN_GENERATOR,
            runtime.SIGN_GENERATOR,
        ),
        "sign_draw": (encoder.COUPLED_SIGN_DRAW, runtime.SIGN_DRAW),
    }
    if any(left != right for left, right in constant_pairs.values()):
        raise RuntimeError(f"encoder/runtime constants differ: {constant_pairs}")

    experts, hidden, local_intermediate = 2, 512, 32
    sidecar_tensors = {
        "w13_trellis": torch.zeros(
            2, experts, hidden // 16, local_intermediate // 16, 64,
            dtype=torch.int16,
        ),
        "w2_trellis": torch.zeros(
            experts, local_intermediate // 16, hidden // 16, 64,
            dtype=torch.int16,
        ),
        "w13_scale_ue8m0": torch.full(
            (experts, 2 * local_intermediate, hidden // 32), 127,
            dtype=torch.uint8,
        ),
        "w2_scale_ue8m0": torch.full(
            (experts, hidden, local_intermediate // 32), 127,
            dtype=torch.uint8,
        ),
        "gate_up_suh_fp16": _signed(hidden),
        "intermediate_scales_fp16": torch.stack(
            [
                torch.cat((_signed(local_intermediate, e),) * 3)
                for e in range(experts)
            ]
        ).contiguous(),
        "down_svh_fp16": _signed(hidden, 1),
        "coupled_sign_draw_u8": torch.zeros(experts, dtype=torch.uint8),
    }
    _, _, _, metadata_bpw = _payload_rates(sidecar_tensors)
    metadata = _rank_metadata(
        sidecar_tensors,
        layer=3,
        rank=2,
        design_sha256="a" * 64,
        scale_source_sha256="b" * 64,
        metadata_bpw=metadata_bpw,
    )
    component = runtime.validate_coupled_component(
        metadata,
        {name: sidecar_tensors[name] for name in runtime.SCALE_NAMES},
        layer=3,
        rank=2,
        experts=experts,
        hidden=hidden,
        intermediate=local_intermediate,
        expected_transform_sha256=encoder.COUPLED_TRANSFORM_SHA256,
    )
    expected_signs = encoder.rank_local_coupled_signs(
        intermediate=local_intermediate, rank=2
    )
    if not torch.equal(component.coupled_signs, expected_signs):
        raise RuntimeError("encoder/runtime procedural sign bytes differ")

    torch.manual_seed(905128)
    intermediate = 128
    scales = encoder.CoupledScaleSet(
        gate_up_suh=_signed(hidden),
        gate_svh=_signed(intermediate)[None],
        up_svh=_signed(intermediate, 1)[None],
        down_suh=_signed(intermediate, 2)[None],
        down_svh=_signed(hidden, 1),
        source_path=Path("/cpu-audit/source"),
        source_sha256="c" * 64,
        source_metadata={},
        tensor_hashes={},
    )
    gate = torch.randn(intermediate, hidden) / 32
    up = torch.randn(intermediate, hidden) / 32
    down = torch.randn(hidden, intermediate) / 16
    hidden_input = torch.randn(3, hidden).to(torch.bfloat16)
    physical = encoder.encode_coupled_scale_weights(
        gate, up, down, scales, expert=0, intermediate_draw=0
    )
    qdq = encoder._qdq_e4m3_k32
    encoder._qdq_e4m3_k32 = lambda value, *_: value
    try:
        observed = encoder.coupled_expert_reference(
            hidden_input,
            physical,
            scales,
            expert=0,
            intermediate_draw=0,
            quantize_activations=True,
        )
    finally:
        encoder._qdq_e4m3_k32 = qdq
    runtime_scales = runtime.P8ScaleSandwich(
        scales.gate_up_suh,
        torch.cat((scales.gate_svh, scales.up_svh, scales.down_suh), dim=1),
        scales.down_svh,
        runtime.rank_local_coupled_signs(intermediate=intermediate, rank=0),
        True,
        encoder.COUPLED_TRANSFORM_SHA256,
    )
    expected = runtime.coupled_reference(hidden_input, *physical, runtime_scales)
    if not torch.equal(observed, expected):
        raise RuntimeError("encoder/runtime complete pre-E4M3 topology differs")

    print(
        json.dumps(
            {
                "runtime_commit": PINNED_RUNTIME_COMMIT,
                "runtime_module_sha256": PINNED_RUNTIME_SHA256,
                "schema": encoder.COUPLED_RANK_SCHEMA,
                "status": "pass",
                "topology_bit_exact_pre_e4m3": True,
                "transform_sha256": encoder.COUPLED_TRANSFORM_SHA256,
                "validator_pass": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
