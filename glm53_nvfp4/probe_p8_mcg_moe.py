"""SM120 MoE closure probe for native P8 K3/K4 procedural MCG weights."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

from .shard_index import sha256_file
from .trellis_nvfp4 import procedural_state_values


def _enable_scaled_test_helper(test_module, *, runtime_supports_scaled: bool) -> None:
    """Add non-unit UE8M0 inputs to the pinned B12X device oracle."""
    source = inspect.getsource(test_module._run_trellis_dynamic)
    source = source.replace(
        "            trellis_direct_lut=direct_lut,\n",
        "            trellis_direct_lut=direct_lut,\n"
        "            trellis_codebook=\"mcg\",\n",
        1,
    )
    source = source.replace(
        "    topk_ids_override: torch.Tensor | None = None,\n):",
        "    topk_ids_override: torch.Tensor | None = None,\n"
        "    scaled_weights: bool = False,\n):",
    )
    source = source.replace(
        "    scaled_weights: bool = False,\n):",
        "    scaled_weights: bool = False,\n"
        "    scale_pattern: str = \"random\",\n"
        "    uniform_scale_code: int = 126,\n"
        "    identity_boundary: bool = False,\n):",
    )
    source = source.replace(
        "    identity_boundary: bool = False,\n):",
        "    identity_boundary: bool = False,\n"
        "    scale_codes_override=None,\n"
        "    weights_include_scales: bool = False,\n):",
    )
    source = source.replace(
        "    w13_ref = torch.stack((gate_w, up_w), dim=0)\n",
        """    if scaled_weights:
        def scale_codes(shape):
            if scale_pattern == "uniform":
                return torch.full(
                    shape, uniform_scale_code, dtype=torch.uint8, device=device
                )
            if scale_pattern != "random":
                raise ValueError(f"unknown scale_pattern {scale_pattern!r}")
            return torch.randint(
                118, 123, shape, generator=gen, dtype=torch.int64
            ).to(device=device, dtype=torch.uint8)
        if scale_codes_override is None:
            gate_scale_codes = scale_codes((E, n, K // 32))
            up_scale_codes = scale_codes((E, n, K // 32))
            down_scale_codes = scale_codes((E, K, n // 32))
        else:
            gate_scale_codes, up_scale_codes, down_scale_codes = (
                code.to(device=device, dtype=torch.uint8).contiguous()
                for code in scale_codes_override
            )
        if not weights_include_scales:
            gate_w = gate_w * torch.pow(
                2.0, gate_scale_codes.float().sub(127).repeat_interleave(32, dim=2)
            )
            up_w = up_w * torch.pow(
                2.0, up_scale_codes.float().sub(127).repeat_interleave(32, dim=2)
            )
            down_w = down_w * torch.pow(
                2.0, down_scale_codes.float().sub(127).repeat_interleave(32, dim=2)
            )
    w13_ref = torch.stack((gate_w, up_w), dim=0)
""",
    )
    source = source.replace(
        "    sentinel_u32 = torch.zeros(1, dtype=torch.uint32, device=device)\n",
        """    sentinel_u32 = torch.zeros(1, dtype=torch.uint32, device=device)
    w13_sfb_rp = sentinel_u32
    down_sfb_rp = sentinel_u32
    if scaled_weights:
        # Repacked FC1 order is up then gate; the trellis payload itself
        # remains projection-major gate then up.
        w13_scale_grid = torch.cat((up_scale_codes, gate_scale_codes), dim=1)
        w13_sfb_rp = _e8m0_scale_to_w4a8_sfb_inplace(
            w13_scale_grid.clone(), weight_E=E, rows=w1_n, k_dim=K,
            gated_half_rows=n,
        ).reshape(-1)
        down_sfb_rp = _e8m0_scale_to_w4a8_sfb_inplace(
            down_scale_codes.clone(), weight_E=E, rows=K, k_dim=n,
        ).reshape(-1)
    sfb_w13_mx = w13_scale_grid if scaled_weights else sentinel_u32
    sfb_down_mx = down_scale_codes if scaled_weights else sentinel_u32
""",
    )
    if runtime_supports_scaled:
        source = source.replace(
            "            trellis_direct_lut=direct_lut,\n",
            "            trellis_direct_lut=direct_lut,\n"
            "            trellis_scaled=scaled_weights,\n",
            1,
        )
    elif "scaled_weights" in source and "trellis_scaled=" in source:
        raise RuntimeError("unexpected pre-existing scaled helper source")
    source = source.replace(
        "            trellis_codebook=\"mcg\",\n",
        "            trellis_codebook=\"mcg\",\n"
        "            trellis_identity_boundary=identity_boundary,\n",
        1,
    )
    source = source.replace(
        "    oracle = trellis_moe_reference(\n"
        "        x.float(),\n"
        "        w13_ref,\n"
        "        down_w,\n"
        "        rotations,\n"
        "        reference_ids,\n"
        "        reference_weights,\n"
        "        coupled=coupled,\n"
        "    )\n",
        "    if identity_boundary:\n"
        "        oracle = trellis_moe_identity_reference(\n"
        "            x.float(), w13_ref, down_w, reference_ids,\n"
        "            reference_weights, activation=activation\n"
        "        )\n"
        "    else:\n"
        "        oracle = trellis_moe_reference(\n"
        "            x.float(), w13_ref, down_w, rotations, reference_ids,\n"
        "            reference_weights, coupled=coupled\n"
        "        )\n",
        1,
    )
    if runtime_supports_scaled:
        source = source.replace(
            "            (\"bits\", _BITS),\n",
            "            (\"bits\", _BITS),\n"
            "            (\"scaled\", int(scaled_weights)),\n",
            1,
        )
    source = source.replace(
        "            (\"scaled\", int(scaled_weights)),\n",
        "            (\"scaled\", int(scaled_weights)),\n"
        "            (\"identity_boundary\", int(identity_boundary)),\n",
        1,
    )
    source = source.replace(
        "        _gptr(cutlass.Uint8, scale_flat),\n"
        "        _gptr(cutlass.Uint8, scale_flat),\n"
        "        _gptr(cutlass.Uint8, scale_flat),\n"
        "        _gptr(cutlass.Uint8, scale_flat),\n",
        "        _gptr(cutlass.Uint8, sfb_w13_mx),\n"
        "        _gptr(cutlass.Uint8, sfb_down_mx),\n"
        "        _gptr(cutlass.Uint8, scale_flat),\n"
        "        _gptr(cutlass.Uint8, scale_flat),\n",
        1,
    )
    source = source.replace(
        "        _gptr(cutlass.Uint32, w13_flat),\n"
        "        _gptr(cutlass.Uint32, sentinel_u32),\n"
        "        _gptr(cutlass.Uint32, down_flat),\n"
        "        _gptr(cutlass.Uint32, sentinel_u32),\n",
        "        _gptr(cutlass.Uint32, w13_flat),\n"
        "        _gptr(cutlass.Uint32, w13_sfb_rp),\n"
        "        _gptr(cutlass.Uint32, down_flat),\n"
        "        _gptr(cutlass.Uint32, down_sfb_rp),\n",
        1,
    )
    required = (
        "scaled_weights: bool = False",
        "scale_codes_override=None",
        "weights_include_scales: bool = False",
        "trellis_codebook=\"mcg\"",
        "trellis_identity_boundary=identity_boundary",
        "trellis_moe_identity_reference(",
        "w13_sfb_rp = _e8m0_scale_to_w4a8_sfb_inplace",
        "_gptr(cutlass.Uint8, sfb_w13_mx)",
    )
    if runtime_supports_scaled:
        required = (
            *required,
            "trellis_scaled=scaled_weights",
            "(\"scaled\", int(scaled_weights))",
        )
    if not all(item in source for item in required):
        raise RuntimeError("pinned B12X helper could not be extended for scaled P8")
    from b12x.moe.fused_moe._impl import _e8m0_scale_to_w4a8_sfb_inplace

    test_module.__dict__["_e8m0_scale_to_w4a8_sfb_inplace"] = (
        _e8m0_scale_to_w4a8_sfb_inplace
    )
    test_module.__dict__["_scaled_helper_source"] = source
    exec(source, test_module.__dict__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b12x-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--mode", choices=("small", "split"), default="small")
    parser.add_argument("--scaled", action="store_true")
    parser.add_argument("--scale-pattern", choices=("uniform", "random"), default="random")
    parser.add_argument("--uniform-scale-code", type=int, default=126)
    parser.add_argument("--boundary", choices=("h128", "identity"), default="h128")
    parser.add_argument("--activation", choices=("silu", "situ"))
    parser.add_argument("--debug-intermediate", action="store_true")
    parser.add_argument("--real-codec-chunk", type=Path)
    parser.add_argument("--real-dense-chunk", type=Path)
    parser.add_argument("--real-experts", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    # Resolve the runtime B12X package first. The source tree is added only to
    # import its pinned test oracle, so it cannot replace the image's kernel.
    runtime_dynamic = importlib.import_module("b12x.moe._shared.kernels.dynamic")
    sys.path.insert(0, str(args.b12x_source))
    test_module = importlib.import_module("tests.moe.test_dynamic_w4a8_trellis")
    reference = importlib.import_module("tests._reference.trellis_moe")
    def identity_moe_reference(
        x, w13, down, topk_ids, topk_weights, *, activation: str
    ):
        flat_ids = topk_ids.reshape(-1).long()
        token_of = torch.arange(x.shape[0], device=x.device).repeat_interleave(
            topk_ids.shape[1]
        )
        xin = x.float()[token_of]
        gate = torch.einsum("rk,rik->ri", xin, w13[0, flat_ids].float())
        up = torch.einsum("rk,rik->ri", xin, w13[1, flat_ids].float())
        middle = (
            torch.nn.functional.silu(gate) * up
            if activation == "silu"
            else reference._situ(gate, up)
        )
        routed = torch.einsum("ri,rki->rk", middle, down[flat_ids].float())
        out = torch.zeros_like(x, dtype=torch.float32)
        out.index_add_(
            0,
            token_of,
            routed * topk_weights.reshape(-1).float()[:, None],
        )
        return out

    test_module.__dict__["trellis_moe_identity_reference"] = identity_moe_reference
    runtime_supports_scaled = (
        "trellis_scaled"
        in inspect.signature(runtime_dynamic.MoEDynamicKernelBackend).parameters
    )
    if args.scaled and not runtime_supports_scaled:
        raise RuntimeError("runtime image does not implement scaled trellis weights")
    _enable_scaled_test_helper(
        test_module, runtime_supports_scaled=runtime_supports_scaled
    )

    def build_mcg(
        generator: torch.Generator,
        experts: int,
        out_features: int,
        in_features: int,
        bits: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k16 = in_features // 16
        n16 = out_features // 16
        windows = experts * k16 * n16
        edges = torch.randint(
            0,
            1 << bits,
            (windows, 256),
            generator=generator,
            device="cpu",
        ).to(device)
        payload = reference._pack_native_tiles(edges, bits).reshape(
            experts, k16, n16, 16 * bits
        )
        states = reference._cyclic_states(edges, bits)
        table = (
            procedural_state_values("mcg")
            .float()
            .mul_(2.0)
            .to(torch.float8_e4m3fn)
            .float()
            .to(device)
        )
        values = table[states.reshape(-1)].reshape(windows, 256)
        rows, columns = reference._tile_position_map(device)
        tiles = torch.zeros(
            windows, 16, 16, dtype=torch.float32, device=device
        )
        tiles[:, rows, columns] = values
        weights = (
            tiles.reshape(experts, k16, n16, 16, 16)
            .permute(0, 2, 3, 1, 4)
            .reshape(experts, out_features, in_features)
            .contiguous()
        )
        return payload, weights

    real_sources: dict[str, object] | None = None
    real_scale_codes = None
    build_call = {"index": 0}
    if args.real_codec_chunk is not None or args.real_dense_chunk is not None:
        if args.real_codec_chunk is None or args.real_dense_chunk is None:
            raise ValueError("real payload closure requires both codec and dense chunks")
        if not args.scaled or args.boundary != "identity":
            raise ValueError("real payload closure requires --scaled --boundary identity")
        if not 1 <= args.real_experts <= 72:
            raise ValueError("--real-experts must be in [1, 72]")
        bases = [
            f"model.language_model.layers.3.mlp.experts.{expert}"
            for expert in range(args.real_experts)
        ]
        codec_data: dict[str, torch.Tensor] = {}
        dense_data: dict[str, torch.Tensor] = {}
        codec_metadata: dict[str, str] | None
        with safe_open(args.real_codec_chunk, framework="pt", device="cpu") as src:
            codec_metadata = src.metadata()
            for projection in ("gate_proj", "up_proj", "down_proj"):
                codec_data[f"{projection}.trellis"] = torch.stack(
                    [src.get_tensor(f"{base}.{projection}.trellis") for base in bases]
                ).cuda().contiguous()
                codec_data[f"{projection}.scale"] = torch.stack(
                    [src.get_tensor(f"{base}.{projection}.scale_ue8m0") for base in bases]
                ).cuda().contiguous()
        with safe_open(args.real_dense_chunk, framework="pt", device="cpu") as src:
            dense_metadata = src.metadata()
            for projection in ("gate_proj", "up_proj", "down_proj"):
                dense_data[projection] = torch.stack(
                    [src.get_tensor(f"{base}.{projection}.weight") for base in bases]
                ).cuda().contiguous()
        required_metadata = {
            "bits": "4",
            "alphabet": "e4m3",
            "block_size": "32",
            "scale": "ue8m0",
            "law": "procedural-mcg-alpha2",
            "boundary": "identity",
            "ldlq": "false",
        }
        if any(codec_metadata.get(key) != value for key, value in required_metadata.items()):
            raise RuntimeError(f"codec metadata violates P8 contract: {codec_metadata}")
        if dense_metadata.get("role") != "decoded-pseudoquant-carrier":
            raise RuntimeError(f"dense reference role mismatch: {dense_metadata}")

        projections = ("gate_proj", "up_proj", "down_proj")

        def build_real(
            generator: torch.Generator,
            experts: int,
            out_features: int,
            in_features: int,
            bits: int,
            device: torch.device,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            projection = projections[build_call["index"]]
            build_call["index"] += 1
            payload = codec_data[f"{projection}.trellis"]
            weight = dense_data[projection]
            if experts != args.real_experts or bits != 4:
                raise RuntimeError("real payload helper received a mismatched geometry")
            if tuple(weight.shape[1:]) != (out_features, in_features):
                raise RuntimeError(
                    f"{projection} geometry {tuple(weight.shape)} does not match "
                    f"{(experts, out_features, in_features)}"
                )
            return payload, weight

        test_module.build_trellis_weight = build_real
        real_scale_codes = (
            codec_data["gate_proj.scale"],
            codec_data["up_proj.scale"],
            codec_data["down_proj.scale"],
        )
        real_sources = {
            "codec": {
                "path": str(args.real_codec_chunk.resolve()),
                "sha256": sha256_file(args.real_codec_chunk),
                "metadata": codec_metadata,
            },
            "dense_reference": {
                "path": str(args.real_dense_chunk.resolve()),
                "sha256": sha256_file(args.real_dense_chunk),
                "metadata": dense_metadata,
            },
            "experts": list(range(args.real_experts)),
        }
    else:
        test_module.build_trellis_weight = build_mcg
    cells: list[dict[str, object]] = []
    token_counts = (3,) if args.mode == "small" else (33, 64)
    bit_values = (4,) if real_sources is not None else (3, 4)
    for bits in bit_values:
        for tokens in token_counts:
            test_module._BITS = bits
            activation = args.activation or (
                "silu" if args.mode == "small" else "situ"
            )
            run_kwargs = dict(
                activation=activation,
                E=args.real_experts if real_sources is not None else 8,
                m=tokens,
                K=4096 if real_sources is not None else 512,
                n=2048 if real_sources is not None else 256,
                top_k=min(4, args.real_experts) if real_sources is not None else 4,
                seed=args.seed + bits * 100 + tokens,
                tile_m=16 if args.mode == "small" else 64,
                split_materialized=args.mode == "split",
                mac=4 if args.mode == "small" else 64,
            )
            if args.scaled:
                run_kwargs.update(
                    scaled_weights=True,
                    scale_pattern=args.scale_pattern,
                    uniform_scale_code=args.uniform_scale_code,
                )
            if real_sources is not None:
                build_call["index"] = 0
                run_kwargs.update(
                    scale_codes_override=real_scale_codes,
                    weights_include_scales=True,
                )
            run_kwargs["identity_boundary"] = args.boundary == "identity"
            keepalive: list[dict[str, object]] | None = (
                [] if args.debug_intermediate else None
            )
            if keepalive is not None:
                run_kwargs["keepalive"] = keepalive
            actual, expected = test_module._run_trellis_dynamic(**run_kwargs)
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    actual.reshape(1, -1), expected.reshape(1, -1)
                ).item()
            )
            relative_l2 = float(
                ((actual - expected).norm() / expected.norm().clamp_min(1e-9)).item()
            )
            cell: dict[str, object] = {
                    "bits": bits,
                    "tokens": tokens,
                    "finite": bool(torch.isfinite(actual).all()),
                    "nonzero": int(torch.count_nonzero(actual).item()),
                    "cosine": cosine,
                    "relative_l2": relative_l2,
                    "actual_norm": float(actual.norm().item()),
                    "expected_norm": float(expected.norm().item()),
                    "norm_ratio": float(
                        (actual.norm() / expected.norm().clamp_min(1e-9)).item()
                    ),
                    "output_sha256": hashlib.sha256(
                        actual.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                    ).hexdigest(),
                    "pass": bool(
                        torch.isfinite(actual).all()
                        and cosine > 0.995
                        and relative_l2 < 0.12
                    ),
                }
            if keepalive is not None:
                if args.mode != "split" or args.boundary != "identity":
                    raise ValueError(
                        "--debug-intermediate is limited to split identity diagnostics"
                    )
                state = keepalive[0]
                rows_padded = int(state["rows_padded"])
                intermediate_u32 = state["intermediate_u32"]
                payload_words = rows_padded * (run_kwargs["n"] // 4)
                payload = (
                    intermediate_u32[:payload_words]
                    .contiguous()
                    .view(torch.uint8)
                    .reshape(rows_padded, run_kwargs["n"])
                    .view(torch.float8_e4m3fn)
                    .float()
                )
                output_tiles = run_kwargs["n"] // 128
                scale_bytes = (
                    intermediate_u32[payload_words:]
                    .contiguous()
                    .view(torch.uint8)
                    .reshape(output_tiles, rows_padded, 4)
                    .permute(1, 0, 2)
                    .reshape(rows_padded, run_kwargs["n"] // 32)
                )
                stored = payload * torch.pow(
                    2.0, scale_bytes.float().sub(127).repeat_interleave(32, dim=1)
                )
                perm = torch.tensor(
                    (
                        0, 1, 8, 9, 4, 5, 12, 13,
                        2, 3, 10, 11, 6, 7, 14, 15,
                        20, 21, 28, 29, 16, 17, 24, 25,
                        22, 23, 30, 31, 18, 19, 26, 27,
                    ),
                    device=stored.device,
                )
                natural = torch.empty_like(stored)
                for block_start in range(0, run_kwargs["n"], 32):
                    natural[:, block_start + perm] = stored[
                        :, block_start : block_start + 32
                    ]
                expected_rows = torch.zeros_like(natural)
                active_rows: list[torch.Tensor] = []
                for expert in range(run_kwargs["E"]):
                    start = int(state["expert_tile_base"][expert].item()) * run_kwargs[
                        "tile_m"
                    ]
                    count = int(state["row_counts"][expert].item())
                    if count == 0:
                        continue
                    row_ids = torch.arange(start, start + count, device=stored.device)
                    token_ids = state["token_map"][row_ids].long()
                    xin = state["x"].float()[token_ids]
                    gate = xin @ state["w13_ref"][0, expert].float().T
                    up = xin @ state["w13_ref"][1, expert].float().T
                    expected_rows[row_ids] = reference._situ(gate, up)
                    active_rows.append(row_ids)
                active = torch.cat(active_rows)
                row_cos = torch.nn.functional.cosine_similarity(
                    natural[active], expected_rows[active], dim=1
                )
                stored_blocks = stored.reshape(rows_padded, -1, 32)
                alt_natural = stored_blocks[:, :, perm].reshape_as(stored)
                direct_cosine = torch.nn.functional.cosine_similarity(
                    stored[active].reshape(1, -1),
                    expected_rows[active].reshape(1, -1),
                )
                alt_cosine = torch.nn.functional.cosine_similarity(
                    alt_natural[active].reshape(1, -1),
                    expected_rows[active].reshape(1, -1),
                )
                actual_columns = torch.nn.functional.normalize(
                    natural[active].T, dim=1
                )
                expected_columns = torch.nn.functional.normalize(
                    expected_rows[active].T, dim=1
                )
                feature_cross = actual_columns @ expected_columns.T
                cell["intermediate_debug"] = {
                    "active_rows": int(active.numel()),
                    "cosine": float(
                        torch.nn.functional.cosine_similarity(
                            natural[active].reshape(1, -1),
                            expected_rows[active].reshape(1, -1),
                        ).item()
                    ),
                    "relative_l2": float(
                        (
                            (natural[active] - expected_rows[active]).norm()
                            / expected_rows[active].norm().clamp_min(1e-9)
                        ).item()
                    ),
                    "row_cosine_min": float(row_cos.min().item()),
                    "row_cosine_mean": float(row_cos.mean().item()),
                    "stored_order_cosine": float(direct_cosine.item()),
                    "alternate_k32_cosine": float(alt_cosine.item()),
                    "best_expected_feature_abs_cosine_mean": float(
                        feature_cross.abs().max(dim=1).values.mean().item()
                    ),
                    "best_expected_feature_cosine_mean": float(
                        feature_cross.max(dim=1).values.mean().item()
                    ),
                }
                output_row_cos = torch.nn.functional.cosine_similarity(
                    actual.float(), expected.float(), dim=1
                )
                cross = torch.nn.functional.normalize(actual.float(), dim=1) @ (
                    torch.nn.functional.normalize(expected.float(), dim=1).T
                )
                cell["output_row_debug"] = {
                    "diagonal_cosine_mean": float(output_row_cos.mean().item()),
                    "best_expected_row_cosine_mean": float(
                        cross.max(dim=1).values.mean().item()
                    ),
                }
                h128_expected = reference.trellis_moe_reference(
                    state["x"].float(),
                    state["w13_ref"],
                    state["down_w"],
                    state["rotations"],
                    state["reference_ids"],
                    state["reference_weights"],
                    coupled=False,
                )
                cell["h128_oracle_debug"] = {
                    "cosine": float(
                        torch.nn.functional.cosine_similarity(
                            actual.reshape(1, -1), h128_expected.reshape(1, -1)
                        ).item()
                    ),
                    "relative_l2": float(
                        (
                            (actual - h128_expected).norm()
                            / h128_expected.norm().clamp_min(1e-9)
                        ).item()
                    ),
                }
            cells.append(cell)

    dynamic_path = Path(runtime_dynamic.__file__).resolve()
    payload = {
        "schema": "glm53-p8-mcg-native-moe-closure.v2",
        "product": "P8",
        "compute": "mxf8f6f4 m16n8k32 with E4M3 activations and weights",
        "law": "procedural MCG alpha 2.0",
        "codebook_selection": "explicit constructor argument: mcg",
        "boundary": args.boundary,
        "weight_scale_contract": (
            "physical non-unit UE8M0/32 consumed by MX MMA"
            if args.scaled
            else "identity UE8M0 control"
        ),
        "scale_pattern": args.scale_pattern if args.scaled else "identity",
        "uniform_scale_code": (
            args.uniform_scale_code
            if args.scaled and args.scale_pattern == "uniform"
            else None
        ),
        "ldlq": False,
        "geometry": {
            "mode": args.mode,
            "experts": args.real_experts if real_sources is not None else 8,
            "tokens": list(token_counts),
            "hidden": 4096 if real_sources is not None else 512,
            "intermediate": 2048 if real_sources is not None else 256,
            "top_k": min(4, args.real_experts) if real_sources is not None else 4,
            "activation": args.activation
            or ("silu" if args.mode == "small" else "situ"),
            "tile_m": 16 if args.mode == "small" else 64,
            "materialize_intermediate": args.mode == "split",
        },
        "real_layer_payload": real_sources,
        "cells": cells,
        "decision": "pass" if all(cell["pass"] for cell in cells) else "fail",
        "runtime_dynamic": {
            "path": str(dynamic_path),
            "sha256": sha256_file(dynamic_path),
        },
        "oracle_source": {
            "path": str(args.b12x_source),
            "test_sha256": sha256_file(
                args.b12x_source / "tests/moe/test_dynamic_w4a8_trellis.py"
            ),
            "reference_sha256": sha256_file(
                args.b12x_source / "tests/_reference/trellis_moe.py"
            ),
        },
        "isa_cost": "mxf8f6f4 uses twice the MMA issue count of NVFP4",
        "scope": (
            "split-prefill device arithmetic closure only; not end-to-end KLD or speed qualification"
            if args.mode == "split"
            else "small-M device arithmetic closure only; not end-to-end KLD or speed qualification"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    if payload["decision"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
