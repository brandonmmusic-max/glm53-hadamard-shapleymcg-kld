"""Fit-role routed-output damage of one GLM-5.3-Flash routed layer at K3/K4/K5.

For every candidate rate the layer's 288 coupled experts are reference-decoded
(`decode_trellis_mxf`, the same procedural MCG E4M3 codebook the kernel uses)
and executed through the exact coupled reference forward
(`coupled_expert_reference`, quantized E4M3 carriers) on a fixed, domain-balanced
sample of fit-role tokens.  The payoff is the squared routed-output error in the
residual stream::

    z_e(t) = w_te (f_q,e(x_t) - f_e(x_t))        S(t) = sum_e z_e(t)
    D_L(r) = sum_t ||S(t)||^2                      psi_e = sum_t 1/2 <z_e(t), S(t)>

``psi_e`` is the exact Shapley value of the quadratic per-token game whose
grand-coalition value is ``1/2 ||S(t)||^2`` (cross terms shared symmetrically),
so ``sum_e psi_e = D_L(r) / 2``.  This is a local, fit-role proxy in the style of
the owner's ShapleyMCG/SQG work; the end-to-end CF32 KLD of the installed
allocation is the only quality claim.

K4 weights are read back from the packed TP4 rank sidecars (inverse of
``build_p8_coupled_scale_tp4_sidecars``); K3/K5 candidates from encoder chunks.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from .capture import LayerCapture
from .p8_coupled_scale import (
    COUPLED_SIGN_DRAW,
    coupled_expert_reference,
    encode_coupled_scale_weights,
    load_exact_exl3_scales,
    source_expert_reference,
)
from .shard_index import IndexedCheckpoint, sha256_file
from .trellis_mxf import decode_trellis_mxf, state_lut

EXPERTS = 288
HIDDEN = 4096
INTERMEDIATE = 2048
WORLD_SIZE = 4
TOP_K = 8
ROWS_PER_WINDOW = 2048
WEIGHTS_PER_EXPERT = 3 * HIDDEN * INTERMEDIATE
SCHEMA = "glm53.p8-layer-rate-damage.v1"


def state_codebook(bits: int, device: torch.device) -> torch.Tensor:
    return state_lut(bits, alphabet="e4m3", law="mcg", compander_scale=2.0, device=device)


def payload_bytes_per_expert(bits: int) -> int:
    return WEIGHTS_PER_EXPERT * (4 * bits + 1) // 32


# ----------------------------------------------------------------- streams


class RankLayerStreams:
    """Per-expert chunk-layout streams recovered from the four TP rank sidecars."""

    def __init__(self, rank_paths: list[Path]):
        if len(rank_paths) != WORLD_SIZE:
            raise ValueError("four TP rank sidecars are required")
        self.paths = [Path(p) for p in rank_paths]
        self.tensors = []
        bits = set()
        for rank, path in enumerate(self.paths):
            with safe_open(path, framework="pt", device="cpu") as src:
                metadata = src.metadata() or {}
                if metadata.get("rank") != str(rank) or metadata.get("world_size") != "4":
                    raise RuntimeError(f"rank sidecar metadata mismatch: {path}")
                bits.add(int(metadata["bits"]))
                self.tensors.append({name: src.get_tensor(name) for name in
                                     ("w13_trellis", "w2_trellis", "w13_scale_ue8m0", "w2_scale_ue8m0")})
        if len(bits) != 1:
            raise RuntimeError("rank sidecars disagree on trellis rate")
        self.bits = bits.pop()
        words = 16 * self.bits
        local_i = INTERMEDIATE // WORLD_SIZE
        for t in self.tensors:
            if (tuple(t["w13_trellis"].shape) != (2, EXPERTS, HIDDEN // 16, local_i // 16, words)
                    or tuple(t["w2_trellis"].shape) != (EXPERTS, local_i // 16, HIDDEN // 16, words)
                    or tuple(t["w13_scale_ue8m0"].shape) != (EXPERTS, 2 * local_i, HIDDEN // 32)
                    or tuple(t["w2_scale_ue8m0"].shape) != (EXPERTS, HIDDEN, local_i // 32)):
                raise RuntimeError("rank sidecar geometry differs from GLM-5.3-Flash TP4")

    def expert(self, expert: int) -> dict[str, torch.Tensor]:
        local_i = INTERMEDIATE // WORLD_SIZE
        gate = torch.cat([t["w13_trellis"][0, expert] for t in self.tensors], dim=1)
        up = torch.cat([t["w13_trellis"][1, expert] for t in self.tensors], dim=1)
        down = torch.cat([t["w2_trellis"][expert] for t in self.tensors], dim=0)
        # Packer plane order is up-then-gate per rank (fc1_scale_plane_order=up-gate).
        up_scale = torch.cat([t["w13_scale_ue8m0"][expert, :local_i] for t in self.tensors], dim=0)
        gate_scale = torch.cat([t["w13_scale_ue8m0"][expert, local_i:] for t in self.tensors], dim=0)
        down_scale = torch.cat([t["w2_scale_ue8m0"][expert] for t in self.tensors], dim=1)
        return {"gate": (gate, gate_scale), "up": (up, up_scale), "down": (down, down_scale)}


class ChunkLayerStreams:
    """Per-expert streams straight from encoder chunk files."""

    def __init__(self, chunk_paths: list[Path], layer: int):
        self.layer = layer
        self.handles: dict[int, Path] = {}
        bits = set()
        for path in chunk_paths:
            with safe_open(path, framework="pt", device="cpu") as src:
                metadata = src.metadata() or {}
                if metadata.get("layer") != str(layer):
                    raise RuntimeError(f"chunk layer mismatch: {path}")
                bits.add(int(metadata["bits"]))
                start, stop = (int(v) for v in metadata["expert_range"].split(":"))
            for expert in range(start, stop):
                self.handles[expert] = Path(path)
        if len(bits) != 1 or set(self.handles) != set(range(EXPERTS)):
            raise RuntimeError("chunks do not form one complete single-rate layer")
        self.bits = bits.pop()
        self.prefix = None

    def expert(self, expert: int) -> dict[str, torch.Tensor]:
        path = self.handles[expert]
        with safe_open(path, framework="pt", device="cpu") as src:
            if self.prefix is None:
                names = [n for n in src.keys() if n.endswith(f".experts.{expert}.gate_proj.trellis")]
                self.prefix = names[0].split(f"layers.{self.layer}.")[0]
            base = f"{self.prefix}layers.{self.layer}.mlp.experts.{expert}"
            return {proj: (src.get_tensor(f"{base}.{proj}_proj.trellis"), src.get_tensor(f"{base}.{proj}_proj.scale_ue8m0"))
                    for proj in ("gate", "up", "down")}


def decode_expert(streams: dict[str, torch.Tensor], bits: int, codebook: torch.Tensor,
                  device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gate = decode_trellis_mxf(streams["gate"][0], codebook, streams["gate"][1], bits=bits, block_size=32,
                              rows=INTERMEDIATE, width=HIDDEN, device=device)
    up = decode_trellis_mxf(streams["up"][0], codebook, streams["up"][1], bits=bits, block_size=32,
                            rows=INTERMEDIATE, width=HIDDEN, device=device)
    down = decode_trellis_mxf(streams["down"][0], codebook, streams["down"][1], bits=bits, block_size=32,
                              rows=HIDDEN, width=INTERMEDIATE, device=device)
    return gate, up, down


# ------------------------------------------------------------------ tokens


def select_tokens(capture: LayerCapture, tokens_per_window: int) -> dict:
    rows, domains = [], []
    for window in capture.window_indices:
        start = window * ROWS_PER_WINDOW
        picks = np.linspace(0, ROWS_PER_WINDOW - 1, tokens_per_window).round().astype(np.int64)
        rows.extend((start + picks).tolist())
        domains.extend([capture.window_domains[window]] * tokens_per_window)
    rows = np.asarray(rows, dtype=np.int64)
    hidden = torch.from_numpy(np.array(capture.hidden_words[rows], copy=True)).view(torch.bfloat16)
    ids = torch.from_numpy(np.array(capture.topk_ids[rows], copy=True).astype(np.int64))
    weights = torch.from_numpy(np.array(capture.topk_weights[rows], copy=True))
    return {"rows": rows, "domains": domains, "hidden": hidden, "ids": ids, "weights": weights}


# ------------------------------------------------------------------ scoring


def shapley_split(z: torch.Tensor, ids: torch.Tensor, experts: int = EXPERTS) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (S, psi) for z[T, K, H] error contributions routed by ids[T, K]."""
    total = z.double().sum(dim=1)  # S(t), accumulated in float64
    inner = 0.5 * (z.double() * total[:, None, :]).sum(dim=-1)  # <z_e(t), S(t)>/2 per slot
    psi = torch.zeros(experts, dtype=torch.float64, device=z.device)
    psi.index_add_(0, ids.reshape(-1), inner.reshape(-1))
    return total, psi


def score_layer(*, layer: int, source: IndexedCheckpoint, scales, tokens: dict, streams, bits: int,
                device: torch.device, base_outputs: torch.Tensor | None, receipt_nmse: dict | None) -> tuple[dict, torch.Tensor]:
    codebook = state_codebook(bits, device)
    hidden = tokens["hidden"].to(device)
    ids = tokens["ids"].to(device)
    weights = tokens["weights"].to(device).float()
    T = hidden.shape[0]
    z = torch.zeros(T, TOP_K, HIDDEN, dtype=torch.float32, device=device)
    base = torch.zeros(T, TOP_K, HIDDEN, dtype=torch.float32, device=device) if base_outputs is None else base_outputs
    prefix = source.expert_prefix(layer, 0).split(f"layers.{layer}.")[0]
    nmse = {"gate_proj": [], "up_proj": [], "down_proj": []}
    nmse_check = []
    started = time.time()
    for expert in range(EXPERTS):
        token_idx, slot_idx = torch.nonzero(ids == expert, as_tuple=True)
        base_name = f"{prefix}layers.{layer}.mlp.experts.{expert}"
        original = {proj: source.get(f"{base_name}.{proj}.weight").to(device).float()
                    for proj in ("gate_proj", "up_proj", "down_proj")}
        target = encode_coupled_scale_weights(original["gate_proj"], original["up_proj"], original["down_proj"],
                                              scales, expert=expert, intermediate_draw=COUPLED_SIGN_DRAW)
        decoded = decode_expert(streams.expert(expert), bits, codebook, device)
        for proj, tgt, dec in zip(("gate_proj", "up_proj", "down_proj"), target, decoded):
            delta = dec - tgt
            value = float((delta.double().square().sum() / tgt.double().square().sum()).item())
            nmse[proj].append(value)
            if receipt_nmse is not None and (expert, proj) in receipt_nmse:
                nmse_check.append({"expert": expert, "projection": proj, "recomputed": value,
                                   "receipt": receipt_nmse[(expert, proj)]})
        if token_idx.numel() == 0:
            continue
        x = hidden[token_idx]
        w = weights[token_idx, slot_idx]
        if base_outputs is None:
            base[token_idx, slot_idx] = source_expert_reference(x, original["gate_proj"], original["up_proj"], original["down_proj"])
        f_q = coupled_expert_reference(x, decoded, scales, expert=expert, intermediate_draw=COUPLED_SIGN_DRAW,
                                       quantize_activations=True)
        z[token_idx, slot_idx] = w[:, None] * (f_q - base[token_idx, slot_idx])
        del original, target, decoded, f_q
    del codebook
    total, psi = shapley_split(z, ids)
    routed_output = (weights[:, :, None] * base).sum(dim=1)
    per_token = total.double().square().sum(dim=-1)
    output_norm = routed_output.double().square().sum(dim=-1)
    domains = np.asarray(tokens["domains"])
    per_domain = {}
    for domain in sorted(set(domains)):
        mask = torch.from_numpy(domains == domain).to(device)
        per_domain[domain] = {"damage_mean": float(per_token[mask].mean().item()),
                              "relative": float((per_token[mask].sum() / output_norm[mask].sum()).item())}
    result = {
        "bits": bits,
        "payload_bytes_per_expert": payload_bytes_per_expert(bits),
        "payload_bytes_layer": payload_bytes_per_expert(bits) * EXPERTS,
        "tokens": int(T),
        "damage_sum": float(per_token.sum().item()),
        "damage_mean_per_token": float(per_token.mean().item()),
        "relative_to_routed_output": float((per_token.sum() / output_norm.sum()).item()),
        "shapley_psi_sum_check": float(psi.sum().item()),
        "per_domain": per_domain,
        "per_expert_psi": [float(v) for v in psi.tolist()],
        "weight_nmse_mean": {proj: float(np.mean(values)) for proj, values in nmse.items()},
        "weight_nmse_max": {proj: float(np.max(values)) for proj, values in nmse.items()},
        "receipt_nmse_check": nmse_check,
        "elapsed_seconds": time.time() - started,
    }
    if not math.isclose(result["shapley_psi_sum_check"], result["damage_sum"] / 2, rel_tol=1e-6, abs_tol=1e-9):
        raise RuntimeError("Shapley shares do not close to half the squared routed damage")
    return result, base


def _receipt_nmse(paths: list[Path]) -> dict:
    values = {}
    for path in paths:
        for row in json.loads(Path(path).read_text())["projection_metrics"]:
            values[(int(row["expert"]), row["projection"])] = float(row["transformed_weight_nmse"])
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, required=True, choices=range(3, 45))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--exl3-scales", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--k4-rank-dir", type=Path, help="flat sidecar dir holding p8-layer-LLL-tp4-rank-R.safetensors")
    parser.add_argument("--k3-chunk", type=Path, action="append", default=[])
    parser.add_argument("--k5-chunk", type=Path, action="append", default=[])
    parser.add_argument("--k4-chunk-receipt", type=Path, action="append", default=[],
                        help="K4 chunk receipts whose projection NMSE cross-checks the rank inverse")
    parser.add_argument("--tokens-per-window", type=int, default=48)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    source = IndexedCheckpoint(args.source, args.source_index)
    scales = load_exact_exl3_scales(args.exl3_scales, layer=args.layer, expected_experts=EXPERTS,
                                    expected_hidden=HIDDEN, expected_intermediate=INTERMEDIATE)
    capture = LayerCapture(args.capture_root, args.layer, args.roles, max_samples=16, data_role="fit",
                           sampling_strategy="domain-balanced")
    tokens = select_tokens(capture, args.tokens_per_window)
    arms: dict[str, tuple[object, int, list]] = {}
    if args.k4_rank_dir:
        ranks = [args.k4_rank_dir / f"p8-layer-{args.layer:03d}-tp4-rank-{rank}.safetensors" for rank in range(4)]
        arms["4"] = (RankLayerStreams(ranks), 4, ranks)
    if args.k3_chunk:
        arms["3"] = (ChunkLayerStreams(args.k3_chunk, args.layer), 3, args.k3_chunk)
    if args.k5_chunk:
        arms["5"] = (ChunkLayerStreams(args.k5_chunk, args.layer), 5, args.k5_chunk)
    if not arms:
        raise ValueError("at least one rate source is required")
    receipt_nmse = _receipt_nmse(args.k4_chunk_receipt) if args.k4_chunk_receipt else None
    base_outputs = None
    results = {}
    for key in sorted(arms):
        streams, bits, paths = arms[key]
        if streams.bits != bits:
            raise RuntimeError(f"rate source for K{bits} carries K{streams.bits} streams")
        with torch.inference_mode():
            result, base_outputs = score_layer(layer=args.layer, source=source, scales=scales, tokens=tokens,
                                               streams=streams, bits=bits, device=device, base_outputs=base_outputs,
                                               receipt_nmse=receipt_nmse if bits == 4 else None)
        result["sources"] = [{"path": str(Path(p).resolve()), "bytes": Path(p).stat().st_size, "sha256": sha256_file(p)}
                             for p in paths]
        results[key] = result
        print(json.dumps({"layer": args.layer, "bits": bits, "damage_mean": result["damage_mean_per_token"],
                          "relative": result["relative_to_routed_output"], "elapsed": result["elapsed_seconds"]}), flush=True)
    payload = {
        "schema": SCHEMA,
        "layer": args.layer,
        "role": "fit",
        "tokens": int(tokens["hidden"].shape[0]),
        "tokens_per_window": args.tokens_per_window,
        "fit_windows": [int(w) for w in capture.window_indices],
        "token_rows": [int(r) for r in tokens["rows"]],
        "payoff": "sum_t ||sum_e w_te (f_q,e(x_t) - f_e(x_t))||^2 on fit tokens; residual-stream units; quantized E4M3 carriers",
        "shapley": "psi_e = sum_t 1/2 <z_e(t), S(t)>; exact for the quadratic per-token game; sum_e psi_e = damage/2",
        "scale_source_sha256": scales.source_sha256,
        "capture_manifest_sha256": sha256_file(args.capture_root / "capture-manifest.json"),
        "roles_sha256": sha256_file(args.roles),
        "rates": results,
        "protected_roles_opened": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output)}))


if __name__ == "__main__":
    main()
