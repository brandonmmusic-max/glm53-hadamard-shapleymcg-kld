"""Native-RNE P4 Viterbi selection and scale refitting, without LDLQ.

The 64-KiB state LUT is an offline Viterbi input only. No LUT is exported;
the serving decoder regenerates the fixed alpha1 law procedurally.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np
import torch

from .block_gptq import _prepare_full_inverse
from .modelopt import PackedNVFP4, pack_codes
from .p4_codec import (
    P4Payload, e2m1_codes, mcg_half_values,
    pack_k4_states, validate_payload,
)
from .p4_serving_codec import LAW
from .trellis_nvfp4 import (
    _encode_tiles, _prepare_tiles,
    _encode_nvfp4_with_gptq_feedback, _states_to_normalized_blocks,
    pack_trellis_edges, reconstruct_trellis_states, unpack_trellis_edges,
)


@dataclass(frozen=True)
class P4Encoding:
    payload: P4Payload
    endpoint: PackedNVFP4
    report: dict


def encoder_state_lut(*, device: torch.device | str = "cpu") -> torch.Tensor:
    """65,536 E4M3 byte labels, each numerically E2M1, for the offline encoder.

    The new name is intentional: legacy e2m1_state_lut(law='mcg') resolves
    midpoint ties differently and canonicalizes negative zero.
    """
    states = np.arange(65536, dtype="<u2")
    codes = e2m1_codes(mcg_half_values(states))
    # Exact E4M3 encoding of E2M1 magnitudes; sign includes negative zero.
    magnitudes = np.array([0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C], dtype=np.uint8)
    raw = magnitudes[codes & 7] | ((codes & 8) << 4)
    return torch.from_numpy(raw).to(device=device).contiguous()


def encoder_lut_sha256() -> str:
    return hashlib.sha256(encoder_state_lut().numpy().tobytes()).hexdigest()


def checked_select_tiles(tiles: torch.Tensor, lut: torch.Tensor, *, tailbite_context: int,
                         validate_law: bool = True):
    """Select with the frozen LUT and reject a backend with incompatible states.

    Returns values reconstructed from the cyclic stored stream, not a backend's
    potentially different scalar labels. This is also the CPU-testable adapter.
    """
    if tiles.ndim != 2 or tiles.shape[1] != 256 or not tiles.numel():
        raise ValueError("P4 Viterbi needs nonempty 256-element tiles")
    if lut.dtype != torch.uint8 or tuple(lut.shape) != (65536,):
        raise ValueError("P4 encoder needs a 65536-byte offline state LUT")
    if validate_law and not torch.equal(lut, encoder_state_lut(device=lut.device)):
        raise ValueError("P4 Viterbi law must equal the frozen native-RNE state LUT")
    quantized, states = _encode_tiles(tiles, lut, bits=4, tailbite_context=tailbite_context)
    if states.shape != tiles.shape or quantized.shape != tiles.shape or states.dtype not in (torch.int16, torch.int32, torch.int64, torch.uint16):
        raise ValueError("Viterbi backend returned incompatible states/values")
    if states.dtype != torch.int16 and bool(((states.to(torch.int64) < 0) | (states.to(torch.int64) > 65535)).any()):
        raise ValueError("Viterbi state labels must be in 0..65535")
    labels = states.to(torch.int64) & 0xFFFF
    packed = pack_trellis_edges(labels, 4)
    reconstructed = reconstruct_trellis_states(unpack_trellis_edges(packed, 4), 4).to(torch.int64)
    if not torch.equal(labels, reconstructed):
        raise ValueError("Viterbi states violate cyclic K4 stream closure")
    expected = lut.view(torch.float8_e4m3fn).float()[labels]
    # Signed zeros in backend float arithmetic are immaterial for selection;
    # retain the expected sign at the physical nibble reconstruction below.
    if not torch.equal(quantized.float(), expected):
        raise ValueError("Viterbi labels disagree with native-RNE decoder law")
    return expected, reconstructed


def _gptq_selector(tiles, lut, *, bits, tailbite_context):
    if bits != 4:
        raise ValueError("the P4 v2 serving encoder is K4 only")
    return checked_select_tiles(tiles, lut, tailbite_context=tailbite_context, validate_law=False)


def feedback_select(weight, hessian, scales, global_scale, lut, *, tailbite_context=128,
                    percdamp=0.01, column_block=128, prepared_inverse=None,
                    validate_inputs=True):
    """CPU-testable adapter to the actual native-group GPTQ feedback path."""
    if hessian.shape != (weight.shape[1], weight.shape[1]) or hessian.device != weight.device:
        raise ValueError("P4 Hessian shape/device mismatch")
    if validate_inputs:
        if not bool(torch.isfinite(hessian).all()) or not torch.allclose(hessian, hessian.T, atol=1e-6, rtol=1e-5):
            raise ValueError("P4 Hessian must be finite and symmetric")
        if not torch.equal(lut, encoder_state_lut(device=lut.device)):
            raise ValueError("P4 feedback requires native-RNE law")
    return _encode_nvfp4_with_gptq_feedback(weight, hessian, scales, global_scale, lut,
        bits=4, tailbite_context=tailbite_context, percdamp=percdamp, column_block=column_block,
        prepared_inverse=prepared_inverse, tile_encoder=_gptq_selector)


def qdq_p4_activations(x: torch.Tensor) -> torch.Tensor:
    """The explicit prologue SFA policy on CPU or CUDA: RNE E2M1, E4M3/16.

    This is an offline calibration operation; runtime does the conversion in
    its CUDA producer. Zero groups may use zero SFA; weight scales stay >0.
    """
    if x.ndim != 2 or x.shape[1] % 16 or not bool(torch.isfinite(x).all()):
        raise ValueError("P4 activation calibration needs finite [M,K16] inputs")
    blocks = x.float().reshape(x.shape[0], -1, 16)
    maximum = blocks.abs().amax(-1)
    scales = (maximum / 6).clamp(2**-9, 448).to(torch.float8_e4m3fn).float()
    scales = torch.where(maximum == 0, 0, scales)
    values = blocks * torch.where(scales == 0, 0, scales.reciprocal())[..., None]
    levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=x.device)
    preference = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=x.device)
    distances = (values.abs()[..., None] - levels[preference]).abs()
    codes = preference[distances.argmin(-1)]
    return (torch.copysign(levels[codes], values) * scales[..., None]).reshape_as(x)


def payload_reconstruction(payload: P4Payload, *, device="cpu") -> torch.Tensor:
    """Vectorized offline reconstruction; never imported by serving hot path."""
    validate_payload(payload)
    stream = torch.from_numpy(payload.trellis).to(device)
    states = reconstruct_trellis_states(unpack_trellis_edges(stream, 4), 4)
    normalized = _states_to_normalized_blocks(states, encoder_state_lut(device=device),
                                               rows=payload.rows, width=payload.width)
    scales = torch.from_numpy(payload.scale_e4m3).to(device).view(torch.float8_e4m3fn).float()
    return (normalized * scales[..., None] * float(payload.global_scale)).reshape(payload.rows, payload.width)


def _positive_scales(scales: torch.Tensor) -> torch.Tensor:
    return scales.float().clamp(2**-9, 448).to(torch.float8_e4m3fn).contiguous()


def refit_scales(weight: torch.Tensor, normalized: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
    """Exact fixed-code SSE minimum over all 126 positive E4M3 values."""
    rows, width = weight.shape
    basis = normalized.reshape(rows, width // 16, 16).double() * global_scale.double()
    target = weight.double().reshape_as(basis)
    numerator, denominator = (basis * target).sum(-1), basis.square().sum(-1)
    optimum = numerator / denominator.clamp_min(torch.finfo(torch.float64).tiny)
    raw = torch.arange(1, 127, dtype=torch.uint8, device=weight.device)
    values = raw.view(torch.float8_e4m3fn).double()
    upper = torch.searchsorted(values, optimum.contiguous()).clamp(0, 125)
    lower = (upper - 1).clamp(0, 125)
    dlow, dhigh = (optimum - values[lower]).abs(), (optimum - values[upper]).abs()
    choose_upper = (dhigh < dlow) | ((dhigh == dlow) & (((upper + 1) & 1) == 0))
    selected = torch.where(choose_upper, upper, lower).to(torch.uint8) + 1
    selected = torch.where(denominator == 0, 0x38, selected).to(torch.uint8)
    return selected.view(torch.float8_e4m3fn).contiguous()


def refit_global(weight: torch.Tensor, normalized: torch.Tensor, scales: torch.Tensor,
                 previous: torch.Tensor) -> torch.Tensor:
    rows, width = weight.shape
    basis = normalized.reshape(rows, width // 16, 16).double() * scales.double()[..., None]
    denominator = basis.square().sum()
    numerator = (basis * weight.double().reshape_as(basis)).sum()
    if float(denominator) == 0 or float(numerator) <= 0:
        return previous
    fitted = (numerator / denominator).float()
    return fitted if bool(torch.isfinite(fitted)) and float(fitted) > 0 else previous


def payload_from_states(states: torch.Tensor, scales: torch.Tensor, global_scale: torch.Tensor,
                        *, rows: int, width: int) -> P4Payload:
    """Export only physically representable closed paths and positive scales."""
    if states.shape != (width // 16, rows // 16, 256):
        raise ValueError("P4 state geometry mismatch")
    if states.dtype not in (torch.int16, torch.int32, torch.int64, torch.uint16):
        raise ValueError("P4 states must be integer 16-bit labels")
    values = states.detach().cpu().to(torch.int64)
    if states.dtype == torch.int16:
        values &= 65535  # signed int16 is the existing backend's state carrier
    if bool(((values < 0) | (values > 65535)).any()):
        raise ValueError("P4 states must be in 0..65535")
    if scales.dtype != torch.float8_e4m3fn or tuple(scales.shape) != (rows, width // 16):
        raise ValueError("P4 scale geometry/dtype mismatch")
    payload = P4Payload(rows, width,
                        pack_k4_states(values.numpy().astype("<u2")),
                        scales.detach().cpu().contiguous().view(torch.uint8).numpy().copy(),
                        global_scale.detach().cpu().reshape(()).float().numpy().copy())
    validate_payload(payload)
    return payload


@torch.no_grad()
def encode_p4_matrix(weight: torch.Tensor, hessian: torch.Tensor, *, global_scale: float | torch.Tensor | None = None,
                     scale_refinement_iterations: int = 2, search_grid: int = 12,
                     tailbite_context: int = 128, percdamp: float = 0.01,
                     column_block: int = 128) -> P4Encoding:
    """Real Viterbi + GPTQ output-error feedback -> exact native P4 matrix.

    Uses the full supplied calibration Hessian, native-group static act order,
    and joint block/global scale refits. No LDLQ is used. The Cholesky inverse
    and static permutation are prepared once and reused across refit passes.
    """
    if weight.device.type != "cuda" or weight.ndim != 2 or not weight.numel() or any(d % 16 for d in weight.shape):
        raise ValueError("P4 Viterbi requires one CUDA matrix aligned to 16x16")
    if scale_refinement_iterations < 0 or type(scale_refinement_iterations) is not int:
        raise ValueError("invalid P4 scale refinement count")
    if type(tailbite_context) is not int or not 1 <= tailbite_context <= 128 or search_grid <= 0:
        raise ValueError("invalid Viterbi tailbite context or scale search grid")
    weight = weight.float()
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("P4 source weights must be finite")
    gs = torch.as_tensor(global_scale, dtype=torch.float32, device=weight.device).reshape(()) if global_scale is not None else weight.abs().max().div(4 * 448).clamp_min(torch.finfo(torch.float32).tiny)
    if not bool(torch.isfinite(gs)) or float(gs) <= 0:
        raise ValueError("P4 global scale must be positive finite FP32")
    hessian = hessian.to(device=weight.device, dtype=torch.float32)
    if hessian.shape != (weight.shape[1], weight.shape[1]) or not bool(torch.isfinite(hessian).all()):
        raise ValueError("P4 requires a finite full calibration Hessian")
    if not torch.allclose(hessian, hessian.T, atol=1e-6, rtol=1e-5):
        raise ValueError("P4 Hessian must be symmetric")
    if not math.isfinite(percdamp) or percdamp <= 0 or column_block <= 0 or column_block % 16:
        raise ValueError("invalid P4 GPTQ damping or native column block")
    prepared = _prepare_full_inverse(hessian, percdamp, 16)
    lut = encoder_state_lut(device=weight.device)
    scales = _positive_scales(_prepare_tiles(weight, gs, search_grid)[1])
    history = []
    normalized = states = None
    for iteration in range(scale_refinement_iterations + 1):
        _, states = feedback_select(weight, hessian, scales, gs, lut,
            tailbite_context=tailbite_context, percdamp=percdamp, column_block=column_block,
            prepared_inverse=prepared, validate_inputs=False)
        normalized = _states_to_normalized_blocks(states, lut, rows=weight.shape[0], width=weight.shape[1])
        if iteration < scale_refinement_iterations:
            scales = refit_scales(weight, normalized, gs)
            gs = refit_global(weight, normalized, scales, gs)
            scales = refit_scales(weight, normalized, gs)
        reconstruction = normalized.double() * scales.double()[..., None] * gs.double()
        sse = (weight.double().reshape_as(reconstruction) - reconstruction).square().sum()
        norm = weight.double().square().sum()
        history.append({"iteration": iteration, "weight_sse": float(sse),
                        "weight_nmse": float(sse / norm) if float(norm) else None,
                        "global_scale": float(gs)})
    rows, width = weight.shape
    payload = payload_from_states(states.reshape(width // 16, rows // 16, 256), scales, gs,
                                  rows=rows, width=width)
    # Exact labels already came from the physical law. Recover native nibble
    # sign via signbit, including -0, rather than the legacy zero canonicalizer.
    basis = normalized.reshape(rows, width)
    levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=weight.device)
    codes = (basis.abs()[..., None] - levels).abs().argmin(-1).to(torch.uint8)
    codes |= torch.signbit(basis).to(torch.uint8) << 3
    endpoint = PackedNVFP4(pack_codes(codes).cpu(), scales.cpu(), gs.cpu())
    error = weight - reconstruction.reshape_as(weight).float()
    output_sse = ((error @ hessian) * error).double().sum()
    output_norm = ((weight @ hessian) * weight).double().sum()
    report = {"law": LAW, "encoder": "kquant-viterbi-native-rne-lut",
              "objective": "viterbi-with-full-hessian-gptq-feedback",
              "scale_subobjective": "fixed-code-weight-sse-block-global-coordinate-refit",
              "ldlq": False, "calibration_used": True, "bits": 4,
              "hessian_sha256": hashlib.sha256(hessian.cpu().numpy().tobytes()).hexdigest(),
              "hessian_shape": list(hessian.shape), "act_order": "static-in-group-16",
              "prepared_inverse_count": 1, "percdamp": percdamp, "column_block": column_block,
              "calibration_output_nmse": float(output_sse / output_norm) if float(output_norm) else None,
              "scale_refinement_iterations": scale_refinement_iterations,
              "tailbite_context": tailbite_context, "encoder_lut_sha256": hashlib.sha256(lut.cpu().numpy().tobytes()).hexdigest(),
              "encoder_only_lut_bytes": lut.numel(), "runtime_state_lut_bytes": 0, "history": history}
    return P4Encoding(payload, endpoint, report)
