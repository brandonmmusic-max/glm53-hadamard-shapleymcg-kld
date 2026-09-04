"""Independent scalar CPU oracle for the P4 runtime, never imported by its hot path.

The law/layout are ports of ExLlamaV3 via the repository's KQuant/QSRT codec.
See LICENSE.exllamav3 and THIRD_PARTY_NOTICES.md. This oracle uses IEEE half
arithmetic and explicit symbol reconstruction, independently of the kernel's
integer fixed-point law and two-word sliding window.
"""
from __future__ import annotations

import hashlib
import struct

import torch


LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def mcg_code(state: int) -> int:
    product = (state * 0xCBAC1FED) % (1 << 32)
    word = (product & 0x8FFF8FFF) ^ 0x3B603B60
    lo, hi = struct.unpack("<ee", struct.pack("<I", word))
    value = struct.unpack("<e", struct.pack("<e", lo + hi))[0]
    magnitude = min(range(8), key=lambda i: abs(abs(value) - LEVELS[i]))
    return magnitude | (8 if value < 0 and magnitude else 0)


def tile_states(raw: bytes) -> list[int]:
    if len(raw) != 128:
        raise ValueError("one K4 tile is exactly 128 bytes")
    words = struct.unpack("<64H", raw)
    edges = [
        (words[index ^ 1] >> shift) & 15
        for index in range(64)
        for shift in (12, 8, 4, 0)
    ]
    states = []
    for end in range(256):
        state = 0
        for position in range(end - 3, end + 1):
            state = state * 16 + edges[position % 256]
        states.append(state)
    return states


def decode_projection(stream: torch.Tensor) -> torch.Tensor:
    """[E,K/16,N/16,64] int16 -> logical [E,N,K/2] packed uint8, CPU only."""
    if stream.device.type != "cpu" or stream.dtype != torch.int16 or stream.ndim != 4:
        raise ValueError("reference expects a CPU int16 projection stream")
    experts, kt, nt, words = stream.shape
    if words != 64:
        raise ValueError("K4 requires 64 int16 words per tile")
    # Construct the forward encoder permutation; do not call the kernel's
    # closed-form inverse or its byte/nibble addressing helpers.
    coordinates = []
    for thread in range(32):
        input0, output0 = (thread % 4) * 2, thread // 4
        coordinates.extend(
            (output0 + dc, input0 + dr)
            for dc in (0, 8) for dr in (0, 1, 8, 9)
        )
    raw = stream.contiguous().numpy().tobytes()
    packed = bytearray(experts * nt * 16 * kt * 8)
    for expert in range(experts):
        for ki in range(kt):
            for ni in range(nt):
                tile = ((expert * kt + ki) * nt + ni) * 128
                for (no, ko), state in zip(coordinates, tile_states(raw[tile:tile + 128]), strict=True):
                    pos = ((expert * nt * 16 + ni * 16 + no) * kt * 16 + ki * 16 + ko)
                    packed[pos // 2] |= mcg_code(state) << ((pos % 2) * 4)
    return torch.frombuffer(packed, dtype=torch.uint8).clone().reshape(experts, nt * 16, kt * 8)


def unpack_values(packed: torch.Tensor) -> torch.Tensor:
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
    levels = torch.tensor(LEVELS, dtype=torch.float32)
    return levels[codes & 7] * torch.where(codes < 8, 1.0, -1.0)


def qdq_activations(x: torch.Tensor) -> torch.Tensor:
    """The candidate's explicit RN-even NVFP4 activation policy, CPU only."""
    if x.device.type != "cpu":
        raise ValueError("P4 reference must stay on CPU")
    blocks = x.float().reshape(*x.shape[:-1], -1, 16)
    maximum = blocks.abs().amax(-1)
    sf = (maximum / 6).clamp(min=2**-9, max=448).to(torch.float8_e4m3fn).float()
    sf = torch.where(maximum == 0, 0, sf)
    inv = torch.where(sf == 0, 0, 1 / sf)
    values = blocks * inv[..., None]
    levels = torch.tensor(LEVELS)
    distances = (values.abs()[..., None] - levels).abs()
    # Native activation cvt uses nearest/even; weights use nearest/lower.
    # Check the even indices first to disambiguate exact midpoint ties.
    preference = torch.tensor((0, 2, 4, 6, 1, 3, 5, 7))
    codes = preference[distances[..., preference].argmin(-1)]
    return (levels[codes] * values.sign() * sf[..., None]).reshape_as(x)


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def synthetic_payload() -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Small public synthetic fixture. No model, corpus, or role data."""
    from .trellis_nvfp4 import pack_trellis_edges

    experts, hidden, intermediate = 3, 128, 64
    generator = torch.Generator(device="cpu").manual_seed(20260904)

    def projection(n: int, k: int, leading: tuple[int, ...]):
        edges = torch.randint(0, 16, (*leading, k // 16, n // 16, 256), generator=generator)
        trellis = pack_trellis_edges(edges, 4)
        # Deliberately non-unit, nonuniform, finite, positive residual bytes.
        scales = torch.randint(1, 111, (*leading, n, k // 16), generator=generator, dtype=torch.uint8)
        return trellis, scales

    w13, s13 = projection(intermediate, hidden, (2, experts))
    w2, s2 = projection(hidden, intermediate, (experts,))
    tensors = {
        "w13_trellis": w13, "w2_trellis": w2,
        "w13_scale_e4m3": s13, "w2_scale_e4m3": s2,
        "w13_global_scale": torch.tensor([[0.125, 0.25, 0.0625], [0.25, 0.125, 0.5]]),
        "w2_global_scale": torch.tensor([0.125, 0.25, 0.0625]),
    }
    metadata = {
        "schema": "glm53-p4-mcg-tp-rank.v1", "role": "physical-codec",
        "layer": "3", "rank": "0", "world_size": "4", "bits": "4",
        "alphabet": "e2m1", "scale": "e4m3-k16", "law": "procedural-mcg",
        "compander": "1", "weight_rounding": "nearest-ties-low-magnitude",
        "boundary": "identity", "w13_order": "gate,up", "ldlq": "false",
        "source_design_sha256": hashlib.sha256(b"p4-astra-synthetic-fixture-v1").hexdigest(),
    }
    return tensors, metadata


def moe_reference(tensors: dict[str, torch.Tensor], x: torch.Tensor,
                  ids: torch.Tensor, weights: torch.Tensor, limit: float = 10.0,
                  *, return_intermediates: bool = False):
    """Float32 CPU matmul oracle, separate from the runtime endpoint."""
    gu = torch.stack([unpack_values(decode_projection(s)) for s in tensors["w13_trellis"]])
    down = unpack_values(decode_projection(tensors["w2_trellis"]))
    gu *= tensors["w13_scale_e4m3"].view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)
    gu *= tensors["w13_global_scale"][:, :, None, None]
    down *= tensors["w2_scale_e4m3"].view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)
    down *= tensors["w2_global_scale"][:, None, None]
    result = torch.zeros_like(x, dtype=torch.float32)
    stages_gu, stages_mid, stages_down = [], [], []
    qx = qdq_activations(x)
    for token in range(x.shape[0]):
        for slot in range(ids.shape[1]):
            e = int(ids[token, slot])
            gate = gu[0, e] @ qx[token]
            up = gu[1, e] @ qx[token]
            middle = torch.nn.functional.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)
            partial = down[e] @ qdq_activations(middle)
            result[token] += partial * weights[token, slot]
            if return_intermediates:
                stages_gu.append(torch.stack((gate, up)))
                stages_mid.append(middle)
                stages_down.append(partial)
    if return_intermediates:
        return result, torch.stack(stages_gu), torch.stack(stages_mid), torch.stack(stages_down)
    return result
