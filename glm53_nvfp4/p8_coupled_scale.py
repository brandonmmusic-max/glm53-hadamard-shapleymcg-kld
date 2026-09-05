"""CPU reference and metadata contract for the coupled P8 scale sandwich.

The transform follows the coupled H512/H128 ordering implemented by the
archived Luke/QSRT reference.  It intentionally does not depend on the CUDA
runtime so encoder artifacts can be rejected before any device work starts.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from .canary_mxfp6_reap import _qdq_e4m3_k32
from .shard_index import sha256_file


COUPLED_BOUNDARY = "coupled-h512-h128-suh-svh-v1"
COUPLED_CHUNK_SCHEMA = "glm53-hessian-trellis-p8-coupled-scale-layer-chunk.v2"
COUPLED_RANK_SCHEMA = "glm53-p8-coupled-h512-h128-tp4-rank.v1"
COUPLED_SIDECAR_SCHEMA = "glm53-p8-mcg-coupled-scale-tp4-sidecars.v2"
COUPLED_COMPONENT = "p8-coupled-h512-h128-sign-v1"
COUPLED_CAST_ORDER = "luke-qsrt-coupled-reference-v1"
COUPLED_TRANSFORM_ID = "normalized-h512-outer-h128-inner-sign-draw0-v1"
COUPLED_SIGN_GENERATOR = "qsrt-coupled-signs-v1"
COUPLED_SIGN_DRAW = 0
COUPLED_ACTIVATION = "silu-cap10"
COUPLED_FC1_INTERLEAVE = "slot0-atom32-slot1-atom32-v1"
COUPLED_TP_SLICE = "contiguous-atom32-v1"
COUPLED_QUANTIZED_INPUT_ORDER = (
    "bf16-cvt-rn-fp16-h512-fp32-suh-fp32-h128-fp32-e4m3-ue8m0-k32"
)
COUPLED_QUANTIZED_DOWN_ORDER = (
    "silu-cap10-fp32-sign-h128-fp32-down-suh-fp32-h128-fp32-e4m3-ue8m0-k32"
)
SUPPORTED_LAYERS = (3, 20, 22)
HADAMARD_INPUT = 128
HADAMARD_RESIDUAL = 512
ACTIVATION_LIMIT = 10.0


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def block_hadamard(
    values: torch.Tensor, *, block_size: int, dim: int = -1
) -> torch.Tensor:
    """Apply a normalized self-inverse block Walsh-Hadamard transform."""
    if values.ndim == 0 or not torch.is_floating_point(values):
        raise TypeError("block Hadamard input must be floating point")
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")
    axis = dim % values.ndim
    if values.shape[axis] % block_size:
        raise ValueError("Hadamard axis must be divisible by block_size")
    if values.numel() == 0:
        return values.float().contiguous().clone()
    output = values.float().movedim(axis, -1).contiguous().clone()
    shape = output.shape
    output = output.reshape(*shape[:-1], shape[-1] // block_size, block_size)
    width = 1
    while width < block_size:
        paired = output.reshape(*output.shape[:-1], -1, 2, width)
        left = paired[..., 0, :].clone()
        right = paired[..., 1, :].clone()
        paired[..., 0, :] = left + right
        paired[..., 1, :] = left - right
        output = paired.reshape(*output.shape)
        width *= 2
    return (
        output.div_(math.sqrt(block_size))
        .reshape(shape)
        .movedim(-1, axis)
        .contiguous()
    )


def rotation_signs(length: int, *, draw: int, axis: int) -> torch.Tensor:
    """Port QSRT's frozen procedural Rademacher family exactly on CPU."""
    if length <= 0 or draw < 0:
        raise ValueError("length must be positive and draw nonnegative")
    if draw == 0:
        return torch.ones(length, dtype=torch.float32)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        (0x6A09E667F3BCC909 * draw + 0xBB67AE8584CAA73B * axis)
        & ((1 << 63) - 1)
    )
    return torch.randint(0, 2, (length,), generator=generator).mul_(2).sub_(1).float()


def rank_local_coupled_signs(
    *, intermediate: int, rank: int, world_size: int = 4
) -> torch.Tensor:
    """Return runtime's packed rank-local ``pre[2I] | post[I]`` FP16 signs."""
    if intermediate % 32 or world_size != 4 or rank not in range(world_size):
        raise ValueError("coupled signs require an atom32-aligned TP4 slice")
    global_intermediate = intermediate * world_size
    first_atom = rank * (intermediate // 32)
    pre = rotation_signs(
        2 * global_intermediate, draw=COUPLED_SIGN_DRAW, axis=1
    )
    post = rotation_signs(
        global_intermediate, draw=COUPLED_SIGN_DRAW, axis=2
    )
    pre_begin = 2 * first_atom * 32
    post_begin = first_atom * 32
    return torch.cat(
        (
            pre[pre_begin : pre_begin + 2 * intermediate],
            post[post_begin : post_begin + intermediate],
        )
    ).to(torch.float16).contiguous()


def _validate_scale(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> None:
    if value.dtype != torch.float16 or tuple(value.shape) != shape or not value.is_contiguous():
        raise TypeError(f"{name} must be contiguous FP16 {shape}")
    if not bool(torch.isfinite(value).all()) or bool((value == 0).any()):
        raise ValueError(f"{name} must contain finite nonzero signed scales")


@dataclass(frozen=True)
class CoupledScaleSet:
    gate_up_suh: torch.Tensor
    gate_svh: torch.Tensor
    up_svh: torch.Tensor
    down_suh: torch.Tensor
    down_svh: torch.Tensor
    source_path: Path
    source_sha256: str
    source_metadata: dict[str, str]
    tensor_hashes: dict[str, str]

    @property
    def experts(self) -> int:
        return int(self.gate_svh.shape[0])

    @property
    def hidden(self) -> int:
        return int(self.gate_up_suh.numel())

    @property
    def intermediate(self) -> int:
        return int(self.gate_svh.shape[1])

    @property
    def metadata_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.gate_up_suh,
                self.gate_svh,
                self.up_svh,
                self.down_suh,
                self.down_svh,
            )
        )

    def validate(self) -> None:
        _validate_scale("gate_up_suh", self.gate_up_suh, (self.hidden,))
        _validate_scale("gate_svh", self.gate_svh, (self.experts, self.intermediate))
        _validate_scale("up_svh", self.up_svh, (self.experts, self.intermediate))
        _validate_scale("down_suh", self.down_suh, (self.experts, self.intermediate))
        _validate_scale("down_svh", self.down_svh, (self.hidden,))

    def tensors(self) -> dict[str, torch.Tensor]:
        return {
            "gate_up_suh_fp16": self.gate_up_suh,
            "intermediate_scales_fp16": torch.cat(
                (self.gate_svh, self.up_svh, self.down_suh), dim=1
            ).contiguous(),
            "down_svh_fp16": self.down_svh,
        }


def _expert_bases(keys: list[str], layer: int) -> dict[int, str]:
    suffix = ".gate_proj.suh"
    marker = f".layers.{layer}.mlp.experts."
    found: dict[int, str] = {}
    for key in keys:
        if marker not in key or not key.endswith(suffix):
            continue
        base = key[: -len(suffix)]
        expert_text = base.split(marker, 1)[1]
        if not expert_text.isdigit():
            continue
        found[int(expert_text)] = base
    return found


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


COUPLED_TRANSFORM_CONTRACT = {
    "activation": COUPLED_ACTIVATION,
    "boundary": COUPLED_BOUNDARY,
    "cast_order": COUPLED_CAST_ORDER,
    "fc1_interleave": COUPLED_FC1_INTERLEAVE,
    "h128": "normalized-sylvester-128-v1",
    "h512": "normalized-sylvester-512-v1",
    "quantized_down_order": COUPLED_QUANTIZED_DOWN_ORDER,
    "quantized_input_order": COUPLED_QUANTIZED_INPUT_ORDER,
    "sign_draw": COUPLED_SIGN_DRAW,
    "sign_generator": COUPLED_SIGN_GENERATOR,
    "sign_post_axis": 2,
    "sign_pre_axis": 1,
    "tp_slice": COUPLED_TP_SLICE,
    "transform_id": COUPLED_TRANSFORM_ID,
}
COUPLED_TRANSFORM_SHA256 = hashlib.sha256(
    _canonical_json(COUPLED_TRANSFORM_CONTRACT)
).hexdigest()


def _verify_json_seal(value: dict[str, object], field: str) -> str:
    digest = value.get(field)
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError(f"EXL3 checkpoint receipt lacks {field}")
    body = dict(value)
    del body[field]
    if hashlib.sha256(_canonical_json(body)).hexdigest() != digest:
        raise RuntimeError(f"EXL3 checkpoint {field} seal differs")
    return digest


def _scale_set_from_tensors(
    tensors_by_name: dict[str, torch.Tensor],
    *,
    layer: int,
    source_path: Path,
    source_sha256: str,
    source_metadata: dict[str, str],
    expected_experts: int | None,
    expected_hidden: int | None,
    expected_intermediate: int | None,
) -> CoupledScaleSet:
    bases = _expert_bases(list(tensors_by_name), layer)
    if not bases or sorted(bases) != list(range(len(bases))):
        raise RuntimeError("EXL3 scale source does not contain contiguous experts from zero")
    if expected_experts is not None and len(bases) != expected_experts:
        raise RuntimeError(
            f"EXL3 scale source has {len(bases)} experts, expected {expected_experts}"
        )
    shared_gate: torch.Tensor | None = None
    shared_down: torch.Tensor | None = None
    gate_private: list[torch.Tensor] = []
    up_private: list[torch.Tensor] = []
    down_private: list[torch.Tensor] = []
    raw_hashes: dict[str, str] = {}
    for expert, base in sorted(bases.items()):
        tensors = {
            "gate_suh": tensors_by_name[f"{base}.gate_proj.suh"].contiguous(),
            "gate_svh": tensors_by_name[f"{base}.gate_proj.svh"].contiguous(),
            "up_suh": tensors_by_name[f"{base}.up_proj.suh"].contiguous(),
            "up_svh": tensors_by_name[f"{base}.up_proj.svh"].contiguous(),
            "down_suh": tensors_by_name[f"{base}.down_proj.suh"].contiguous(),
            "down_svh": tensors_by_name[f"{base}.down_proj.svh"].contiguous(),
        }
        hidden = int(tensors["gate_suh"].numel())
        intermediate = int(tensors["gate_svh"].numel())
        for name, shape in (
            ("gate_suh", (hidden,)),
            ("up_suh", (hidden,)),
            ("gate_svh", (intermediate,)),
            ("up_svh", (intermediate,)),
            ("down_suh", (intermediate,)),
            ("down_svh", (hidden,)),
        ):
            _validate_scale(f"expert {expert} {name}", tensors[name], shape)
            raw_hashes[f"expert.{expert}.{name}"] = _tensor_sha256(tensors[name])
        if not torch.equal(tensors["gate_suh"], tensors["up_suh"]):
            raise RuntimeError(f"expert {expert} gate/up suh are not byte-identical")
        if shared_gate is None:
            shared_gate = tensors["gate_suh"]
            shared_down = tensors["down_svh"]
        elif not torch.equal(shared_gate, tensors["gate_suh"]):
            raise RuntimeError("gate/up suh is not shared across every expert")
        elif not torch.equal(shared_down, tensors["down_svh"]):
            raise RuntimeError("down svh is not shared across every expert")
        gate_private.append(tensors["gate_svh"])
        up_private.append(tensors["up_svh"])
        down_private.append(tensors["down_suh"])
    assert shared_gate is not None and shared_down is not None
    found_hidden = int(shared_gate.numel())
    found_intermediate = int(gate_private[0].numel())
    if expected_hidden is not None and found_hidden != expected_hidden:
        raise RuntimeError(
            f"EXL3 scale source hidden size {found_hidden}, expected {expected_hidden}"
        )
    if expected_intermediate is not None and found_intermediate != expected_intermediate:
        raise RuntimeError(
            "EXL3 scale source intermediate size "
            f"{found_intermediate}, expected {expected_intermediate}"
        )
    result = CoupledScaleSet(
        gate_up_suh=shared_gate,
        gate_svh=torch.stack(gate_private).contiguous(),
        up_svh=torch.stack(up_private).contiguous(),
        down_suh=torch.stack(down_private).contiguous(),
        down_svh=shared_down,
        source_path=source_path,
        source_sha256=source_sha256,
        source_metadata=source_metadata,
        tensor_hashes=raw_hashes,
    )
    result.validate()
    return result


def _load_indexed_exl3_scales(
    root: Path,
    *,
    layer: int,
    expected_experts: int | None,
    expected_hidden: int | None,
    expected_intermediate: int | None,
) -> CoupledScaleSet:
    """Load a layer from a sealed, materialized GLM-5.3 EXL3 checkpoint."""
    receipt_path = root / "materialization-receipt.json"
    index_path = root / "model.safetensors.index.json"
    config_path = root / "config.json"
    quant_path = root / "quantization_config.json"
    for required in (receipt_path, index_path, config_path, quant_path):
        if not required.is_file() or required.is_symlink():
            raise FileNotFoundError(required)
    receipt = json.loads(receipt_path.read_text())
    receipt_sha = _verify_json_seal(receipt, "receipt_sha256")
    if (
        receipt.get("schema") != "quant-pipeline.glm53-k4-materialization-receipt.v1"
        or receipt.get("complete") is not True
        or receipt.get("codec_family") != "exl3-mcg"
        or receipt.get("mcg_multiplier_hex") != "0xCBAC1FED"
        or receipt.get("bits") != 4
        or receipt.get("main_and_mtp_complete") is not True
        or receipt.get("nonrouted_native_exact") is not True
    ):
        raise RuntimeError("EXL3 checkpoint materialization semantics differ")
    for path, field in (
        (index_path, "index_sha256"),
        (config_path, "config_sha256"),
        (quant_path, "quantization_config_sha256"),
    ):
        if sha256_file(path) != receipt.get(field):
            raise RuntimeError(f"EXL3 checkpoint {field} differs")
    config = json.loads(config_path.read_text())
    text = config.get("text_config", {})
    if not isinstance(text, dict):
        raise RuntimeError("EXL3 checkpoint lacks GLM text geometry")
    if expected_hidden is not None and text.get("hidden_size") != expected_hidden:
        raise RuntimeError("EXL3 checkpoint text hidden size differs")
    if expected_experts is not None and text.get("n_routed_experts") != expected_experts:
        raise RuntimeError("EXL3 checkpoint routed expert count differs")
    if (
        expected_intermediate is not None
        and text.get("moe_intermediate_size") != expected_intermediate
    ):
        raise RuntimeError("EXL3 checkpoint MoE intermediate size differs")
    quant = json.loads(quant_path.read_text())
    if (
        quant.get("quant_method") != "exl3"
        or quant.get("codebook") != "mcg"
        or quant.get("bits") != 4
    ):
        raise RuntimeError("EXL3 checkpoint quantization contract differs")
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError("EXL3 checkpoint index lacks weight_map")
    marker = f".layers.{layer}.mlp.experts."
    scale_names = sorted(
        name
        for name in weight_map
        if marker in name and name.endswith((".suh", ".svh"))
    )
    if expected_experts is not None and len(scale_names) != expected_experts * 6:
        raise RuntimeError("EXL3 checkpoint layer scale tensor census differs")
    by_shard: dict[str, list[str]] = {}
    for name in scale_names:
        shard = weight_map[name]
        if not isinstance(shard, str) or not shard:
            raise RuntimeError("EXL3 checkpoint index contains an invalid shard")
        by_shard.setdefault(shard, []).append(name)
    top_shards = receipt.get("shard_sha256")
    top_receipts = receipt.get("shard_receipt_sha256")
    if not isinstance(top_shards, dict) or not isinstance(top_receipts, list):
        raise RuntimeError("EXL3 checkpoint lacks shard receipt closure")
    tensors: dict[str, torch.Tensor] = {}
    for shard, names in sorted(by_shard.items()):
        shard_path = root / shard
        shard_receipt_path = root / ".materialization" / "shards" / f"{shard}.json"
        if not shard_path.is_file() or shard_path.is_symlink():
            raise FileNotFoundError(shard_path)
        if not shard_receipt_path.is_file() or shard_receipt_path.is_symlink():
            raise FileNotFoundError(shard_receipt_path)
        shard_receipt = json.loads(shard_receipt_path.read_text())
        shard_receipt_sha = _verify_json_seal(shard_receipt, "receipt_sha256")
        if (
            shard_receipt.get("schema")
            != "quant-pipeline.glm53-k4-materialized-shard-receipt.v1"
            or shard_receipt_sha not in top_receipts
            or shard_receipt.get("plan_sha256") != receipt.get("plan_sha256")
            or shard_receipt.get("shard") != shard
            or shard_receipt.get("shard_sha256") != top_shards.get(shard)
            or shard_receipt.get("complete") is not True
            or shard_path.stat().st_size != shard_receipt.get("shard_bytes")
        ):
            raise RuntimeError(f"EXL3 checkpoint shard receipt differs: {shard}")
        rows = {
            row.get("name"): row
            for row in shard_receipt.get("tensors", [])
            if isinstance(row, dict)
        }
        with safe_open(shard_path, framework="pt", device="cpu") as source:
            for name in names:
                if name not in source.keys() or name not in rows:
                    raise RuntimeError(f"EXL3 checkpoint scale is absent: {name}")
                tensor = source.get_tensor(name).contiguous()
                if (
                    rows[name].get("origin") != "sealed_exl3_mcg_packed_choice"
                    or rows[name].get("payload_sha256") != _tensor_sha256(tensor)
                ):
                    raise RuntimeError(f"EXL3 checkpoint scale payload differs: {name}")
                tensors[name] = tensor
    return _scale_set_from_tensors(
        tensors,
        layer=layer,
        source_path=root,
        source_sha256=receipt_sha,
        source_metadata={
            "format": "indexed-materialized-exl3-k4",
            "receipt_sha256": receipt_sha,
            "index_sha256": str(receipt["index_sha256"]),
            "source_model_revision": str(receipt.get("source_model_revision", "")),
        },
        expected_experts=expected_experts,
        expected_hidden=expected_hidden,
        expected_intermediate=expected_intermediate,
    )


def load_exact_exl3_scales(
    path: Path,
    *,
    layer: int,
    expected_experts: int | None = None,
    expected_hidden: int | None = None,
    expected_intermediate: int | None = None,
) -> CoupledScaleSet:
    """Load and hash the exact EXL3 scale tensors for one GLM layer.

    Shared vectors must be byte-identical across every expert.  The function
    deliberately rejects dimension mismatch rather than resizing, repeating,
    or learning a substitute vector.
    """
    path = Path(path).resolve()
    if layer not in SUPPORTED_LAYERS:
        raise ValueError(f"coupled-scale preparation is limited to layers {SUPPORTED_LAYERS}")
    if path.is_dir():
        return _load_indexed_exl3_scales(
            path,
            layer=layer,
            expected_experts=expected_experts,
            expected_hidden=expected_hidden,
            expected_intermediate=expected_intermediate,
        )
    if not path.is_file():
        raise FileNotFoundError(path)
    with safe_open(path, framework="pt", device="cpu") as src:
        metadata = src.metadata() or {}
        if metadata.get("codec") not in {None, "exl3-mcg"}:
            raise RuntimeError("EXL3 scale source has an unexpected codec")
        if metadata.get("layer") not in {None, str(layer)}:
            raise RuntimeError("EXL3 scale source layer metadata mismatch")
        tensors_by_name = {
            name: src.get_tensor(name).contiguous() for name in src.keys()
        }
    return _scale_set_from_tensors(
        tensors_by_name,
        layer=layer,
        source_path=path,
        source_sha256=sha256_file(path),
        source_metadata=dict(metadata),
        expected_experts=expected_experts,
        expected_hidden=expected_hidden,
        expected_intermediate=expected_intermediate,
    )


def _atom_interleave(slot0: torch.Tensor, slot1: torch.Tensor) -> torch.Tensor:
    if slot0.shape != slot1.shape or slot0.shape[-1] % 32:
        raise ValueError("coupled slots must be aligned and divisible by 32")
    atoms = slot0.shape[-1] // 32
    prefix = slot0.shape[:-1]
    return torch.stack(
        (slot0.reshape(*prefix, atoms, 32), slot1.reshape(*prefix, atoms, 32)),
        dim=-2,
    ).reshape(*prefix, 2 * slot0.shape[-1])


def _atom_deinterleave_rows(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 2 or values.shape[0] % 64:
        raise ValueError("coupled row slab must have 64-row atom pairs")
    atoms = values.shape[0] // 64
    view = values.reshape(atoms, 2, 32, values.shape[1])
    return view[:, 0].reshape(-1, values.shape[1]), view[:, 1].reshape(-1, values.shape[1])


def encode_coupled_scale_weights(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    scales: CoupledScaleSet,
    *,
    expert: int,
    intermediate_draw: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map one source expert to the exact physical coupled-scale coordinates."""
    if gate.shape != up.shape or down.shape != (gate.shape[1], gate.shape[0]):
        raise ValueError("expected gate/up [I,H] and down [H,I]")
    intermediate, hidden = gate.shape
    if (scales.hidden, scales.intermediate) != (hidden, intermediate):
        raise ValueError("scale geometry does not match source weights")
    if not 0 <= expert < scales.experts or not 0 <= intermediate_draw < 8:
        raise ValueError("expert/draw is outside the coupled scale contract")
    device = gate.device
    source_suh = scales.gate_up_suh.to(device=device).float()
    gate_svh = scales.gate_svh[expert].to(device=device).float()
    up_svh = scales.up_svh[expert].to(device=device).float()
    down_suh = scales.down_suh[expert].to(device=device).float()
    down_svh = scales.down_svh.to(device=device).float()
    pre_signs = rotation_signs(2 * intermediate, draw=intermediate_draw, axis=1).to(device)
    post_signs = rotation_signs(intermediate, draw=intermediate_draw, axis=2).to(device)

    # Desired logical preactivations are neuron-interleaved [g0,u0,g1,u1,...].
    logical = torch.stack((gate.float(), up.float()), dim=1).reshape(2 * intermediate, hidden)
    raw_rows = logical * pre_signs[:, None]
    raw_rows = block_hadamard(raw_rows, block_size=HADAMARD_INPUT, dim=0)
    pair_scales = _atom_interleave(gate_svh[None], up_svh[None])[0]
    raw_rows = raw_rows / pair_scales[:, None]
    raw_rows = block_hadamard(raw_rows, block_size=HADAMARD_INPUT, dim=0)
    slot0, slot1 = _atom_deinterleave_rows(raw_rows)
    # Invert runtime H512 -> suh -> H128 on the input axis.
    slot0 = block_hadamard(slot0, block_size=HADAMARD_RESIDUAL, dim=1)
    slot1 = block_hadamard(slot1, block_size=HADAMARD_RESIDUAL, dim=1)
    slot0 = block_hadamard(slot0 / source_suh[None], block_size=HADAMARD_INPUT, dim=1)
    slot1 = block_hadamard(slot1 / source_suh[None], block_size=HADAMARD_INPUT, dim=1)

    # Invert post-sign/H128/down-suh/H128 on K and H128/down-svh/H512 on N.
    down_core = block_hadamard(down.float(), block_size=HADAMARD_RESIDUAL, dim=0)
    down_core = block_hadamard(
        down_core / down_svh[:, None], block_size=HADAMARD_INPUT, dim=0
    )
    down_core = down_core * post_signs[None]
    down_core = block_hadamard(down_core, block_size=HADAMARD_INPUT, dim=1)
    down_core = block_hadamard(
        down_core / down_suh[None], block_size=HADAMARD_INPUT, dim=1
    )
    return slot0.contiguous(), slot1.contiguous(), down_core.contiguous()


def coupled_input_carrier(
    hidden: torch.Tensor, gate_up_suh: torch.Tensor, *, quantize: bool
) -> torch.Tensor:
    """Coupled input carrier with the archived quantized/nonquantized cast fork."""
    if hidden.ndim != 2 or hidden.shape[1] != gate_up_suh.numel():
        raise ValueError("hidden rows do not match gate_up_suh")
    work = hidden.to(torch.bfloat16).to(torch.float16).float()
    work = block_hadamard(work, block_size=HADAMARD_RESIDUAL)
    work = work * gate_up_suh.to(work.device).float()
    # Luke's quantized-activation branch retains H512*suh through H128 in
    # FP32 before E4M3.  Only its explicit nonquantized closure branch stores
    # this boundary in FP16.
    if not quantize:
        work = work.to(torch.float16).float()
    work = block_hadamard(work, block_size=HADAMARD_INPUT)
    return _qdq_e4m3_k32(work, 1.0, "amax").float() if quantize else work


def coupled_middle_carrier(
    fc1_input: torch.Tensor,
    gate_core: torch.Tensor,
    up_core: torch.Tensor,
    scales: CoupledScaleSet,
    *,
    expert: int,
    intermediate_draw: int,
    quantize: bool,
) -> torch.Tensor:
    """Exact paired FC1 epilogue through the transformed down A8 carrier."""
    physical_gate = F.linear(fc1_input.float(), gate_core.float()).to(torch.float16).float()
    physical_up = F.linear(fc1_input.float(), up_core.float()).to(torch.float16).float()
    raw = _atom_interleave(physical_gate, physical_up)
    pair_scales = _atom_interleave(
        scales.gate_svh[expert : expert + 1], scales.up_svh[expert : expert + 1]
    )[0].to(raw.device).float()
    pre = block_hadamard(raw, block_size=HADAMARD_INPUT)
    pre = block_hadamard(pre * pair_scales, block_size=HADAMARD_INPUT)
    pre_signs = rotation_signs(pre.shape[1], draw=intermediate_draw, axis=1).to(raw.device)
    pre = pre * pre_signs
    gate = pre[:, 0::2].clamp(max=ACTIVATION_LIMIT)
    up = pre[:, 1::2].clamp(-ACTIVATION_LIMIT, ACTIVATION_LIMIT)
    middle = F.silu(gate) * up
    post_signs = rotation_signs(middle.shape[1], draw=intermediate_draw, axis=2).to(raw.device)
    middle = block_hadamard(middle * post_signs, block_size=HADAMARD_INPUT)
    middle = middle * scales.down_suh[expert].to(raw.device).float()
    # Match the same archived fork on the down carrier: the native E4M3 path
    # keeps activated*down_suh through H128 in FP32.
    if not quantize:
        middle = middle.to(torch.float16).float()
    middle = block_hadamard(middle, block_size=HADAMARD_INPUT)
    return _qdq_e4m3_k32(middle, 1.0, "amax").float() if quantize else middle


def coupled_expert_reference(
    hidden: torch.Tensor,
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    scales: CoupledScaleSet,
    *,
    expert: int,
    intermediate_draw: int,
    quantize_activations: bool,
) -> torch.Tensor:
    """Execute one physical coupled expert and return caller-basis output."""
    gate_core, up_core, down_core = weights
    fc1_input = coupled_input_carrier(
        hidden, scales.gate_up_suh, quantize=quantize_activations
    )
    down_input = coupled_middle_carrier(
        fc1_input,
        gate_core,
        up_core,
        scales,
        expert=expert,
        intermediate_draw=intermediate_draw,
        quantize=quantize_activations,
    )
    down = F.linear(down_input.float(), down_core.float()).to(torch.float16).float()
    route = block_hadamard(down, block_size=HADAMARD_INPUT)
    route = route * scales.down_svh.to(route.device).float()
    return block_hadamard(route, block_size=HADAMARD_RESIDUAL)


def source_expert_reference(
    hidden: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor
) -> torch.Tensor:
    source = hidden.to(torch.bfloat16).to(torch.float16).float()
    gate_out = F.linear(source, gate.float()).clamp(max=ACTIVATION_LIMIT)
    up_out = F.linear(source, up.float()).clamp(-ACTIVATION_LIMIT, ACTIVATION_LIMIT)
    return F.linear(F.silu(gate_out) * up_out, down.float())
