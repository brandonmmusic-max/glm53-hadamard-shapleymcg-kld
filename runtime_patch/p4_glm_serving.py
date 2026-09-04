"""Fail-closed P4 installation into GLM-5.3's real modular MoE dispatch.

Import is inert. install() is called by sitecustomize only when explicitly
enabled. The original factory constructs routing/shared expert ownership;
only the selected RoutedExperts method is specialized to the B12X P4 backend.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re

import torch


LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
CODEC_SCHEMA = "glm53-p4-mcg-tp-rank.v2"
MANIFEST_SCHEMA = "glm53-p4-mcg-tp4-manifest.v1"
_METHOD_CLASSES = {}


def _sha256(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate P4 manifest key: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class P4ServingConfig:
    sidecar_dir: Path
    layers: frozenset[int]
    design_sha256: str
    manifest_sha256: str
    entries: dict
    build_dir: Path

    @classmethod
    def from_env(cls, env):
        if env.get("GLM53_P4_NATIVE", "").strip().lower() not in {"1", "true", "yes", "on"}:
            raise ValueError("GLM53_P4_NATIVE must be explicitly enabled")
        for incompatible in ("GLM53_P8_NATIVE", "GLM53_P8_PSEUDOQUANT", "GLM53_MIXED_MXFP6",
                             "GLM53_ROUTED_ROTATION", "GLM53_HUMMING_FP4_BUFFER_PATCH"):
            if env.get(incompatible, "").strip():
                raise ValueError(f"P4 identity backend is incompatible with {incompatible}")
        names = ("SIDECAR_DIR", "DESIGN", "MANIFEST", "MANIFEST_SHA256")
        values = {name: env.get("GLM53_P4_NATIVE_" + name, "").strip() for name in names}
        if not all(values.values()):
            raise ValueError("P4 requires SIDECAR_DIR, DESIGN, MANIFEST and MANIFEST_SHA256")
        sidecar_dir = Path(values["SIDECAR_DIR"]).resolve(strict=True)
        if not sidecar_dir.is_dir():
            raise ValueError("P4 sidecar root must be a directory")
        specification = env.get("GLM53_P4_NATIVE_LAYERS", "3").strip()
        if not re.fullmatch(r"\d+(?:,\d+)*", specification):
            raise ValueError("P4 layers must be a comma-separated list")
        parsed = [int(value) for value in specification.split(",")]
        layers = frozenset(parsed)
        if len(layers) != len(parsed) or not layers.issubset(range(3, 45)):
            raise ValueError("P4 layers must be unique GLM routed layers 3..44")
        design_sha256 = hashlib.sha256(Path(values["DESIGN"]).read_bytes()).hexdigest()
        raw = Path(values["MANIFEST"]).read_bytes()
        manifest_sha256 = hashlib.sha256(raw).hexdigest()
        if manifest_sha256 != _sha256(values["MANIFEST_SHA256"], "manifest pin"):
            raise ValueError("P4 manifest identity mismatch")
        manifest = json.loads(raw, object_pairs_hook=_unique_object)
        if (manifest.get("schema") != MANIFEST_SCHEMA
                or manifest.get("codec_schema") != CODEC_SCHEMA
                or type(manifest.get("world_size")) is not int or manifest["world_size"] != 4
                or manifest.get("source_design_sha256") != design_sha256):
            raise ValueError("P4 manifest schema/topology/design mismatch")
        manifest_layers = manifest.get("layers")
        if (not isinstance(manifest_layers, list)
                or any(type(value) is not int or not 3 <= value <= 44 for value in manifest_layers)
                or len(set(manifest_layers)) != len(manifest_layers)
                or not layers.issubset(manifest_layers)):
            raise ValueError("P4 manifest layer inventory mismatch")
        entries = {}
        for entry in manifest.get("entries", []):
            layer, rank = entry.get("layer"), entry.get("rank")
            if (type(layer) is not int or layer not in manifest_layers or type(rank) is not int
                    or not 0 <= rank < 4 or (layer, rank) in entries):
                raise ValueError("P4 manifest duplicate/invalid layer-rank entry")
            expected_name = f"p4-layer-{layer:03d}-tp4-rank-{rank}.safetensors"
            if entry.get("path") != expected_name:
                raise ValueError("P4 manifest sidecar filename mismatch")
            _sha256(entry.get("sha256"), "sidecar hash")
            if type(entry.get("bytes")) is not int or entry["bytes"] <= 0:
                raise ValueError("P4 manifest requires positive exact file bytes")
            # Reject symlinks escaping the mounted sidecar directory, and missing
            # files, before model construction allocates any selected layer.
            path = sidecar_dir / expected_name
            if path.resolve(strict=True).parent != sidecar_dir or not path.is_file():
                raise ValueError("P4 sidecar path escapes its root")
            if path.stat().st_size != entry["bytes"]:
                raise ValueError("P4 sidecar length mismatch")
            entries[(layer, rank)] = dict(entry)
        expected = {(layer, rank) for layer in manifest_layers for rank in range(4)}
        if set(entries) != expected:
            raise ValueError("P4 manifest requires every TP4 rank for every listed layer")
        return cls(sidecar_dir, layers, design_sha256, manifest_sha256, entries,
                   Path(env.get("GLM53_P4_NATIVE_BUILD_DIR", "/tmp/glm53-p4-native")))

    def sidecar(self, layer, rank):
        entry = self.entries[(layer, rank)]
        return self.sidecar_dir / entry["path"], entry


def validate_layer(runner, factory_kwargs):
    layer = runner.routed_experts
    config = layer.moe_config
    parallel = config.moe_parallel_config
    if (parallel.tp_size != 4 or not 0 <= config.tp_rank < 4
            or parallel.ep_size != 1 or parallel.use_ep or parallel.enable_eplb
            or parallel.dp_size != 1 or parallel.pcp_size != 1
            or parallel.sp_size != 1):
        raise ValueError("P4 requires TP4 EP1 DP1 PCP1 SP1 without EPLB")
    if (layer.global_num_experts, layer.local_num_experts, layer.hidden_size,
            layer.intermediate_size_per_partition, layer.top_k) != (288, 288, 4096, 512, 8):
        raise ValueError("P4 GLM expert/top-k/shard geometry mismatch")
    activation = getattr(layer.activation, "value", layer.activation)
    if (activation != "silu" or layer.swiglu_limit != 10.0
            or layer.apply_router_weight_on_input
            or getattr(config, "has_bias", False)
            or getattr(config, "is_lora_enabled", False)
            or getattr(layer, "expert_map", None) is not None
            or factory_kwargs.get("fuse_shared_experts", False)
            or getattr(runner, "enable_dbo", False)):
        raise ValueError("P4 requires unfused routed SwiGLU10, output route weights, no bias/LoRA/EPLB/DBO")
    if (getattr(runner, "routed_input_transform", None) is not None
            or getattr(runner, "routed_output_transform", None) is not None):
        raise ValueError("P4 sidecar requires identity routed boundaries")
    if config.in_dtype != torch.bfloat16:
        raise ValueError("P4 GLM boundary activation must be BF16")


def _release_carriers(layer):
    released = 0
    for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale",
                 "w13_weight_scale_2", "w2_weight_scale_2", "w13_input_scale", "w2_input_scale"):
        value = getattr(layer, name, None)
        if isinstance(value, torch.Tensor):
            released += value.numel() * value.element_size()
            setattr(layer, name, torch.nn.Parameter(value.new_empty(0), requires_grad=False))
    return released


class _P4ServingMethod:
    """Per-selected-instance mixin; stock method classes are never modified."""

    @property
    def is_monolithic(self):
        return False  # The unchanged GLM router supplies grouped top-k decisions.

    @property
    def topk_indices_dtype(self):
        return torch.int32

    @property
    def mk_can_overlap_shared_experts(self):
        return False  # Runner retains shared expert stream scheduling.

    @property
    def supports_internal_mk(self):
        return False

    def get_fused_moe_quant_config(self, layer):
        # The real MoERunner invokes _ensure_moe_quant_config_init before
        # forward_modular. P4 owns its operand quantization; inheriting the
        # ModelOpt builder would dereference the released carrier scales.
        # None is the public optional-config contract for custom methods.
        return None

    def process_weights_after_loading(self, layer):
        if getattr(layer, "_glm53_p4_runtime", None) is not None:
            raise RuntimeError("P4 selected layer weights processed twice")
        config = self._glm53_p4_config
        layer_id, rank = self._glm53_p4_identity
        sidecar, entry = config.sidecar(layer_id, rank)
        # P4TrellisMoEBackend validates physical CPU payload, then allocates only
        # stream/scales/globals, compiles and loads CUDA functions before graphs.
        runtime = self._glm53_p4_backend(sidecar, device=layer.w13_weight.device,
                    tp_rank=rank, layer=layer_id, expected_design_sha256=config.design_sha256,
                    expected_file_sha256=entry["sha256"], expected_file_bytes=entry["bytes"],
                    topk=layer.top_k, swiglu_limit=layer.swiglu_limit, build_dir=config.build_dir)
        layer._glm53_p4_runtime = runtime
        self.moe_kernel = None
        released = _release_carriers(layer)
        print("GLM53_P4_NATIVE_WEIGHTS_READY "
              f"layer={layer_id} rank={rank} sidecar_sha256={entry['sha256']} "
              f"design_sha256={config.design_sha256} manifest_sha256={config.manifest_sha256} "
              f"released_carrier_bytes={released} backend={runtime.backend_name} "
              "schema=glm53-p4-mcg-tp-rank.v2 law=procedural-mcg-alpha1-rne-e2m1 "
              "rounding=nearest-even-satfinite signed_zero=preserve E4M3_K16 ldlq=false", flush=True)

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts=None, shared_experts_input=None):
        runtime = getattr(layer, "_glm53_p4_runtime", None)
        if runtime is None:
            raise RuntimeError("P4 selected layer reached forward before physical sidecar loading")
        output = runtime.apply(x, topk_weights, topk_ids)
        if not getattr(layer, "_glm53_p4_forward_logged", False):
            layer_id, rank = self._glm53_p4_identity
            print("GLM53_P4_NATIVE_FORWARD "
                  f"layer={layer_id} rank={rank} backend={runtime.backend_name} "
                  "FC1=trellis_E2M1_K64 SwiGLU=clamp10 FC2=trellis_E2M1_K64 "
                  "mma=mxf4nvf4 scale=E4M3_K16 routing=stock_glm topk_sum=fixed_order "
                  "tp_reduction=runner physical_payload_bpw=4.5 ldlq=false", flush=True)
            layer._glm53_p4_forward_logged = True
        return output

    def apply_monolithic(self, *args, **kwargs):
        raise RuntimeError("P4 requires the GLM modular routing path")


def install(env=None, *, components=None):
    env = os.environ if env is None else env
    config = P4ServingConfig.from_env(env)
    if components is None:
        import vllm.models.glm5next.nvidia.model as model
        from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod
        from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4FusedMoE
        try:
            from b12x_h16.b12x.moe._shared.kernels.p4_native import P4TrellisMoEBackend
        except ModuleNotFoundError as error:
            if error.name != "b12x_h16":
                raise
            from runtime_patch.b12x_h16.b12x.moe._shared.kernels.p4_native import P4TrellisMoEBackend
        method_types = (UnquantizedFusedMoEMethod, ModelOptNvFp4FusedMoE)
        backend = P4TrellisMoEBackend
    else:
        # Exact host integration seam. The tests execute these same factory
        # and method bodies with inert model/backend objects; no CUDA access.
        model, method_types, backend = components
    original = model.FusedMoEFactory
    if getattr(original, "_glm53_p4_installed", False):
        raise RuntimeError("P4 factory already installed")

    def factory(*args, **kwargs):
        match = LAYER_RE.search(kwargs.get("prefix", ""))
        runner = original(*args, **kwargs)
        if match is None or int(match.group(1)) not in config.layers:
            return runner
        validate_layer(runner, kwargs)
        layer = runner.routed_experts
        method = layer.quant_method
        if type(method) not in method_types:
            raise RuntimeError("P4 requires a BF16 or ModelOpt NVFP4 carrier method")
        base = type(method)
        if base not in _METHOD_CLASSES:
            _METHOD_CLASSES[base] = type("GLM53P4" + base.__name__, (_P4ServingMethod, base), {})
        # Retain original initialization/create_weights/loaders and all carrier
        # tensors until the ordinary post-load hook consumes the pinned sidecar.
        method.__class__ = _METHOD_CLASSES[base]
        method._glm53_p4_config = config
        method._glm53_p4_identity = (int(match.group(1)), int(layer.moe_config.tp_rank))
        method._glm53_p4_backend = backend
        method.moe_kernel = None
        method.moe_quant_config = None
        return runner

    factory._glm53_p4_installed = True
    model.FusedMoEFactory = factory
    print("GLM53_P4_NATIVE_PATCH_ACTIVE "
          f"layers={','.join(map(str, sorted(config.layers)))} tp=4 "
          f"design_sha256={config.design_sha256} manifest_sha256={config.manifest_sha256} "
          "schema=glm53-p4-mcg-tp-rank.v2 backend=b12x-p4-trellis-mxf4nvf4 "
          "law=procedural-mcg-alpha1-rne-e2m1 mma=mxf4nvf4 E2M1 E4M3_K16 "
          "identity physical_bpw=4.5 ldlq=false", flush=True)
    return config
