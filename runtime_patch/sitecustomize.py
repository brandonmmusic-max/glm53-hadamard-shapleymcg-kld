"""Inject the sealed GLM-5.3 routed-input block rotation into vLLM.

Enable only with ``GLM53_ROUTED_ROTATION=had16`` or ``learned``.  The patched
vLLM MoE factory already has a routed-only input-transform contract: routed
experts receive the transformed tensor while the router and shared expert keep
the original tensor.
"""
from __future__ import annotations

import os
import re


MODE = os.environ.get("GLM53_ROUTED_ROTATION", "").strip().lower()
MIXED_MXFP6 = os.environ.get("GLM53_MIXED_MXFP6", "").strip().lower()


if MIXED_MXFP6:
    if MIXED_MXFP6 not in {"1", "true", "yes", "on"}:
        raise RuntimeError(f"invalid GLM53_MIXED_MXFP6={MIXED_MXFP6!r}")

    # The pinned image already contains both implementations needed by the
    # exact 6-bpw ladder: ModelOpt's mixed NVFP4 dispatcher and B12X's native
    # MXFP6/W6A8 method.  Upstream ModelOpt does not name MXFP6 in its mixed
    # switch, so bridge only that one declared per-layer algorithm and leave
    # every existing FP8/NVFP4/MXFP8 branch untouched.
    os.environ.setdefault("B12X_ENABLE_FP6", "1")
    os.environ.setdefault("B12X_FP6_MODEL_DIR", "/model")

    from b12x.integration.vllm import plugin as _fp6_plugin
    from vllm.model_executor.layers.quantization import modelopt as _modelopt

    _fp6_plugin.register_b12x_fp6()
    if _fp6_plugin._CONFIG_CLS is None:
        raise RuntimeError("B12X MXFP6 quantization plugin did not register")
    _B12X_FP6_CONFIG = _fp6_plugin._CONFIG_CLS(
        os.environ["B12X_FP6_MODEL_DIR"]
    )
    _ORIGINAL_MIXED_GET_QUANT_METHOD = (
        _modelopt.ModelOptMixedPrecisionConfig.get_quant_method
    )

    def _mixed_get_quant_method(self, layer, prefix):
        if self._resolve_quant_algo(prefix) == "MXFP6":
            method = _B12X_FP6_CONFIG.get_quant_method(layer, prefix)
            if method is None:
                raise RuntimeError(
                    f"MXFP6 layer {prefix!r} did not bind to the B12X method"
                )
            # vLLM's virtual-TP MoE loader needs the logical element geometry
            # for grouped w2 scales.  Without these existing loader attrs it
            # treats the number of scale groups as the logical K extent and
            # requests twice the stored groups in the DCP4/EP4 regime.
            original_create_weights = method.create_weights

            def _create_weights_with_scale_geometry(*args, **kwargs):
                original_create_weights(*args, **kwargs)
                target = kwargs.get("layer")
                if target is None:
                    if not args:
                        raise RuntimeError("MXFP6 create_weights did not receive a layer")
                    target = args[0]
                scale = target.w2_weight_scale
                scale.b12x_mxfp4_w2_scale_group_size = 32
                scale.b12x_mxfp4_w2_logical_k = int(scale.shape[-1]) * 32

            method.create_weights = _create_weights_with_scale_geometry
            return method
        return _ORIGINAL_MIXED_GET_QUANT_METHOD(self, layer, prefix)

    _modelopt.ModelOptMixedPrecisionConfig.get_quant_method = (
        _mixed_get_quant_method
    )
    # Preserve exact tensor geometry on any remaining mixed-loader failure.
    # This wrapper is observational on success and re-raises the original
    # exception unchanged.
    from vllm.model_executor.layers.fused_moe.routed_experts import (
        RoutedExperts as _RoutedExperts,
    )

    _ORIGINAL_ROUTED_WEIGHT_LOADER = _RoutedExperts.weight_loader

    def _diagnostic_routed_weight_loader(
        self, param, loaded_weight, weight_name, shard_id, expert_id,
        return_success=False,
    ):
        try:
            return _ORIGINAL_ROUTED_WEIGHT_LOADER(
                self, param, loaded_weight, weight_name, shard_id, expert_id,
                return_success,
            )
        except Exception:
            print(
                "GLM53_MXFP6_LOAD_DIAGNOSTIC "
                f"name={weight_name} shard={shard_id} expert={expert_id} "
                f"param_shape={tuple(param.shape)} "
                f"loaded_shape={tuple(loaded_weight.shape)} "
                f"tp_size={self.moe_config.moe_parallel_config.tp_size} "
                f"tp_rank={self.moe_config.tp_rank}",
                flush=True,
            )
            raise

    _RoutedExperts.weight_loader = _diagnostic_routed_weight_loader
    print("GLM53_MIXED_MXFP6_PATCH_ACTIVE", flush=True)


if MODE:
    if MODE not in {"had16", "learned"}:
        raise RuntimeError(f"unsupported GLM53_ROUTED_ROTATION={MODE!r}")

    import torch
    from torch import nn
    from safetensors.torch import load_file
    import vllm.models.glm5next.nvidia.model as glm_model

    _LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
    _LAYER_SPEC = os.environ.get("GLM53_ROTATED_LAYERS", "3-44").strip()


    def _parse_layers(spec: str) -> frozenset[int]:
        result: set[int] = set()
        for item in spec.split(","):
            item = item.strip()
            if not item:
                continue
            if "-" in item:
                first_text, last_text = item.split("-", 1)
                first, last = int(first_text), int(last_text)
                if first > last:
                    raise RuntimeError(f"invalid GLM53_ROTATED_LAYERS range {item!r}")
                result.update(range(first, last + 1))
            else:
                result.add(int(item))
        if not result or not result.issubset(set(range(3, 45))):
            raise RuntimeError(f"invalid GLM53_ROTATED_LAYERS={spec!r}")
        return frozenset(result)


    _ROTATED_LAYERS = _parse_layers(_LAYER_SPEC)
    _ORIGINAL_FACTORY = glm_model.FusedMoEFactory
    _LEARNED = None
    if MODE == "learned":
        rotation_path = os.environ.get("GLM53_ROTATION_FILE")
        if not rotation_path:
            raise RuntimeError("learned rotation requires GLM53_ROTATION_FILE")
        _LEARNED = load_file(rotation_path, device="cpu")


    def _hadamard16() -> torch.Tensor:
        h = torch.ones((1, 1), dtype=torch.float32)
        while h.shape[0] < 16:
            h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
        return h / 4.0


    class _Block16RoutedInput(nn.Module):
        def __init__(self, layer: int):
            super().__init__()
            if MODE == "had16":
                rotation = _hadamard16()
            else:
                assert _LEARNED is not None
                key = f"layer_{layer:03d}"
                if key not in _LEARNED:
                    raise RuntimeError(f"{key} absent from learned rotation bundle")
                rotation = _LEARNED[key].float()
            if rotation.ndim not in (2, 3) or rotation.shape[-2:] != (16, 16):
                raise RuntimeError(f"invalid layer {layer} rotation shape {tuple(rotation.shape)}")
            gram = rotation.transpose(-1, -2) @ rotation
            # vLLM constructs each worker under a rank-local default CUDA
            # device.  A learned safetensors matrix may therefore already be
            # on that device; keep the orthogonality reference colocated.
            eye = torch.eye(16, dtype=rotation.dtype, device=rotation.device)
            if float((gram - eye).abs().max()) > 2e-4:
                raise RuntimeError(f"layer {layer} rotation is not orthogonal")
            # vLLM constructs the model in its target-device context.  Keeping
            # this persistent makes it part of module/device movement but not
            # part of checkpoint loading.
            self.register_buffer("rotation", rotation, persistent=False)
            self.layer = layer

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            blocks = hidden_states.shape[-1] // 16
            if blocks * 16 != hidden_states.shape[-1]:
                raise RuntimeError("routed hidden width is not divisible by 16")
            rotation = self.rotation
            if rotation.ndim == 3 and rotation.shape[0] != blocks:
                raise RuntimeError(
                    f"layer {self.layer} has {rotation.shape[0]} rotations for {blocks} blocks"
                )
            if rotation.device != hidden_states.device or rotation.dtype != hidden_states.dtype:
                rotation = rotation.to(device=hidden_states.device, dtype=hidden_states.dtype)
                self.rotation = rotation
            view = hidden_states.reshape(*hidden_states.shape[:-1], blocks, 16)
            if rotation.ndim == 2:
                output = view @ rotation
            else:
                output = torch.einsum("...bg,bgh->...bh", view, rotation)
            return output.reshape_as(hidden_states)


    def _rotation_factory(*args, **kwargs):
        prefix = kwargs.get("prefix", "")
        match = _LAYER_RE.search(prefix)
        if match is not None:
            layer = int(match.group(1))
            # Base routed-expert layers only.  The MTP layer is outside this
            # experiment and must retain the carrier's unrotated ABI.
            if layer in _ROTATED_LAYERS:
                if kwargs.get("routed_input_transform") is not None:
                    raise RuntimeError(f"layer {layer} already has a routed input transform")
                kwargs["routed_input_transform"] = _Block16RoutedInput(layer)
        return _ORIGINAL_FACTORY(*args, **kwargs)


    glm_model.FusedMoEFactory = _rotation_factory
    print(
        f"GLM53_BLOCK_ROTATION_PATCH_ACTIVE mode={MODE} layers={_LAYER_SPEC}",
        flush=True,
    )
