"""Inject the sealed GLM-5.3 routed-input block rotation into vLLM.

Enable only with ``GLM53_ROUTED_ROTATION=had16`` or ``learned``.  The patched
vLLM MoE factory already has a routed-only input-transform contract: routed
experts receive the transformed tensor while the router and shared expert keep
the original tensor.
"""
from __future__ import annotations

import os
import re
import hashlib


MODE = os.environ.get("GLM53_ROUTED_ROTATION", "").strip().lower()
MIXED_MXFP6 = os.environ.get("GLM53_MIXED_MXFP6", "").strip().lower()
HUMMING_FP4_BUFFER_PATCH = os.environ.get(
    "GLM53_HUMMING_FP4_BUFFER_PATCH", ""
).strip().lower()


if HUMMING_FP4_BUFFER_PATCH:
    if HUMMING_FP4_BUFFER_PATCH not in {"1", "true", "yes", "on"}:
        raise RuntimeError(
            "invalid GLM53_HUMMING_FP4_BUFFER_PATCH="
            f"{HUMMING_FP4_BUFFER_PATCH!r}"
        )
    from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
        HummingExpertsBase as _HummingFp4ExpertsBase,
    )
    from vllm.utils.humming import dtypes as _humming_dtypes

    _ORIGINAL_HUMMING_GET_BUFFER_METAS = _HummingFp4ExpertsBase.get_buffer_metas
    _ORIGINAL_HUMMING_INIT_MOE = _HummingFp4ExpertsBase.init_humming_moe
    _ORIGINAL_HUMMING_FP4_QUANTIZE_INPUT = (
        _HummingFp4ExpertsBase.quantize_input
    )

    def _init_humming_moe_with_m_major_input_scale(self):
        _ORIGINAL_HUMMING_INIT_MOE(self)
        if any(
            config.a_dtype == _humming_dtypes.float4e2m1
            and config.input_scale_group_size > 0
            for config in self.humming_configs.values()
        ):
            self.compute_config["use_m_major_input_scale"] = True
            import json as _json

            self.compute_config_str = _json.dumps(self.compute_config)

    _HummingFp4ExpertsBase.init_humming_moe = (
        _init_humming_moe_with_m_major_input_scale
    )

    def _quantize_float4_input_with_aligned_output(
        self, sublayer_name, inputs, quanted_input
    ):
        config = self.humming_configs[sublayer_name]
        if config.a_dtype == _humming_dtypes.float4e2m1:
            # The shared workspace view is not guaranteed to meet the FP4
            # grouped kernel's vector alignment.  Let Humming allocate its
            # packed output directly from the CUDA allocator.
            quanted_input = None
        return _ORIGINAL_HUMMING_FP4_QUANTIZE_INPUT(
            self, sublayer_name, inputs, quanted_input
        )

    _HummingFp4ExpertsBase.quantize_input = (
        _quantize_float4_input_with_aligned_output
    )

    def _get_buffer_metas_with_float4_storage(self, M, topk, activation):
        # Humming packs both integer4 and float4 activations two per uint8.
        # The pinned allocator maps int4 -> uint8 but omits float4e2m1.
        # Substitute only while it computes storage shapes/dtypes; restore the
        # real FP4 LayerConfig before quantization or GEMM dispatch.
        changed = []
        for config in self.humming_configs.values():
            if config.a_dtype == _humming_dtypes.float4e2m1:
                changed.append(config)
                object.__setattr__(config, "a_dtype", _humming_dtypes.int4)
        try:
            return _ORIGINAL_HUMMING_GET_BUFFER_METAS(self, M, topk, activation)
        finally:
            for config in changed:
                object.__setattr__(
                    config, "a_dtype", _humming_dtypes.float4e2m1
                )

    _HummingFp4ExpertsBase.get_buffer_metas = (
        _get_buffer_metas_with_float4_storage
    )
    print("GLM53_HUMMING_FP4_BUFFER_PATCH_ACTIVE", flush=True)


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
    _W13_SCALE_NORMALIZATION_LOGGED = False

    def _diagnostic_routed_weight_loader(
        self, param, loaded_weight, weight_name, shard_id, expert_id,
        return_success=False,
    ):
        global _W13_SCALE_NORMALIZATION_LOGGED
        if (
            self.quant_method.__class__.__name__ == "_VllmMoEMethod"
            and weight_name.endswith("w13_weight_scale")
            and loaded_weight.ndim >= 2
            and loaded_weight.shape[-1] == 2 * param.shape[-1]
        ):
            # The virtual-TP preload pads this group-32 axis to the carrier's
            # group-16 width.  The second half is padding, not FP6 scales.
            loaded_weight = loaded_weight.narrow(
                loaded_weight.ndim - 1, 0, param.shape[-1]
            ).contiguous()
            if not _W13_SCALE_NORMALIZATION_LOGGED:
                print(
                    "GLM53_MXFP6_W13_SCALE_GEOMETRY_ACTIVE "
                    f"groups={param.shape[-1]}",
                    flush=True,
                )
                _W13_SCALE_NORMALIZATION_LOGGED = True
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
    from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
        HummingExpertsBase as _HummingExpertsBase,
    )
    from vllm.model_executor.layers.quantization import modelopt as _rotation_modelopt

    _LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
    _LAYER_SPEC = os.environ.get("GLM53_ROTATED_LAYERS", "3-44").strip()
    _ROTATION_SCOPE = os.environ.get(
        "GLM53_ROTATION_SCOPE", "gate-up"
    ).strip().lower()
    _ROTATION_PLACEMENT = os.environ.get(
        "GLM53_ROTATION_PLACEMENT", "runner"
    ).strip().lower()
    _RUNTIME_NEGATE_W13 = os.environ.get(
        "GLM53_RUNTIME_NEGATE_W13", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    if _ROTATION_SCOPE not in {"gate-up", "mid-only", "all"}:
        raise RuntimeError(f"invalid GLM53_ROTATION_SCOPE={_ROTATION_SCOPE!r}")
    if _ROTATION_PLACEMENT not in {"runner", "humming-inner"}:
        raise RuntimeError(
            f"invalid GLM53_ROTATION_PLACEMENT={_ROTATION_PLACEMENT!r}"
        )


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


    def _rotation_for(layer: int, kind: str) -> torch.Tensor:
        if MODE == "had16":
            return _hadamard16()
        assert _LEARNED is not None
        preferred = f"layer_{layer:03d}_{kind}"
        legacy = f"layer_{layer:03d}"
        key = preferred if preferred in _LEARNED else legacy
        if key not in _LEARNED:
            raise RuntimeError(f"{preferred} absent from learned rotation bundle")
        rotation = _LEARNED[key].float()
        blocks = 256 if kind == "in" else 128
        if rotation.ndim == 3 and rotation.shape[0] != blocks:
            raise RuntimeError(
                f"layer {layer} {kind} rotation has {rotation.shape[0]} blocks, "
                f"expected {blocks}"
            )
        return rotation


    def _apply_block16(hidden_states: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
        blocks = hidden_states.shape[-1] // 16
        if blocks * 16 != hidden_states.shape[-1]:
            raise RuntimeError("rotated width is not divisible by 16")
        if rotation.ndim == 3 and rotation.shape[0] != blocks:
            raise RuntimeError(
                f"rotation has {rotation.shape[0]} blocks for runtime width "
                f"{hidden_states.shape[-1]} ({blocks} blocks)"
            )
        if rotation.device != hidden_states.device or rotation.dtype != hidden_states.dtype:
            rotation = rotation.to(
                device=hidden_states.device, dtype=hidden_states.dtype
            )
        view = hidden_states.reshape(*hidden_states.shape[:-1], blocks, 16)
        if rotation.ndim == 2:
            output = view @ rotation
        else:
            output = torch.einsum("...bg,bgh->...bh", view, rotation)
        return output.reshape_as(hidden_states)


    class _Block16RoutedInput(nn.Module):
        def __init__(self, layer: int):
            super().__init__()
            rotation = _rotation_for(layer, "in")
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
            self._rotation_sha256 = hashlib.sha256(
                rotation.detach().float().cpu().numpy().tobytes()
            ).hexdigest()
            self._forward_logged = False

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            rotation = self.rotation
            if rotation.device != hidden_states.device or rotation.dtype != hidden_states.dtype:
                rotation = rotation.to(device=hidden_states.device, dtype=hidden_states.dtype)
                self.rotation = rotation
            output = _apply_block16(hidden_states, rotation)
            if not self._forward_logged:
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    rank = str(torch.distributed.get_rank())
                else:
                    rank = os.environ.get("LOCAL_RANK", "unknown")
                print(
                    "GLM53_BLOCK_ROTATION_FORWARD "
                    f"layer={self.layer} rank={rank} "
                    f"rotation_sha256={self._rotation_sha256} "
                    f"hidden_width={hidden_states.shape[-1]} dtype={hidden_states.dtype}",
                    flush=True,
                )
                self._forward_logged = True
            return output


    _ORIGINAL_HUMMING_APPLY_ACTIVATION = _HummingExpertsBase.apply_activation
    _ORIGINAL_HUMMING_QUANTIZE_INPUT = _HummingExpertsBase.quantize_input


    def _humming_quantize_input_with_in_rotation(
        self, sublayer_name, inputs, quanted_input
    ):
        rotation = getattr(self, "_glm53_in_rotation", None)
        if sublayer_name == "w13" and rotation is not None:
            inputs = _apply_block16(inputs, rotation)
            if not getattr(self, "_glm53_in_rotation_logged", False):
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    rank = str(torch.distributed.get_rank())
                else:
                    rank = os.environ.get("LOCAL_RANK", "unknown")
                print(
                    "GLM53_BLOCK_ROTATION_FORWARD "
                    f"layer={self._glm53_rotation_layer} rank={rank} "
                    f"rotation_sha256={self._glm53_in_rotation_sha256} "
                    f"hidden_width={inputs.shape[-1]} dtype={inputs.dtype} "
                    "placement=humming-inner "
                    f"humming_a_dtype={self.humming_configs['w13'].a_dtype} "
                    "humming_input_group_size="
                    f"{self.humming_configs['w13'].input_scale_group_size}",
                    flush=True,
                )
                self._glm53_in_rotation_logged = True
        return _ORIGINAL_HUMMING_QUANTIZE_INPUT(
            self, sublayer_name, inputs, quanted_input
        )


    _HummingExpertsBase.quantize_input = _humming_quantize_input_with_in_rotation


    def _humming_apply_activation_with_mid_rotation(
        self, activation, output, input
    ):
        _ORIGINAL_HUMMING_APPLY_ACTIVATION(self, activation, output, input)
        rotation = getattr(self, "_glm53_mid_rotation", None)
        if rotation is None:
            return
        if rotation.device != output.device or rotation.dtype != output.dtype:
            rotation = rotation.to(device=output.device, dtype=output.dtype)
            self._glm53_mid_rotation = rotation
        output.copy_(_apply_block16(output, rotation))
        if not getattr(self, "_glm53_mid_rotation_logged", False):
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = str(torch.distributed.get_rank())
            else:
                rank = os.environ.get("LOCAL_RANK", "unknown")
            print(
                "GLM53_BLOCK_MID_ROTATION_FORWARD "
                f"layer={self._glm53_rotation_layer} rank={rank} "
                f"rotation_sha256={self._glm53_mid_rotation_sha256} "
                f"hidden_width={output.shape[-1]} dtype={output.dtype}",
                flush=True,
            )
            self._glm53_mid_rotation_logged = True


    _HummingExpertsBase.apply_activation = _humming_apply_activation_with_mid_rotation

    _ORIGINAL_NVFP4_PROCESS = (
        _rotation_modelopt.ModelOptNvFp4FusedMoE.process_weights_after_loading
    )


    def _nvfp4_process_with_mid_rotation(self, layer):
        if _RUNTIME_NEGATE_W13 and hasattr(layer, "_glm53_in_rotation"):
            if layer.w13_weight.dtype != torch.uint8:
                raise RuntimeError(
                    "runtime w13 sign canary requires pre-conversion uint8 weights, "
                    f"got {layer.w13_weight.dtype}"
                )
            with torch.no_grad():
                # Each byte holds two E2M1 values.  Bit 3 is the low-nibble
                # sign and bit 7 the high-nibble sign.  Toggling both negates
                # every represented value; signed zero remains numerical zero.
                layer.w13_weight.bitwise_xor_(0x88)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = str(torch.distributed.get_rank())
            else:
                rank = os.environ.get("LOCAL_RANK", "unknown")
            print(
                "GLM53_RUNTIME_W13_SIGN_CANARY "
                f"layer={layer._glm53_rotation_layer} rank={rank} "
                "stage=pre-humming-conversion xor=0x88",
                flush=True,
            )
        _ORIGINAL_NVFP4_PROCESS(self, layer)
        in_rotation = getattr(layer, "_glm53_in_rotation", None)
        rotation = getattr(layer, "_glm53_mid_rotation", None)
        if in_rotation is None and rotation is None:
            return
        experts = self.moe_kernel.fused_experts
        if not isinstance(experts, _HummingExpertsBase):
            raise RuntimeError(
                "in-block rotation requires --moe-backend humming; "
                f"selected {experts.__class__.__name__}"
            )
        # HummingExpertsBase is a lightweight runtime object, not an nn.Module.
        # Keep the immutable rotation as a plain tensor attribute; the wrapper
        # below moves it to the activation device/dtype on first use.
        experts._glm53_rotation_layer = layer._glm53_rotation_layer
        if in_rotation is not None:
            experts._glm53_in_rotation = in_rotation.detach().clone()
            experts._glm53_in_rotation_sha256 = layer._glm53_in_rotation_sha256
            experts._glm53_in_rotation_logged = False
        if rotation is not None:
            experts._glm53_mid_rotation = rotation.detach().clone()
            experts._glm53_mid_rotation_sha256 = layer._glm53_mid_rotation_sha256
            experts._glm53_mid_rotation_logged = False


    _rotation_modelopt.ModelOptNvFp4FusedMoE.process_weights_after_loading = (
        _nvfp4_process_with_mid_rotation
    )


    def _rotation_factory(*args, **kwargs):
        prefix = kwargs.get("prefix", "")
        match = _LAYER_RE.search(prefix)
        if match is not None:
            layer = int(match.group(1))
            # Base routed-expert layers only.  The MTP layer is outside this
            # experiment and must retain the carrier's unrotated ABI.
            if (
                layer in _ROTATED_LAYERS
                and _ROTATION_SCOPE in {"gate-up", "all"}
                and _ROTATION_PLACEMENT == "runner"
            ):
                if kwargs.get("routed_input_transform") is not None:
                    raise RuntimeError(f"layer {layer} already has a routed input transform")
                kwargs["routed_input_transform"] = _Block16RoutedInput(layer)
        runner = _ORIGINAL_FACTORY(*args, **kwargs)
        if (
            match is not None
            and layer in _ROTATED_LAYERS
            and _ROTATION_SCOPE in {"gate-up", "all"}
            and _ROTATION_PLACEMENT == "humming-inner"
        ):
            in_rotation = _rotation_for(layer, "in")
            gram = in_rotation.transpose(-1, -2) @ in_rotation
            eye = torch.eye(16, dtype=in_rotation.dtype, device=in_rotation.device)
            if float((gram - eye).abs().max()) > 2e-4:
                raise RuntimeError(f"layer {layer} input rotation is not orthogonal")
            routed = runner.routed_experts
            routed.register_buffer("_glm53_in_rotation", in_rotation, persistent=False)
            routed._glm53_rotation_layer = layer
            routed._glm53_in_rotation_sha256 = hashlib.sha256(
                in_rotation.detach().float().cpu().numpy().tobytes()
            ).hexdigest()
        if match is not None and layer in _ROTATED_LAYERS and _ROTATION_SCOPE in {"mid-only", "all"}:
            mid_rotation = _rotation_for(layer, "mid")
            gram = mid_rotation.transpose(-1, -2) @ mid_rotation
            eye = torch.eye(
                16, dtype=mid_rotation.dtype, device=mid_rotation.device
            )
            if float((gram - eye).abs().max()) > 2e-4:
                raise RuntimeError(f"layer {layer} mid rotation is not orthogonal")
            routed = runner.routed_experts
            routed.register_buffer(
                "_glm53_mid_rotation", mid_rotation, persistent=False
            )
            routed._glm53_rotation_layer = layer
            routed._glm53_mid_rotation_sha256 = hashlib.sha256(
                mid_rotation.detach().float().cpu().numpy().tobytes()
            ).hexdigest()
        return runner


    glm_model.FusedMoEFactory = _rotation_factory
    print(
        f"GLM53_BLOCK_ROTATION_PATCH_ACTIVE mode={MODE} "
        f"layers={_LAYER_SPEC} scope={_ROTATION_SCOPE} placement={_ROTATION_PLACEMENT}",
        flush=True,
    )
