"""Inject the sealed GLM-5.3 routed-input block rotation into vLLM.

Enable only with ``GLM53_ROUTED_ROTATION=had16``, ``had32``, ``had64``, or
``learned``.  The patched
vLLM MoE factory already has a routed-only input-transform contract: routed
experts receive the transformed tensor while the router and shared expert keep
the original tensor.
"""
from __future__ import annotations

import os
import re
import hashlib
from pathlib import Path


if os.environ.get("VLLM_NVFP4_MLA_SCALE_COUNTER", ""):
    try:
        from nvfp4_mla_scale_counter import install_from_environment as _install_mla_counter

        _install_mla_counter()
    except BaseException:
        import traceback

        traceback.print_exc()
        os._exit(78)


if os.environ.get("GLM53_P8_INDEX_ORDER", ""):
    try:
        from p8_index_order import install as _install_index_order
        _install_index_order()
        if os.environ.get("GLM53_P8_INDEX_ORDER_RECEIPT", ""):
            from p8_index_order_receipt import install as _install_index_order_receipt
            _install_index_order_receipt()
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(78)


if os.environ.get("GLM53_P8_INDEX_ORDER_RECEIPT", "") and not os.environ.get("GLM53_P8_INDEX_ORDER", ""):
    os._exit(78)


if os.environ.get("GLM53_P8_INDEX_TRACE", ""):
    try:
        from p8_index_trace import install as _install_index_trace
        _install_index_trace()
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(78)


MODE = os.environ.get("GLM53_ROUTED_ROTATION", "").strip().lower()
MIXED_MXFP6 = os.environ.get("GLM53_MIXED_MXFP6", "").strip().lower()
HUMMING_FP4_BUFFER_PATCH = os.environ.get(
    "GLM53_HUMMING_FP4_BUFFER_PATCH", ""
).strip().lower()
ROUTED_EXPERTS_SPARSE_MLA_PATCH = os.environ.get(
    "GLM53_ROUTED_EXPERTS_SPARSE_MLA_PATCH", ""
).strip().lower()
P8_PSEUDOQUANT = os.environ.get("GLM53_P8_PSEUDOQUANT", "").strip().lower()
P8_NATIVE = os.environ.get("GLM53_P8_NATIVE", "").strip().lower()
P8_SMALL_M = os.environ.get("GLM53_P8_SMALL_M", "").strip().lower()
P8_FC1_TILE_N = os.environ.get("GLM53_P8_FC1_TILE_N", "128").strip()
P8_FUSED_SCRATCH = os.environ.get("GLM53_P8_FUSED_SCRATCH", "").strip().lower()
P4_NATIVE = os.environ.get("GLM53_P4_NATIVE", "").strip().lower()


if P4_NATIVE:
    try:
        from p4_glm_serving import install as _install_p4_glm_serving
        _install_p4_glm_serving()
    except BaseException:
        # Python normally reports and SWALLOWS sitecustomize exceptions. That
        # would quietly serve the carrier after a bad P4 identity or import.
        # P4 activation is explicit, so installation failure terminates startup.
        try:
            import traceback
            traceback.print_exc()
        finally:
            os._exit(78)


if P8_NATIVE and P8_PSEUDOQUANT:
    raise RuntimeError("GLM53_P8_NATIVE and GLM53_P8_PSEUDOQUANT are mutually exclusive")


if P8_NATIVE:
    if P8_NATIVE not in {"1", "true", "yes", "on"}:
        raise RuntimeError(f"invalid GLM53_P8_NATIVE={P8_NATIVE!r}")
    if P8_SMALL_M and P8_SMALL_M not in {"1", "true", "yes", "on"}:
        raise RuntimeError(f"invalid GLM53_P8_SMALL_M={P8_SMALL_M!r}")
    _P8N_SMALL_M = bool(P8_SMALL_M)
    if P8_FC1_TILE_N not in {"128", "64", "32"}:
        raise RuntimeError("GLM53_P8_FC1_TILE_N must be 128, 64 or 32")
    if P8_FUSED_SCRATCH and P8_FUSED_SCRATCH not in {"1", "true", "yes", "on"}:
        raise RuntimeError("invalid GLM53_P8_FUSED_SCRATCH")
    _P8N_FC1_TILE_N = int(P8_FC1_TILE_N)
    _P8N_FUSED_SCRATCH = bool(P8_FUSED_SCRATCH)
    _P8N_ROUTE_CAPTURE_MAX = (
        int(os.environ.get("GLM53_P8_ROUTE_CAPTURE_CALLS", "0"))
        if os.environ.get("GLM53_P8_ROUTE_CAPTURE") == "1" else 0
    )
    _P8N_ROUTE_CAPTURE_COUNTS = {}
    _P8N_TP2_REPACK = os.environ.get("GLM53_P8_NATIVE_TP2_REPACK", "") == "1"
    _P8N_COMPACT_SCALES = os.environ.get("GLM53_P8_COMPACT_SCALES", "") == "1"
    _P8N_TP2_MANIFEST = None
    if _P8N_TP2_REPACK:
        import json as _p8n_json
        _p8n_manifest_bytes = Path(os.environ["GLM53_P8_NATIVE_MANIFEST"]).read_bytes()
        if hashlib.sha256(_p8n_manifest_bytes).hexdigest() != os.environ["GLM53_P8_NATIVE_MANIFEST_SHA256"]:
            raise RuntimeError("TP2 source manifest hash mismatch")
        _P8N_TP2_MANIFEST = _p8n_json.loads(_p8n_manifest_bytes)
        if not _P8N_COMPACT_SCALES:
            raise RuntimeError("TP2 requires compact scale storage")
    if (_P8N_FC1_TILE_N != 128 or _P8N_FUSED_SCRATCH) and not _P8N_SMALL_M:
        raise RuntimeError("narrow FC1 and fused scratch require GLM53_P8_SMALL_M")
    import torch as _p8n_torch
    if _P8N_TP2_REPACK:
        from runtime_patch.p8_kda_autotune import install as _p8_install_kda_spill_guard
        _p8_install_kda_spill_guard()
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner as _P8TP2Runner
        _P8TP2_ORIGINAL_SINGLE_PROFILE = _P8TP2Runner.profile_single_request_prefill

        def _p8_tp2_single_profile(self):
            # Completed dummy passes leave unused allocator segments behind.
            # Triton/driver allocations cannot reclaim PyTorch's free cache.
            # Release only unused segments; preserve live tensors and every
            # profiling/warmup call, peak counter, and cache-size calculation.
            import gc
            gc.collect()
            before = _p8n_torch.cuda.memory_reserved()
            _p8n_torch.cuda.empty_cache()
            after = _p8n_torch.cuda.memory_reserved()
            print(f"P8_TP2_PROFILE_CACHE_RELEASE reserved_before={before} reserved_after={after}", flush=True)
            if os.environ.get("GLM53_P8_MEMORY_AUDIT", "") == "1":
                import json
                seen = set()
                def unique_bytes(tensors):
                    total = 0
                    for tensor in tensors:
                        if not isinstance(tensor, _p8n_torch.Tensor) or tensor.device.type != "cuda":
                            continue
                        storage = tensor.untyped_storage()
                        key = (tensor.device.index, storage.data_ptr())
                        if key not in seen:
                            seen.add(key)
                            total += storage.nbytes()
                    return total
                registered = unique_bytes(list(self.model.parameters()) + list(self.model.buffers()))
                native = 0
                for module in self.model.modules():
                    runtime = getattr(module, "_glm53_p8_native_runtime", None)
                    if runtime is not None:
                        native += unique_bytes(vars(runtime).values())
                free, total = _p8n_torch.cuda.mem_get_info()
                allocated = _p8n_torch.cuda.memory_allocated()
                reserved = _p8n_torch.cuda.memory_reserved()
                print("P8_TP2_MEMORY_AUDIT " + json.dumps(dict(
                    registered_storage=registered, native_storage=native,
                    other_torch_allocated=allocated-registered-native,
                    allocated=allocated, reserved=reserved, unused_allocator_cache=reserved-allocated,
                    device_used=total-free, non_torch_estimate=total-free-reserved,
                )), flush=True)
                from vllm.v1.worker.workspace import current_workspace_manager
                workspace = current_workspace_manager()
                print("P8_TP2_WORKSPACE_AUDIT " + json.dumps([
                    0 if tensor is None else tensor.untyped_storage().nbytes()
                    for tensor in workspace._current_workspaces
                ]), flush=True)
                # Read-only inventory of live storage outside model weights.
                # Do not free tensors or change the profiler's accounting.
                extras = []
                for obj in gc.get_objects():
                    if not isinstance(obj, _p8n_torch.Tensor):
                        continue
                    if obj.device.type != "cuda":
                        continue
                    storage = obj.untyped_storage()
                    key = (obj.device.index, storage.data_ptr())
                    if key in seen:
                        continue
                    seen.add(key)
                    if storage.nbytes() >= 1024 * 1024:
                        extras.append(dict(bytes=storage.nbytes(), shape=list(obj.shape), dtype=str(obj.dtype)))
                print("P8_TP2_EXTRA_STORAGE_AUDIT " + json.dumps(
                    sorted(extras, key=lambda item: item["bytes"], reverse=True)
                ), flush=True)
            return _P8TP2_ORIGINAL_SINGLE_PROFILE(self)

        _P8TP2Runner.profile_single_request_prefill = _p8_tp2_single_profile
    import vllm.models.glm5next.nvidia.model as _p8n_glm_model
    from p8_native_kernel import P8NativeTPMoE as _P8NativeTPMoE
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod as _P8NativeUnquantizedFusedMoEMethod,
    )
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase as _P8NativeFusedMoEMethodBase,
    )
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4FusedMoE as _P8NativeModelOptNvFp4FusedMoE,
    )

    _P8N_SIDECAR_TEXT = os.environ.get("GLM53_P8_NATIVE_SIDECAR_DIR", "").strip()
    if not _P8N_SIDECAR_TEXT:
        raise RuntimeError("GLM53_P8_NATIVE requires GLM53_P8_NATIVE_SIDECAR_DIR")
    _P8N_SIDECAR_DIR = Path(_P8N_SIDECAR_TEXT)
    _P8N_LAYER_SPEC = os.environ.get("GLM53_P8_NATIVE_LAYERS", "3").strip()
    _P8N_LAYERS = frozenset(
        int(value) for value in _P8N_LAYER_SPEC.split(",") if value
    )
    if not _P8N_LAYERS or not _P8N_LAYERS.issubset(set(range(3, 45))):
        raise RuntimeError(f"invalid GLM53_P8_NATIVE_LAYERS={_P8N_LAYER_SPEC!r}")
    _P8N_DESIGN_TEXT = os.environ.get("GLM53_P8_NATIVE_DESIGN", "").strip()
    if not _P8N_DESIGN_TEXT:
        raise RuntimeError("GLM53_P8_NATIVE requires GLM53_P8_NATIVE_DESIGN")
    # v10: one or more ':'-separated encoder design files. Every sidecar must
    # carry the hash of exactly one listed design; identity and coupled layers
    # encoded under different pinned designs can therefore share one process.
    _P8N_DESIGN_PATHS = [Path(value) for value in _P8N_DESIGN_TEXT.split(":") if value]
    if not _P8N_DESIGN_PATHS:
        raise RuntimeError("GLM53_P8_NATIVE_DESIGN lists no design file")
    for _p8n_design in _P8N_DESIGN_PATHS:
        if not _p8n_design.is_file():
            raise RuntimeError(f"missing P8 native design: {_p8n_design}")
    _P8N_DESIGN_SHA256S = frozenset(
        hashlib.sha256(path.read_bytes()).hexdigest() for path in _P8N_DESIGN_PATHS
    )
    if len(_P8N_DESIGN_SHA256S) != len(_P8N_DESIGN_PATHS):
        raise RuntimeError("GLM53_P8_NATIVE_DESIGN lists a design twice")
    _P8N_DESIGN_SHA256 = ",".join(sorted(_P8N_DESIGN_SHA256S))
    from safetensors import safe_open as _p8n_safe_open
    _P8N_TRANSFORM_TEXT = os.environ.get(
        "GLM53_P8_NATIVE_TRANSFORM", ""
    ).strip()
    _P8N_TRANSFORM_SHA256 = None
    if _P8N_TRANSFORM_TEXT:
        _P8N_TRANSFORM = Path(_P8N_TRANSFORM_TEXT)
        if not _P8N_TRANSFORM.is_file():
            raise RuntimeError(f"missing P8 encoder transform: {_P8N_TRANSFORM}")
        _P8N_TRANSFORM_SHA256 = hashlib.sha256(
            _P8N_TRANSFORM.read_bytes()
        ).hexdigest()
    _P8N_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
    _P8N_ORIGINAL_UNQUANTIZED_PROCESS = (
        _P8NativeUnquantizedFusedMoEMethod.process_weights_after_loading
    )
    _P8N_ORIGINAL_UNQUANTIZED_FORWARD = (
        _P8NativeUnquantizedFusedMoEMethod.forward_native
    )
    _P8N_ORIGINAL_MODELOPT_PROCESS = (
        _P8NativeModelOptNvFp4FusedMoE.process_weights_after_loading
    )
    _P8N_ORIGINAL_MODELOPT_APPLY = _P8NativeModelOptNvFp4FusedMoE.apply
    _P8N_ORIGINAL_IS_MONOLITHIC = _P8NativeFusedMoEMethodBase.is_monolithic.fget

    def _p8n_release_carrier_parameters(layer):
        released = 0
        for name in (
            "w13_weight",
            "w2_weight",
            "w13_weight_scale",
            "w2_weight_scale",
            "w13_weight_scale_2",
            "w2_weight_scale_2",
            "w13_input_scale",
            "w2_input_scale",
        ):
            value = getattr(layer, name, None)
            if not isinstance(value, _p8n_torch.Tensor):
                continue
            released += value.numel() * value.element_size()
            empty = _p8n_torch.nn.Parameter(
                _p8n_torch.empty(0, dtype=value.dtype, device=value.device),
                requires_grad=False,
            )
            setattr(layer, name, empty)
        return released

    def _p8n_attach(layer, method):
        method.moe_kernel = None
        rank = int(layer._glm53_p8_tp_rank)
        layer_id = int(layer._glm53_p8_layer)
        world_size = int(layer._glm53_p8_tp_size)
        parents = None
        if world_size == 2:
            record = next(r for r in _P8N_TP2_MANIFEST['layers'] if int(r['layer']) == layer_id)
            entries = [record['ranks'][r] for r in (2*rank, 2*rank+1)]
            sidecar = tuple(_P8N_SIDECAR_DIR / Path(e['path']).name for e in entries)
            parents = tuple(e['sha256'] for e in entries)
            _p8n_sidecar_design = record['source_design_sha256']
        else:
            sidecar = _P8N_SIDECAR_DIR / f"p8-layer-{layer_id:03d}-tp4-rank-{rank}.safetensors"
            if not sidecar.is_file():
                raise RuntimeError(f"missing P8 native sidecar: {sidecar}")
            with _p8n_safe_open(sidecar, framework="pt", device="cpu") as _p8n_src:
                _p8n_sidecar_design = (_p8n_src.metadata() or {}).get("source_design_sha256")
        if _p8n_sidecar_design not in _P8N_DESIGN_SHA256S:
            raise RuntimeError(
                f"P8 sidecar {sidecar} carries design {_p8n_sidecar_design!r} "
                "outside the GLM53_P8_NATIVE_DESIGN allowlist"
            )
        device = layer.w13_weight.device
        # TP2 is a capacity-constrained adapter: release the unused stock MoE
        # carrier before allocating its replacement, not after peak allocation.
        released = _p8n_release_carrier_parameters(layer) if world_size == 2 else 0
        layer._glm53_p8_native_runtime = _P8NativeTPMoE(
            sidecar,
            device=device,
            tp_rank=rank,
            world_size=world_size,
            tp4_parent_sha256=parents,
            layer=layer_id,
            expected_design_sha256=_p8n_sidecar_design,
            expected_transform_sha256=_P8N_TRANSFORM_SHA256,
            topk=8,
            hidden=4096,
            intermediate=2048 // world_size,
            swiglu_limit=10.0,
            small_m_scheduler=_P8N_SMALL_M,
            fc1_tile_n=_P8N_FC1_TILE_N,
            fuse_scratch_zero=_P8N_FUSED_SCRATCH,
            compact_scale_storage=_P8N_COMPACT_SCALES,
            compact_input_storage=world_size == 2,
            shared_workspace=world_size == 2,
            prefill_chunk_tokens=int(os.environ.get("GLM53_P8_PREFILL_CHUNK_TOKENS", "0")),
        )
        if world_size == 4:
            released = _p8n_release_carrier_parameters(layer)
        _p8n_scale_component = (
            layer._glm53_p8_native_runtime.scale_component is not None
        )
        _p8n_full_coupled = layer._glm53_p8_native_runtime.full_coupled
        _p8n_boundary = "identity"
        if _p8n_scale_component:
            _p8n_boundary = "h128-suh-svh-scale-component"
        if _p8n_full_coupled:
            _p8n_boundary = "coupled-h512-h128-suh-svh-v1"
        print(
            "GLM53_P8_NATIVE_WEIGHTS_READY "
            f"layer={layer_id} rank={rank} sidecar={sidecar} "
            f"design_sha256={_p8n_sidecar_design} released_carrier_bytes={released} "
            f"design_allowlist_size={len(_P8N_DESIGN_SHA256S)} "
            f"transform_sha256={_P8N_TRANSFORM_SHA256 or 'none'} "
            f"stream=K{layer._glm53_p8_native_runtime.trellis_bits} law=mcg alphabet=E4M3 scale=UE8M0_K32 "
            f"boundary={_p8n_boundary} "
            f"full_coupled={str(_p8n_full_coupled).lower()} ldlq=false",
            f"tp={world_size} compact_scales={str(_P8N_COMPACT_SCALES).lower()} parent_sha256={parents}",
            f"small_m_scheduler={str(_P8N_SMALL_M).lower()}",
            flush=True,
        )

    def _p8n_process_weights_after_loading(self, layer):
        if not getattr(layer, "_glm53_p8_native", False):
            return _P8N_ORIGINAL_UNQUANTIZED_PROCESS(self, layer)
        _p8n_attach(layer, self)

    def _p8n_modelopt_process_weights_after_loading(self, layer):
        if not getattr(layer, "_glm53_p8_native", False):
            return _P8N_ORIGINAL_MODELOPT_PROCESS(self, layer)
        _p8n_attach(layer, self)

    def _p8n_modelopt_is_monolithic(self):
        if getattr(self, "_glm53_p8_native", False):
            return False
        assert _P8N_ORIGINAL_IS_MONOLITHIC is not None
        return _P8N_ORIGINAL_IS_MONOLITHIC(self)

    def _p8n_run(layer, x, topk_weights, topk_ids):
        output = layer._glm53_p8_native_runtime(x, topk_weights, topk_ids)
        if (_P8N_ROUTE_CAPTURE_MAX and int(layer._glm53_p8_tp_rank) == 0
                and 1 < x.shape[0] <= 16):
            layer_id = int(layer._glm53_p8_layer)
            batch = int(x.shape[0])
            capture_count = _P8N_ROUTE_CAPTURE_COUNTS.get((layer_id, batch), 0)
            if capture_count < _P8N_ROUTE_CAPTURE_MAX:
                routes = topk_ids.detach().cpu().tolist()
                print(
                    "GLM53_P8_ROUTE_CAPTURE "
                    f"layer={layer_id} rank=0 m={batch} call={capture_count} "
                    f"routes={routes}",
                    flush=True,
                )
                _P8N_ROUTE_CAPTURE_COUNTS[(layer_id, batch)] = capture_count + 1
        if (_P8N_SMALL_M and x.shape[0] == 1
                and not getattr(layer, "_glm53_p8_m1_dispatch_logged", False)):
            print(
                "GLM53_P8_M1_DISPATCH "
                f"layer={layer._glm53_p8_layer} rank={layer._glm53_p8_tp_rank} "
                f"fc1_tile_n={_P8N_FC1_TILE_N} "
                f"fused_scratch_zero={str(_P8N_FUSED_SCRATCH).lower()}",
                flush=True,
            )
            layer._glm53_p8_m1_dispatch_logged = True
        if not getattr(layer, "_glm53_p8_native_forward_logged", False):
            runtime = layer._glm53_p8_native_runtime
            boundary = "identity"
            if runtime.scale_component is not None:
                boundary = "h128-suh-svh-scale-component"
            if runtime.full_coupled:
                boundary = "coupled-h512-h128-suh-svh-v1"
            print(
                "GLM53_P8_NATIVE_FORWARD "
                f"layer={layer._glm53_p8_layer} rank={layer._glm53_p8_tp_rank} "
                f"stream=K{runtime.trellis_bits} mma=mxf8f6f4 alphabet=E4M3 scale=UE8M0_K32 "
                f"law=procedural_mcg boundary={boundary} "
                "deterministic=route_topk_sum "
                f"physical_bpw={runtime.trellis_bits + 0.25} ldlq=false",
                f"small_m_scheduler={str(_P8N_SMALL_M).lower()}",
                flush=True,
            )
            layer._glm53_p8_native_forward_logged = True
        return output

    def _p8n_forward_native(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        if not getattr(layer, "_glm53_p8_native", False):
            return _P8N_ORIGINAL_UNQUANTIZED_FORWARD(
                self,
                layer,
                x,
                topk_weights,
                topk_ids,
                shared_experts,
                shared_experts_input,
            )
        return _p8n_run(layer, x, topk_weights, topk_ids)

    def _p8n_modelopt_apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        if not getattr(layer, "_glm53_p8_native", False):
            return _P8N_ORIGINAL_MODELOPT_APPLY(
                self,
                layer,
                x,
                topk_weights,
                topk_ids,
                shared_experts,
                shared_experts_input,
            )
        return _p8n_run(layer, x, topk_weights, topk_ids)

    _P8NativeUnquantizedFusedMoEMethod.process_weights_after_loading = (
        _p8n_process_weights_after_loading
    )
    _P8NativeUnquantizedFusedMoEMethod.forward_native = _p8n_forward_native
    _P8NativeModelOptNvFp4FusedMoE.process_weights_after_loading = (
        _p8n_modelopt_process_weights_after_loading
    )
    _P8NativeModelOptNvFp4FusedMoE.apply = _p8n_modelopt_apply
    _P8NativeModelOptNvFp4FusedMoE.is_monolithic = property(
        _p8n_modelopt_is_monolithic
    )
    _P8N_ORIGINAL_FACTORY = _p8n_glm_model.FusedMoEFactory

    def _p8n_factory(*args, **kwargs):
        prefix = kwargs.get("prefix", "")
        match = _P8N_LAYER_RE.search(prefix)
        runner = _P8N_ORIGINAL_FACTORY(*args, **kwargs)
        if match is None or int(match.group(1)) not in _P8N_LAYERS:
            return runner
        layer_id = int(match.group(1))
        routed = runner.routed_experts
        if not isinstance(
            routed.quant_method,
            (_P8NativeUnquantizedFusedMoEMethod, _P8NativeModelOptNvFp4FusedMoE),
        ):
            raise RuntimeError(
                f"P8 native layer {layer_id} requires BF16 or ModelOpt NVFP4 carrier"
            )
        tp_size = int(routed.moe_config.moe_parallel_config.tp_size)
        tp_rank = int(routed.moe_config.tp_rank)
        if tp_size not in (2,4) or not 0 <= tp_rank < tp_size or (tp_size == 2 and not _P8N_TP2_REPACK):
            raise RuntimeError(f"P8 native requires TP4 or explicit TP2 adapter, got size={tp_size} rank={tp_rank}")
        routed._glm53_p8_native = True
        routed.quant_method._glm53_p8_native = True
        routed._glm53_p8_layer = layer_id
        routed._glm53_p8_tp_rank = tp_rank
        routed._glm53_p8_tp_size = tp_size
        routed._glm53_p8_native_forward_logged = False
        routed._glm53_p8_m1_dispatch_logged = False
        return runner

    _p8n_glm_model.FusedMoEFactory = _p8n_factory
    print(
        f"GLM53_P8_NATIVE_PATCH_ACTIVE layers={_P8N_LAYER_SPEC} tp2_adapter={_P8N_TP2_REPACK} "
        f"design_sha256_allowlist={_P8N_DESIGN_SHA256} K3/K4/K5 procedural_mcg "
        "E4M3 UE8M0_K32 identity deterministic_route_topk_sum "
        "physical_bpw=per_layer ldlq=false",
        f"small_m_scheduler={str(_P8N_SMALL_M).lower()}",
        flush=True,
    )


if P8_PSEUDOQUANT:
    if P8_PSEUDOQUANT not in {"1", "true", "yes", "on"}:
        raise RuntimeError(f"invalid GLM53_P8_PSEUDOQUANT={P8_PSEUDOQUANT!r}")
    import torch as _p8_torch
    import torch.nn.functional as _p8_F
    from safetensors import safe_open as _p8_safe_open
    import vllm.models.glm5next.nvidia.model as _p8_glm_model
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod as _P8UnquantizedFusedMoEMethod,
    )

    _P8_LAYER_SPEC = os.environ.get("GLM53_P8_LAYERS", "3").strip()
    _P8_ARM = os.environ.get("GLM53_P8_PSEUDOQUANT_ARM", "candidate").strip().lower()
    if _P8_ARM not in {"candidate", "control", "hybrid", "mid-butterfly"}:
        raise RuntimeError(f"invalid GLM53_P8_PSEUDOQUANT_ARM={_P8_ARM!r}")
    _P8_MID_BUTTERFLY_ANGLE_PI = 0.0
    if _P8_ARM == "mid-butterfly":
        import math as _p8_math

        _p8_angle_text = os.environ.get(
            "GLM53_P8_MID_BUTTERFLY_ANGLE_PI", ""
        ).strip()
        try:
            _P8_MID_BUTTERFLY_ANGLE_PI = float(_p8_angle_text)
        except ValueError as error:
            raise RuntimeError(
                "mid-butterfly P8 requires a finite scalar angle_pi"
            ) from error
        if (
            not _p8_math.isfinite(_P8_MID_BUTTERFLY_ANGLE_PI)
            or _P8_MID_BUTTERFLY_ANGLE_PI != 0.0625
        ):
            raise RuntimeError(
                "the frozen P8 interaction requires GLM53_P8_MID_BUTTERFLY_ANGLE_PI=0.0625"
            )
    _P8_LAYERS = frozenset(int(value) for value in _P8_LAYER_SPEC.split(",") if value)
    if not _P8_LAYERS or not _P8_LAYERS.issubset(set(range(3, 45))):
        raise RuntimeError(f"invalid GLM53_P8_LAYERS={_P8_LAYER_SPEC!r}")
    _P8_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
    _P8_BOUNDARY_PATHS = [
        value for value in os.environ.get("GLM53_P8_BOUNDARY_FILES", "").split(":") if value
    ]
    if not _P8_BOUNDARY_PATHS and _P8_ARM != "mid-butterfly":
        raise RuntimeError("GLM53_P8_PSEUDOQUANT requires GLM53_P8_BOUNDARY_FILES")
    _P8_POLICY_STATES = tuple(1 for _ in range(288)) if _P8_ARM == "candidate" else tuple(0 for _ in range(288))
    _P8_H128_EXPERTS = frozenset(range(288)) if _P8_ARM == "candidate" else frozenset()
    _P8_POLICY_SHA256 = "none"
    if _P8_ARM == "hybrid":
        import json as _p8_json

        _p8_policy_path = os.environ.get("GLM53_P8_POLICY", "").strip()
        if not _p8_policy_path:
            raise RuntimeError("hybrid P8 pseudoquant requires GLM53_P8_POLICY")
        _p8_policy_bytes = Path(_p8_policy_path).read_bytes()
        _p8_policy = _p8_json.loads(_p8_policy_bytes)
        _p8_schema = _p8_policy.get("schema")
        if _p8_policy.get("phase") != "selection" or _p8_policy.get("ldlq_used") is not False:
            raise RuntimeError("invalid P8 boundary policy")
        if _p8_schema == "glm53-p8-identity-h128-boundary-policy.v1":
            if _p8_policy.get("policy_bytes") != 39:
                raise RuntimeError("invalid P8 V1 policy byte count")
            _P8_H128_EXPERTS = frozenset(int(value) for value in _p8_policy["h128_experts"])
            if len(_P8_H128_EXPERTS) != int(_p8_policy["h128_count"]):
                raise RuntimeError("P8 boundary policy expert count mismatch")
            _P8_POLICY_STATES = tuple(int(expert in _P8_H128_EXPERTS) for expert in range(288))
        elif _p8_schema == "glm53-p8-joint-routed-sum-fc1-policy.v2":
            if (
                _p8_policy.get("policy_bytes") != 75
                or _p8_policy.get("state_names")
                != ["identity", "full-h128", "fc1-h128-identity-down"]
            ):
                raise RuntimeError("invalid P8 V2 policy contract")
            _P8_POLICY_STATES = tuple(int(value) for value in _p8_policy["expert_states"])
            if len(_P8_POLICY_STATES) != 288 or any(value not in (0, 1, 2) for value in _P8_POLICY_STATES):
                raise RuntimeError("invalid P8 V2 expert states")
            _P8_H128_EXPERTS = frozenset(
                expert for expert, state in enumerate(_P8_POLICY_STATES) if state != 0
            )
        else:
            raise RuntimeError("unsupported P8 boundary policy schema")
        _P8_POLICY_SHA256 = hashlib.sha256(_p8_policy_bytes).hexdigest()

    _p8_parts = []
    _p8_next = 0
    for _p8_path in _P8_BOUNDARY_PATHS:
        with _p8_safe_open(_p8_path, framework="pt", device="cpu") as _p8_handle:
            _p8_meta = _p8_handle.metadata() or {}
            if (
                _p8_meta.get("schema") != "glm53-p8-scaled-h128-mcg-layer-chunk.v1"
                or _p8_meta.get("role") != "runtime-boundary"
                or _p8_meta.get("ldlq") != "false"
            ):
                raise RuntimeError(f"invalid P8 boundary artifact: {_p8_path}")
            _p8_start = int(_p8_meta["expert_start"])
            if _p8_start != _p8_next:
                raise RuntimeError(
                    f"non-contiguous P8 boundary artifacts: expected {_p8_next}, got {_p8_start}"
                )
            _p8_part = _p8_handle.get_tensor("down_diagonal").float()
            _p8_parts.append(_p8_part)
            _p8_next += int(_p8_part.shape[0])
    _P8_DOWN_DIAGONAL = (
        _p8_torch.cat(_p8_parts, dim=0).contiguous()
        if _p8_parts
        else _p8_torch.ones((288, 2048), dtype=_p8_torch.float32)
    )
    if _P8_DOWN_DIAGONAL.ndim != 2 or not _p8_torch.isfinite(_P8_DOWN_DIAGONAL).all():
        raise RuntimeError("P8 boundary diagonal is invalid")
    if not (_P8_DOWN_DIAGONAL > 0).all():
        raise RuntimeError("P8 boundary diagonal must be positive")

    def _p8_hadamard128_last(values: _p8_torch.Tensor) -> _p8_torch.Tensor:
        if values.shape[-1] % 128:
            raise RuntimeError("P8 H128 width is not divisible by 128")
        original_shape = values.shape
        work = values.float().reshape(-1, 128).clone()
        width = 1
        while width < 128:
            view = work.reshape(-1, 128 // (2 * width), 2, width)
            left = view[:, :, 0, :].clone()
            right = view[:, :, 1, :].clone()
            view[:, :, 0, :] = left + right
            view[:, :, 1, :] = left - right
            width *= 2
        return work.mul_(128**-0.5).reshape(original_shape)

    def _p8_qdq_e4m3_k32(values: _p8_torch.Tensor) -> _p8_torch.Tensor:
        if values.shape[-1] % 32:
            raise RuntimeError("P8 E4M3 carrier width is not divisible by 32")
        blocks = values.float().reshape(*values.shape[:-1], values.shape[-1] // 32, 32)
        maximum = blocks.abs().amax(-1)
        exponent = _p8_torch.ceil(
            _p8_torch.log2((maximum / 448.0).clamp(min=2.0**-127))
        ).clamp(-127, 128)
        scale = _p8_torch.exp2(exponent)
        scale = _p8_torch.where(maximum == 0, _p8_torch.zeros_like(scale), scale)
        inverse = _p8_torch.where(scale == 0, _p8_torch.zeros_like(scale), 1.0 / scale)
        quantized = (blocks * inverse[..., None]).clamp(-448.0, 448.0)
        quantized = quantized.to(_p8_torch.float8_e4m3fn).float()
        return (quantized * scale[..., None]).reshape_as(values).to(_p8_torch.bfloat16)

    def _p8_shared_butterfly16_staged(
        values: _p8_torch.Tensor,
    ) -> _p8_torch.Tensor:
        """Exact staged reference for the fused +pi/16 middle transform."""
        if values.shape[-1] % 16:
            raise RuntimeError("P8 middle-butterfly width is not divisible by 16")
        angle = _p8_torch.tensor(
            _p8_math.pi * _P8_MID_BUTTERFLY_ANGLE_PI,
            dtype=_p8_torch.float32,
            device=values.device,
        )
        cosine = _p8_torch.cos(angle)
        sine = _p8_torch.sin(angle)
        work = values.to(_p8_torch.bfloat16).float().reshape(-1, 16).clone()
        for stride in (1, 2, 4, 8):
            previous = work.clone()
            for base in range(0, 16, 2 * stride):
                for offset in range(stride):
                    left = base + offset
                    right = left + stride
                    a = previous[:, left]
                    b = previous[:, right]
                    work[:, left] = cosine * a - sine * b
                    work[:, right] = sine * a + cosine * b
        return work.reshape_as(values).to(_p8_torch.bfloat16)

    def _p8_forward(
        layer,
        x: _p8_torch.Tensor,
        topk_weights: _p8_torch.Tensor,
        topk_ids: _p8_torch.Tensor,
    ) -> _p8_torch.Tensor:
        carrier = _p8_qdq_e4m3_k32(x)
        output = _p8_torch.zeros_like(x)
        flat_ids = topk_ids.reshape(-1)
        active = _p8_torch.unique(flat_ids, sorted=True).tolist()
        diagonal = layer._glm53_p8_down_diagonal
        if diagonal.device != x.device:
            diagonal = diagonal.to(device=x.device)
            layer._glm53_p8_down_diagonal = diagonal
        intermediate = layer.w2_weight.shape[-1]
        if layer.w13_weight.shape[1] != 2 * intermediate:
            raise RuntimeError("P8 pseudoquant requires contiguous [gate;up] W13")
        for expert in active:
            locations = (topk_ids == expert).nonzero(as_tuple=False)
            token_index = locations[:, 0]
            slot_index = locations[:, 1]
            expert_input = carrier.index_select(0, token_index)
            gate = _p8_F.linear(expert_input, layer.w13_weight[expert, :intermediate])
            up = _p8_F.linear(expert_input, layer.w13_weight[expert, intermediate:])
            boundary_state = _P8_POLICY_STATES[expert]
            if boundary_state == 1:
                gate = _p8_hadamard128_last(gate).clamp(max=10.0)
                up = _p8_hadamard128_last(up).clamp(-10.0, 10.0)
                middle = _p8_F.silu(gate) * up * diagonal[expert]
                middle = _p8_hadamard128_last(middle)
            elif boundary_state == 2:
                gate = _p8_hadamard128_last(gate).clamp(max=10.0)
                up = _p8_hadamard128_last(up).clamp(-10.0, 10.0)
                middle = _p8_F.silu(gate) * up
            else:
                gate = gate.clamp(max=10.0)
                up = up.clamp(-10.0, 10.0)
                middle = _p8_F.silu(gate) * up
            if _P8_ARM == "mid-butterfly":
                middle = _p8_shared_butterfly16_staged(middle)
            middle = _p8_qdq_e4m3_k32(middle)
            partial = _p8_F.linear(middle, layer.w2_weight[expert])
            partial = partial * topk_weights[token_index, slot_index, None]
            output.index_add_(0, token_index, partial.to(output.dtype))
        if not getattr(layer, "_glm53_p8_forward_logged", False):
            rank = (
                str(_p8_torch.distributed.get_rank())
                if _p8_torch.distributed.is_available() and _p8_torch.distributed.is_initialized()
                else os.environ.get("LOCAL_RANK", "unknown")
            )
            print(
                "GLM53_P8_PSEUDOQUANT_FORWARD "
                f"layer={layer._glm53_p8_layer} rank={rank} "
                f"active_experts={len(active)} carrier=E4M3_K32 arm={_P8_ARM} "
                f"boundary={'shared_butterfly_p00625' if _P8_ARM == 'mid-butterfly' else ('hybrid_policy' if _P8_ARM == 'hybrid' else ('H128_balance_0.5' if _P8_ARM == 'candidate' else 'identity'))} "
                f"angle_pi={_P8_MID_BUTTERFLY_ANGLE_PI} "
                f"policy_sha256={_P8_POLICY_SHA256} ldlq=false",
                flush=True,
            )
            layer._glm53_p8_forward_logged = True
        return output

    _P8_ORIGINAL_PROCESS = _P8UnquantizedFusedMoEMethod.process_weights_after_loading
    _P8_ORIGINAL_FORWARD_NATIVE = _P8UnquantizedFusedMoEMethod.forward_native

    def _p8_process_weights_after_loading(self, layer):
        if getattr(layer, "_glm53_p8_pseudoquant", False):
            # Preserve checkpoint-order [gate;up] BF16 parameters.  Backend
            # conversion can transpose/interleave them and would invalidate
            # the explicit reference evaluator below.
            self.moe_kernel = None
            print(
                "GLM53_P8_PSEUDOQUANT_WEIGHTS_READY "
                f"layer={layer._glm53_p8_layer} w13={tuple(layer.w13_weight.shape)} "
                f"w2={tuple(layer.w2_weight.shape)}",
                flush=True,
            )
            return
        return _P8_ORIGINAL_PROCESS(self, layer)

    def _p8_forward_native(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        if getattr(layer, "_glm53_p8_pseudoquant", False):
            return _p8_forward(layer, x, topk_weights, topk_ids)
        return _P8_ORIGINAL_FORWARD_NATIVE(
            self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
        )

    _P8UnquantizedFusedMoEMethod.process_weights_after_loading = _p8_process_weights_after_loading
    _P8UnquantizedFusedMoEMethod.forward_native = _p8_forward_native
    _P8_ORIGINAL_FACTORY = _p8_glm_model.FusedMoEFactory

    def _p8_factory(*args, **kwargs):
        prefix = kwargs.get("prefix", "")
        match = _P8_LAYER_RE.search(prefix)
        runner = _P8_ORIGINAL_FACTORY(*args, **kwargs)
        if match is None or int(match.group(1)) not in _P8_LAYERS:
            return runner
        layer_number = int(match.group(1))
        routed = runner.routed_experts
        if not isinstance(routed.quant_method, _P8UnquantizedFusedMoEMethod):
            raise RuntimeError(
                f"P8 layer {layer_number} did not select the unquantized BF16 carrier"
            )
        tp_size = int(routed.moe_config.moe_parallel_config.tp_size)
        tp_rank = int(routed.moe_config.tp_rank)
        full_width = int(_P8_DOWN_DIAGONAL.shape[1])
        if full_width % tp_size:
            raise RuntimeError("P8 boundary width is not divisible by TP size")
        local_width = full_width // tp_size
        local = _P8_DOWN_DIAGONAL[:, tp_rank * local_width : (tp_rank + 1) * local_width]
        routed.register_buffer(
            "_glm53_p8_down_diagonal", local.to(_p8_torch.float32), persistent=False
        )
        routed._glm53_p8_pseudoquant = True
        routed._glm53_p8_layer = layer_number
        routed._glm53_p8_forward_logged = False
        return runner

    _p8_glm_model.FusedMoEFactory = _p8_factory
    print(
        "GLM53_P8_PSEUDOQUANT_PATCH_ACTIVE "
        f"layers={_P8_LAYER_SPEC} experts={_P8_DOWN_DIAGONAL.shape[0]} "
        f"intermediate={_P8_DOWN_DIAGONAL.shape[1]} arm={_P8_ARM} "
        f"angle_pi={_P8_MID_BUTTERFLY_ANGLE_PI} "
        f"h128_experts={len(_P8_H128_EXPERTS)} policy_sha256={_P8_POLICY_SHA256} ldlq=false",
        flush=True,
    )


if ROUTED_EXPERTS_SPARSE_MLA_PATCH:
    if ROUTED_EXPERTS_SPARSE_MLA_PATCH not in {"1", "true", "yes", "on"}:
        raise RuntimeError(
            "invalid GLM53_ROUTED_EXPERTS_SPARSE_MLA_PATCH="
            f"{ROUTED_EXPERTS_SPARSE_MLA_PATCH!r}"
        )

    # GLM-5.3's sparse-MLA KV layout deliberately combines every MLA and
    # kpool-indexer layer into one UniformTypeKVCacheSpecs group.  vLLM's
    # routed-expert recorder only recognizes a bare FullAttentionSpec and
    # therefore rejects this otherwise attention-equivalent wrapper.  Unwrap
    # only a homogeneous all-full-attention group, require exactly one match,
    # and leave KpoolTailSpec/MambaSpec groups ineligible.
    from vllm.model_executor.layers.fused_moe import (
        routed_experts_capturer as _routed_experts_capturer,
    )
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec as _FullAttentionSpec,
        UniformTypeKVCacheSpecs as _UniformTypeKVCacheSpecs,
    )

    def _glm53_get_routed_experts_attn_gid(kv_cache_config):
        matches = []
        descriptions = []
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            if isinstance(spec, _FullAttentionSpec):
                eligible = True
                inner_types = [type(spec).__name__]
            elif isinstance(spec, _UniformTypeKVCacheSpecs):
                inner = list(spec.kv_cache_specs.values())
                eligible = bool(inner) and all(
                    isinstance(item, _FullAttentionSpec) for item in inner
                )
                inner_types = sorted({type(item).__name__ for item in inner})
            else:
                eligible = False
                inner_types = [type(spec).__name__]
            descriptions.append(
                f"gid={gid}:outer={type(spec).__name__}:"
                f"inner={','.join(inner_types)}:eligible={eligible}"
            )
            if eligible:
                matches.append(gid)
        if len(matches) != 1:
            raise ValueError(
                "GLM53 routed-expert capture requires exactly one full-attention "
                f"KV cache group, got {matches}; groups={';'.join(descriptions)}"
            )
        print(
            "GLM53_ROUTED_EXPERTS_SPARSE_MLA_PATCH_ACTIVE "
            f"attn_gid={matches[0]} groups={';'.join(descriptions)}",
            flush=True,
        )
        return matches[0]

    _routed_experts_capturer.get_routed_experts_attn_gid = (
        _glm53_get_routed_experts_attn_gid
    )


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
    from b12x.integration.vllm import fp6_serving as _fp6_serving
    from vllm.model_executor.layers.quantization import modelopt as _modelopt

    _fp6_plugin.register_b12x_fp6()
    if _fp6_plugin._CONFIG_CLS is None:
        raise RuntimeError("B12X MXFP6 quantization plugin did not register")
    _B12X_FP6_CONFIG = _fp6_plugin._CONFIG_CLS(
        os.environ["B12X_FP6_MODEL_DIR"]
    )

    # GLM-5.3-Flash uses SiluAndMulWithClamp(swiglu_limit=10).  The pinned
    # v79 MXFP6 bridge preserved only the string activation name, so the
    # otherwise clamp-capable B12X kernel planned an ordinary, unclipped SiLU
    # path.  Carry the model-locked limit into Caps; include it in the scratch
    # cache identity so an unclipped plan can never be reused accidentally.
    _w6a8_limit_raw = os.environ.get("GLM53_W6A8_SWIGLU_LIMIT", "").strip()
    if _w6a8_limit_raw:
        import torch as _torch

        _w6a8_limit = float(_w6a8_limit_raw)
        if not (_w6a8_limit > 0.0):
            raise RuntimeError(
                f"GLM53_W6A8_SWIGLU_LIMIT must be positive, got {_w6a8_limit!r}"
            )

        def _plan_and_scratch_with_glm_clamp(self, m, topk, device):
            from b12x.moe import fused_moe as _fused_moe

            key = (
                int(m),
                int(topk),
                device,
                id(self.weight_plan),
                bool(self.apply_router_weight_on_input),
                "glm53-swiglu-limit",
                _w6a8_limit,
            )
            cached = _fp6_serving._SCRATCH_CACHE.get(key)
            if cached is None:
                plan = _fused_moe.plan(
                    _fused_moe.Caps(
                        max_tokens=m,
                        num_topk=topk,
                        device=device,
                        weight_plan=self.weight_plan,
                        core_token_counts=(m,),
                        route_num_experts=0,
                        quant_mode="w6a8_mx",
                        apply_router_weight_on_input=self.apply_router_weight_on_input,
                        swiglu_limit=_w6a8_limit,
                    )
                )
                scratch = tuple(
                    _torch.empty(
                        shape,
                        dtype=dtype,
                        device=plan.scratch_specs()[index].device,
                    )
                    for index, (shape, dtype) in enumerate(plan.shapes_and_dtypes())
                )
                cached = (plan, scratch)
                _fp6_serving._SCRATCH_CACHE[key] = cached
            return cached

        _fp6_serving.B12XFP6MoEMethod._plan_and_scratch = (
            _plan_and_scratch_with_glm_clamp
        )
        print(
            f"GLM53_W6A8_SWIGLU_CLAMP_ACTIVE limit={_w6a8_limit:g}",
            flush=True,
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
    if MODE not in {"had16", "had32", "had64", "learned"}:
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


    def _hadamard(size: int) -> torch.Tensor:
        h = torch.ones((1, 1), dtype=torch.float32)
        while h.shape[0] < size:
            h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
        return h / (size**0.5)


    def _rotation_for(layer: int, kind: str) -> torch.Tensor:
        if MODE in {"had16", "had32", "had64"}:
            return _hadamard(int(MODE[3:]))
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


    def _apply_block_rotation(hidden_states: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
        size = rotation.shape[-1]
        blocks = hidden_states.shape[-1] // size
        if blocks * size != hidden_states.shape[-1]:
            raise RuntimeError(f"rotated width is not divisible by {size}")
        if rotation.ndim == 3 and rotation.shape[0] != blocks:
            raise RuntimeError(
                f"rotation has {rotation.shape[0]} blocks for runtime width "
                f"{hidden_states.shape[-1]} ({blocks} blocks)"
            )
        if rotation.device != hidden_states.device or rotation.dtype != hidden_states.dtype:
            rotation = rotation.to(
                device=hidden_states.device, dtype=hidden_states.dtype
            )
        view = hidden_states.reshape(*hidden_states.shape[:-1], blocks, size)
        if rotation.ndim == 2:
            output = view @ rotation
        else:
            output = torch.einsum("...bg,bgh->...bh", view, rotation)
        return output.reshape_as(hidden_states)


    class _Block16RoutedInput(nn.Module):
        def __init__(self, layer: int):
            super().__init__()
            rotation = _rotation_for(layer, "in")
            if rotation.ndim not in (2, 3) or rotation.shape[-2] != rotation.shape[-1]:
                raise RuntimeError(f"invalid layer {layer} rotation shape {tuple(rotation.shape)}")
            gram = rotation.transpose(-1, -2) @ rotation
            # vLLM constructs each worker under a rank-local default CUDA
            # device.  A learned safetensors matrix may therefore already be
            # on that device; keep the orthogonality reference colocated.
            eye = torch.eye(rotation.shape[-1], dtype=rotation.dtype, device=rotation.device)
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
            output = _apply_block_rotation(hidden_states, rotation)
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
            inputs = _apply_block_rotation(inputs, rotation)
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
        output.copy_(_apply_block_rotation(output, rotation))
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
            eye = torch.eye(
                in_rotation.shape[-1], dtype=in_rotation.dtype, device=in_rotation.device
            )
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
                mid_rotation.shape[-1],
                dtype=mid_rotation.dtype,
                device=mid_rotation.device,
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
