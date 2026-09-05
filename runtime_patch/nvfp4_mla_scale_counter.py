"""Opt-in, writer-input-only diagnostics for GLM NoPE NVFP4 MLA.

The production writer consumes a 512-wide BF16/FP16 latent after the caller
divides by a calibrated per-layer outer scale.  It partitions that latent into
32 groups of 16, encodes ``group_amax / 6`` in E4M3, and packs the normalized
values as E2M1.  This module observes that exact writer input without changing
it or the 288-byte cache record.

The runtime integration is deliberately diagnostic-only.  It adds GPU work and
host synchronization at dump boundaries, so no timing result from an
instrumented process is admissible.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import weakref
from pathlib import Path
from typing import Any

import torch


LATENT_DIM = 512
GROUP_SIZE = 16
GROUPS_PER_TOKEN = LATENT_DIM // GROUP_SIZE
NOPE_RECORD_BYTES = 288
E2M1_MAX = 6.0
E4M3_MAX = 448.0
WRITER_AMAX_LIMIT = E2M1_MAX * E4M3_MAX
E4M3_RNE_ZERO_MAX = 2.0**-10

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"", "0", "false", "no", "off"})


def _env_enabled(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise RuntimeError(f"{name} must be a boolean, got {value!r}")


def _layer_index(layer_name: str) -> int | str:
    match = _LAYER_RE.search(layer_name)
    return int(match.group(1)) if match is not None else "UNKNOWN"


class _LayerAccumulator:
    """Small persistent tensors updated on the writer's CUDA stream."""

    # Integer vector fields.  Keeping them in one tensor makes reset and dump
    # simple while preserving exact integer counts.
    INT_FIELDS = (
        "writer_calls",
        "valid_tokens",
        "valid_groups",
        "valid_values",
        "saturated_groups",
        "saturated_tokens",
        "all_zero_groups",
        "all_zero_tokens",
        "zero_values",
        "scale_rounds_zero_groups",
        "nonfinite_values",
        "nonfinite_groups",
        "nonfinite_tokens",
    )
    FLOAT_FIELDS = (
        "max_writer_amax",
        "max_writer_amax_ratio",
        "max_requested_e4m3_scale",
        "max_original_domain_amax",
        "min_outer_scale",
        "max_outer_scale",
    )

    def __init__(self, *, device: torch.device, outer_scale: float) -> None:
        self.ints = torch.zeros(len(self.INT_FIELDS), dtype=torch.int64, device=device)
        self.floats = torch.zeros(
            len(self.FLOAT_FIELDS), dtype=torch.float32, device=device
        )
        # This scalar is read by captured update kernels.  Warmup/capture keeps
        # it at zero; the outer model-runner hook flips it after warmup without
        # recapturing or changing the writer graph.
        self.gate = torch.zeros((), dtype=torch.int64, device=device)
        self.floats[4] = float(outer_scale)
        self.floats[5] = float(outer_scale)

    def reset(self, *, outer_scale: float) -> None:
        self.ints.zero_()
        self.floats.zero_()
        self.floats[4] = float(outer_scale)
        self.floats[5] = float(outer_scale)

    def set_enabled(self, enabled: bool) -> None:
        self.gate.fill_(1 if enabled else 0)

    def update(
        self,
        writer_input: torch.Tensor,
        slot_mapping: torch.Tensor,
        *,
        slot_capacity: int,
        outer_scale: float,
    ) -> None:
        num_tokens = int(slot_mapping.numel())
        if num_tokens == 0:
            return
        values = writer_input[:num_tokens].reshape(
            num_tokens, GROUPS_PER_TOKEN, GROUP_SIZE
        )
        slots = slot_mapping.reshape(-1)
        valid_token = (slots >= 0) & (slots < int(slot_capacity))
        valid_group = valid_token[:, None]
        valid_value = valid_group[:, :, None]

        finite = torch.isfinite(values)
        finite_abs = torch.where(finite, values.abs(), torch.zeros_like(values))
        group_amax = finite_abs.amax(dim=-1).float()
        requested_scale = group_amax / E2M1_MAX
        saturated_group = valid_group & (requested_scale > E4M3_MAX)
        saturated_token = saturated_group.any(dim=-1)
        all_zero_value = finite & (values == 0)
        all_zero_group = valid_group & all_zero_value.all(dim=-1)
        all_zero_token = valid_token & all_zero_value.all(dim=-1).all(dim=-1)
        nonfinite_value = (~finite) & valid_value
        nonfinite_group = valid_group & (~finite).any(dim=-1)
        nonfinite_token = valid_token & (~finite).any(dim=-1).any(dim=-1)
        scale_rounds_zero = (
            valid_group
            & (requested_scale > 0)
            & (requested_scale <= E4M3_RNE_ZERO_MAX)
        )

        counts = torch.stack(
            (
                torch.ones((), dtype=torch.int64, device=writer_input.device),
                valid_token.sum(dtype=torch.int64),
                valid_group.expand_as(group_amax).sum(dtype=torch.int64),
                valid_value.expand_as(values).sum(dtype=torch.int64),
                saturated_group.sum(dtype=torch.int64),
                saturated_token.sum(dtype=torch.int64),
                all_zero_group.sum(dtype=torch.int64),
                all_zero_token.sum(dtype=torch.int64),
                (all_zero_value & valid_value).sum(dtype=torch.int64),
                scale_rounds_zero.sum(dtype=torch.int64),
                nonfinite_value.sum(dtype=torch.int64),
                nonfinite_group.sum(dtype=torch.int64),
                nonfinite_token.sum(dtype=torch.int64),
            )
        )
        self.ints.add_(counts * self.gate)

        masked_amax = torch.where(
            valid_group, group_amax, torch.zeros_like(group_amax)
        ).amax()
        new_maxima = torch.stack(
            (
                masked_amax,
                masked_amax / WRITER_AMAX_LIMIT,
                masked_amax / E2M1_MAX,
                masked_amax * float(outer_scale),
                torch.tensor(float(outer_scale), device=writer_input.device),
                torch.tensor(float(outer_scale), device=writer_input.device),
            )
        ).to(torch.float32)
        enabled = self.gate.to(torch.bool)
        enabled_f32 = self.gate.to(torch.float32)
        self.floats[:4] = torch.maximum(
            self.floats[:4], new_maxima[:4] * enabled_f32
        )
        self.floats[4] = torch.where(
            enabled, torch.minimum(self.floats[4], new_maxima[4]), self.floats[4]
        )
        self.floats[5] = torch.where(
            enabled, torch.maximum(self.floats[5], new_maxima[5]), self.floats[5]
        )

    def snapshot(self) -> dict[str, int | float]:
        ints = self.ints.detach().cpu().tolist()
        floats = self.floats.detach().cpu().tolist()
        result: dict[str, int | float] = {
            key: int(value) for key, value in zip(self.INT_FIELDS, ints, strict=True)
        }
        result.update(
            {
                key: float(value)
                for key, value in zip(self.FLOAT_FIELDS, floats, strict=True)
            }
        )
        valid_groups = int(result["valid_groups"])
        valid_tokens = int(result["valid_tokens"])
        result["saturated_group_fraction"] = (
            float(result["saturated_groups"]) / valid_groups
            if valid_groups
            else 0.0
        )
        result["saturated_token_fraction"] = (
            float(result["saturated_tokens"]) / valid_tokens
            if valid_tokens
            else 0.0
        )
        return result


class WriterScaleCounter:
    """Request-scoped collection with explicit reset/dump boundaries."""

    def __init__(self) -> None:
        self.active_request_id: str | None = None
        self.layers: dict[str, _LayerAccumulator] = {}
        self.layer_labels: dict[str, dict[str, Any]] = {}

    def reset(self, request_id: str) -> None:
        if not request_id:
            raise ValueError("request_id must be non-empty")
        self.active_request_id = request_id
        for accumulator in self.layers.values():
            outer_scale = float(accumulator.floats[5].detach().cpu())
            accumulator.reset(outer_scale=outer_scale)
            accumulator.set_enabled(True)

    def disarm(self) -> None:
        self.active_request_id = None
        for accumulator in self.layers.values():
            accumulator.set_enabled(False)

    def observe(
        self,
        *,
        layer_name: str,
        writer_input: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        outer_scale: float,
        rank_labels: dict[str, Any],
    ) -> None:
        if writer_input.ndim != 2 or int(writer_input.shape[1]) != LATENT_DIM:
            raise ValueError(
                f"writer_input must be (tokens,{LATENT_DIM}), got "
                f"{tuple(writer_input.shape)}"
            )
        if writer_input.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise TypeError(f"unsupported writer_input dtype {writer_input.dtype}")
        if kv_cache.ndim != 3 or int(kv_cache.shape[-1]) != NOPE_RECORD_BYTES:
            raise RuntimeError(
                "NVFP4 MLA scale counter is restricted to the static 288-byte "
                f"NoPE ABI, got cache shape {tuple(kv_cache.shape)}"
            )
        if kv_cache.dtype != torch.uint8:
            raise TypeError(f"expected uint8 NVFP4 cache, got {kv_cache.dtype}")
        if slot_mapping.dtype != torch.int64:
            raise TypeError(f"slot_mapping must be int64, got {slot_mapping.dtype}")
        if not math.isfinite(float(outer_scale)) or float(outer_scale) <= 0:
            raise ValueError(f"outer_scale must be finite and positive, got {outer_scale}")
        if int(writer_input.shape[0]) < int(slot_mapping.numel()):
            raise ValueError("writer_input does not cover slot_mapping")

        accumulator = self.layers.get(layer_name)
        if accumulator is None:
            accumulator = _LayerAccumulator(
                device=writer_input.device, outer_scale=float(outer_scale)
            )
            self.layers[layer_name] = accumulator
            self.layer_labels[layer_name] = {
                "layer_name": layer_name,
                "layer_index": _layer_index(layer_name),
                **rank_labels,
            }
            accumulator.set_enabled(self.active_request_id is not None)
        accumulator.update(
            writer_input,
            slot_mapping,
            slot_capacity=int(kv_cache.shape[0]) * int(kv_cache.shape[1]),
            outer_scale=float(outer_scale),
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "request_id": self.active_request_id,
            "layers": [
                {**self.layer_labels[name], **self.layers[name].snapshot()}
                for name in sorted(self.layers)
            ],
        }


class _RuntimeController:
    def __init__(self, control_path: Path, output_dir: Path) -> None:
        self.control_path = control_path
        self.output_dir = output_dir
        self.counter = WriterScaleCounter()
        self.last_sequence = -1
        self.last_identity: tuple[int, int, int, int] | None = None
        self.last_rank_labels: dict[str, Any] | None = None
        self.lock = threading.RLock()

    @staticmethod
    def _rank_labels(impl: Any | None) -> dict[str, Any]:
        global_rank: int | str = "UNKNOWN"
        tp_rank: int | str = "UNKNOWN"
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            global_rank = int(torch.distributed.get_rank())
        elif os.environ.get("RANK", "").isdigit():
            global_rank = int(os.environ["RANK"])
        try:
            from vllm.distributed.parallel_state import get_tp_group

            tp_rank = int(get_tp_group().rank_in_group)
        except Exception:
            pass
        local_rank: int | str = (
            int(os.environ["LOCAL_RANK"])
            if os.environ.get("LOCAL_RANK", "").isdigit()
            else "UNKNOWN"
        )
        if not isinstance(global_rank, int):
            raise RuntimeError(
                "NVFP4 MLA scale counter requires an initialized distributed "
                "global rank or an integer RANK environment variable"
            )
        return {
            "global_rank": global_rank,
            "local_rank": local_rank,
            "tp_rank": tp_rank,
            "dcp_rank": int(getattr(impl, "dcp_rank", 0)),
            "tp_world_size": int(getattr(impl, "tp_world_size", 1)),
            "dcp_world_size": int(getattr(impl, "dcp_world_size", 1)),
        }

    def _receipt_path(self, sequence: int, global_rank: int | str) -> Path:
        rank_text = f"{global_rank:03d}" if isinstance(global_rank, int) else "unknown"
        return self.output_dir / (
            f"nvfp4-mla-scale-counter-seq-{sequence:06d}-rank-{rank_text}.json"
        )

    def _write_receipt(
        self,
        *,
        command: dict[str, Any],
        dumped: dict[str, Any] | None,
        rank_labels: dict[str, Any],
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sequence = int(command["sequence"])
        path = self._receipt_path(sequence, rank_labels["global_rank"])
        payload = {
            "schema": "glm53.nvfp4-mla-scale-counter-receipt.v1",
            "sequence": sequence,
            "action": command["action"],
            "counter_abi": {
                "kv_cache_dtype": "nvfp4_ds_mla",
                "record_bytes": NOPE_RECORD_BYTES,
                "latent_dim": LATENT_DIM,
                "group_size": GROUP_SIZE,
                "e2m1_max": E2M1_MAX,
                "e4m3_max": E4M3_MAX,
                "writer_amax_limit": WRITER_AMAX_LIMIT,
                "mutates_writer_input_or_cache": False,
            },
            "rank": rank_labels,
            "dump": dumped,
            "timing_admissible": False,
        }
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)

    def _process_command(self, rank_labels: dict[str, Any]) -> None:
        try:
            stat = self.control_path.stat()
        except FileNotFoundError:
            return
        identity = (stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size)
        if identity == self.last_identity:
            return
        command = json.loads(self.control_path.read_text(encoding="utf-8"))
        if not isinstance(command, dict):
            raise ValueError("counter control must contain a JSON object")
        sequence = command.get("sequence")
        action = command.get("action")
        if type(sequence) is not int or sequence < 0:
            raise ValueError("counter control sequence must be a non-negative integer")
        if sequence <= self.last_sequence:
            self.last_identity = identity
            return
        if action not in {"reset", "dump_reset", "dump"}:
            raise ValueError(f"unsupported counter action {action!r}")

        dumped: dict[str, Any] | None = None
        if action in {"dump", "dump_reset"}:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dumped = self.counter.snapshot()
            expected = command.get("dump_request_id")
            if expected != dumped["request_id"]:
                raise RuntimeError(
                    "counter dump request mismatch: "
                    f"expected {expected!r}, active {dumped['request_id']!r}"
                )
        if action in {"reset", "dump_reset"}:
            request_id = command.get("request_id")
            if not isinstance(request_id, str) or not request_id:
                raise ValueError(f"{action} requires non-empty request_id")
            self.counter.reset(request_id)
        else:
            self.counter.disarm()

        self._write_receipt(command=command, dumped=dumped, rank_labels=rank_labels)
        self.last_sequence = sequence
        self.last_identity = identity

    def observe(
        self,
        impl: Any,
        writer_input: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        with self.lock:
            rank_labels = self._rank_labels(impl)
            self.last_rank_labels = rank_labels
            owner_ref = getattr(impl, "_nvfp4_scale_counter_owner", None)
            owner = owner_ref() if owner_ref is not None else None
            if owner is None:
                raise RuntimeError("NVFP4 scale counter lost its MLAAttention owner")
            self.counter.observe(
                layer_name=str(owner.layer_name),
                writer_input=writer_input,
                kv_cache=kv_cache,
                slot_mapping=slot_mapping,
                outer_scale=float(getattr(owner, "_nvfp4_mla_outer_scale", 1.0)),
                rank_labels=rank_labels,
            )

    def before_execute_model(self) -> None:
        """Process request commands outside CUDA-graph capture/replay."""

        with self.lock:
            if self.last_rank_labels is None:
                # A ready server has executed/captured every MLA layer.  If a
                # command arrives earlier, leave it pending instead of emitting
                # an unlabeled or empty receipt.
                return
            self._process_command(self.last_rank_labels)


_INSTALLED = False
_RUNTIME: _RuntimeController | None = None


def install() -> None:
    """Install the opt-in observer without changing the writer signature."""

    global _INSTALLED, _RUNTIME
    if _INSTALLED:
        return
    control = os.environ.get("VLLM_NVFP4_MLA_COUNTER_CONTROL", "").strip()
    output = os.environ.get("VLLM_NVFP4_MLA_COUNTER_OUTPUT_DIR", "").strip()
    if not control or not output:
        raise RuntimeError(
            "NVFP4 MLA scale counter requires VLLM_NVFP4_MLA_COUNTER_CONTROL "
            "and VLLM_NVFP4_MLA_COUNTER_OUTPUT_DIR"
        )
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention
    from vllm.v1.attention.backends.mla.b12x_mla_sparse import B12xMLASparseImpl
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    runtime = _RuntimeController(Path(control), Path(output))
    original_init = MLAAttention.__init__
    original_update = B12xMLASparseImpl.do_kv_cache_update
    original_execute_model = GPUModelRunner.execute_model

    def observed_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self.impl._nvfp4_scale_counter_owner = weakref.ref(self)

    def observed_update(
        self: Any,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        if kv_cache_dtype == "nvfp4_ds_mla" and bool(
            getattr(self, "_kv_nope_nvfp4", False)
        ):
            runtime.observe(self, kv_c_normed, kv_cache, slot_mapping.flatten())
        return original_update(
            self,
            kv_c_normed,
            k_pe,
            kv_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
        )

    def observed_execute_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        # This method is outside the captured model graph.  It can therefore
        # reset/dump persistent counter tensors between API requests while the
        # reductions themselves remain part of FULL CUDA-graph replay.
        runtime.before_execute_model()
        return original_execute_model(self, *args, **kwargs)

    MLAAttention.__init__ = observed_init
    B12xMLASparseImpl.do_kv_cache_update = observed_update
    GPUModelRunner.execute_model = observed_execute_model
    _RUNTIME = runtime
    _INSTALLED = True
    print(
        "GLM53_NVFP4_MLA_SCALE_COUNTER installed=true record_bytes=288 "
        "timing_admissible=false",
        flush=True,
    )


def install_from_environment() -> None:
    if _env_enabled("VLLM_NVFP4_MLA_SCALE_COUNTER"):
        install()
