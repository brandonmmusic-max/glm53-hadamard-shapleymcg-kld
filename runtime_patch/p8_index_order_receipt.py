"""Serving-only receipt for the pinned P8 index-order transformation.

Install this after ``p8_index_order.install()`` and before importing the fused
indexer.  It does not alter the transformed kernel source: it wraps the source
loader once, verifies the loader's completed receipt, and then wraps only the
public paged-indexer entry point to report its first successful eligible call.
"""
from __future__ import annotations

import functools
import json
import os
import sys
import threading
from types import ModuleType
from typing import Any


MODE = "logical-short-v1"
MODULE = "b12x.attention.nsa_indexer.fused_indexer"
ORIGINAL_SHA256 = "69110dcf9d54d4e14ee4d501990245a2cbad7621d0a3d84f035f93d368add7af"
EMITTED_SHA256 = "655236be67b31d61a159fcce01ad85fe74b9c2445aedcbac3b474fb664e6de86"
CACHE_SUFFIX = "_p8logicalshortv1"
PREFIX = "GLM53_P8_INDEX_ORDER_RECEIPT "
SCHEMA = "glm53-p8.index-order-serving-receipt.v1"

_INSTALLED = False


def _validated_source_receipt(index_order: ModuleType) -> dict[str, Any]:
    receipt = index_order.TRANSFORMATIONS.get(MODULE)
    if not isinstance(receipt, dict):
        raise RuntimeError("P8 index-order loader completed without its source receipt")
    expected = {
        "module": MODULE,
        "mode": MODE,
        "original_sha256": ORIGINAL_SHA256,
        "emitted_sha256": EMITTED_SHA256,
        "inspect_source_sha256": EMITTED_SHA256,
        "cache_suffix": CACHE_SUFFIX,
    }
    observed = {key: receipt.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError(
            "P8 index-order source receipt differs from the serving contract: "
            + json.dumps(observed, sort_keys=True, separators=(",", ":"))
        )
    return receipt


def _rank_context() -> dict[str, Any]:
    """Read initialized rank state without importing or initializing a backend."""
    torch = sys.modules.get("torch")
    if torch is None:
        raise RuntimeError("eligible P8 index-order call occurred before torch import")
    distributed = getattr(torch, "distributed", None)
    if (
        distributed is None
        or not distributed.is_available()
        or not distributed.is_initialized()
    ):
        raise RuntimeError("eligible P8 index-order call occurred before distributed init")

    parallel_state = sys.modules.get("vllm.distributed.parallel_state")
    if parallel_state is None or not parallel_state.model_parallel_is_initialized():
        raise RuntimeError("eligible P8 index-order call occurred before vLLM TP init")
    return {
        "tp_rank": int(parallel_state.get_tensor_model_parallel_rank()),
        "tp_world_size": int(parallel_state.get_tensor_model_parallel_world_size()),
        "global_rank": int(distributed.get_rank()),
        # These are observations, not rank authorities: the mp executor passes
        # ranks as constructor arguments and may leave launcher values inherited.
        "local_rank_env": os.environ.get("LOCAL_RANK"),
        "rank_env": os.environ.get("RANK"),
    }


def _eligible_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    # The pinned API is keyword-only.  A positional call is left to the original
    # function to reject with its native error and can never qualify a receipt.
    return (
        not args
        and kwargs.get("num_heads") == 32
        and kwargs.get("topk") == 512
        and kwargs.get("output_physical_slots", False) is False
    )


def _wrap_entrypoint(module: ModuleType, index_order: ModuleType) -> None:
    original = getattr(module, "run_fused_paged_indexer", None)
    if not callable(original):
        raise RuntimeError("pinned fused indexer has no paged entry point")
    if getattr(original, "_glm53_p8_index_order_receipt", False):
        raise RuntimeError("P8 index-order paged entry point is already receipt-wrapped")

    source_receipt = _validated_source_receipt(index_order)
    emitted = False
    emit_lock = threading.Lock()

    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any):
        nonlocal emitted
        eligible = _eligible_call(args, kwargs)
        ranks = _rank_context() if eligible and not emitted else None
        result = original(*args, **kwargs)
        if eligible and not emitted:
            with emit_lock:
                if not emitted:
                    assert ranks is not None
                    marker = {
                        "schema": SCHEMA,
                        "module": MODULE,
                        "mode": MODE,
                        "original_sha256": source_receipt["original_sha256"],
                        "emitted_sha256": source_receipt["emitted_sha256"],
                        "inspect_source_sha256": source_receipt[
                            "inspect_source_sha256"
                        ],
                        "cache_suffix": source_receipt["cache_suffix"],
                        "num_heads": 32,
                        "topk": 512,
                        "output_physical_slots": False,
                        "pid": os.getpid(),
                        **ranks,
                    }
                    print(
                        PREFIX
                        + json.dumps(marker, sort_keys=True, separators=(",", ":")),
                        flush=True,
                    )
                    emitted = True
        return result

    wrapped._glm53_p8_index_order_receipt = True  # type: ignore[attr-defined]
    module.run_fused_paged_indexer = wrapped


def install(index_order: ModuleType | None = None) -> None:
    """Wrap the already-installed exact loader; fail closed on ordering drift."""
    global _INSTALLED
    if os.environ.get("GLM53_P8_INDEX_ORDER_RECEIPT") != "1":
        raise RuntimeError("P8 index-order receipt requires GLM53_P8_INDEX_ORDER_RECEIPT=1")
    if os.environ.get("GLM53_P8_INDEX_ORDER") != MODE:
        raise RuntimeError(f"P8 index-order receipt requires GLM53_P8_INDEX_ORDER={MODE}")
    if _INSTALLED:
        raise RuntimeError("duplicate P8 index-order receipt installation")
    if MODULE in sys.modules:
        raise RuntimeError("P8 index-order receipt must install before fused-indexer import")

    if index_order is None:
        import p8_index_order as index_order
    if index_order.TRANSFORMATIONS:
        raise RuntimeError("P8 index-order receipt must install before source transformation")
    loader = getattr(index_order, "_Loader", None)
    original_exec = getattr(loader, "exec_module", None)
    if loader is None or not callable(original_exec):
        raise RuntimeError("P8 index-order exact loader is unavailable")
    if getattr(original_exec, "_glm53_p8_index_order_receipt", False):
        raise RuntimeError("P8 index-order exact loader is already receipt-wrapped")

    @functools.wraps(original_exec)
    def exec_module(self, module):
        result = original_exec(self, module)
        if self.name == MODULE:
            _wrap_entrypoint(module, index_order)
        return result

    exec_module._glm53_p8_index_order_receipt = True  # type: ignore[attr-defined]
    loader.exec_module = exec_module
    _INSTALLED = True

