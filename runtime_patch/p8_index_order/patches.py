"""Exact-source P8 short-index-order transformation.

This changes only the logical-output, paged, 32-head, K512 cross-CTA kernel.
Rows with more than 512 compressed pools retain the pinned relay verbatim.
"""
from __future__ import annotations

import hashlib
import textwrap


MODULE = "b12x.attention.nsa_indexer.fused_indexer"
ORIGINAL_SHA = "69110dcf9d54d4e14ee4d501990245a2cbad7621d0a3d84f035f93d368add7af"
MODE = "logical-short-v1"
CACHE_SUFFIX = "_p8logicalshortv1"


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"{label}: expected exactly one source anchor, got {count}")
    return source.replace(old, new, 1)


def direct_ranges(seq_len: int, ctas_per_group: int, *, topk: int = 512) -> list[tuple[int, int]]:
    """CPU reference for the direct branch's disjoint logical ownership."""
    if seq_len < 0 or seq_len > topk or topk != 512 or ctas_per_group < 1:
        raise ValueError("direct placement requires 0<=pool seq_len<=topk==512 and CTAs>=1")
    page = 64
    pages = (seq_len + page - 1) // page
    pages_per_cta = (pages + ctas_per_group - 1) // ctas_per_group
    ranges = []
    for cta in range(ctas_per_group):
        page_start = min(cta * pages_per_cta, pages)
        page_end = min(page_start + pages_per_cta, pages)
        begin = min(page_start * page, seq_len)
        end = min(page_end * page, seq_len)
        ranges.append((begin, max(begin, end)))
    return ranges


def _patch_relay(source: str) -> str:
    start_anchor = """            else:
                # ---- in-kernel cross-CTA merge (relay) ----
"""
    start = source.find(start_anchor)
    if start < 0 or source.find(start_anchor, start + 1) >= 0:
        raise ValueError("short relay anchor missing or ambiguous")
    end_anchor = "\n\n\n@lru_cache(maxsize=64)\ndef _build_fused_indexer_kernel("
    end = source.find(end_anchor, start)
    if end < 0 or source.find(end_anchor, end + 1) >= 0:
        raise ValueError("short relay end anchor missing or ambiguous")

    original = source[start:end]
    legacy_body = original[len("            else:\n") :]
    if "atomic_add_global_i32(woff_ptr, carry_count)" not in legacy_body:
        raise ValueError("pinned atomic pack allocator missing")
    if "if total > topk_static:" not in legacy_body:
        raise ValueError("pinned legacy top-k boundary missing")

    direct = '''            else:
                # P8 logical-short-v1: when every live compressed pool is retained,
                # place each CTA's already-scored carry in logical page order.  The
                # exact-table scan is redundant in every CTA, hence the branch is
                # group-uniform; malformed tables keep the pinned atomic relay.
                p8_direct_short = Int32(0)
                if cutlass.const_expr(
                    self.kv_layout == KV_LAYOUT_PAGED
                    and not self.paged_output
                    and self.num_heads_static == 32
                    and self.topk == 512
                ):
                    if tx == Int32(0):
                        p8_ok = Int32(1)
                        p8_scan_pages = total_pages
                        if seq_len > topk_static:
                            p8_ok = Int32(0)
                            # Preserve the legacy >512 path's work: do not add a
                            # redundant O(total_pages) validation scan per CTA.
                            p8_scan_pages = Int32(0)
                        if p8_scan_pages > Int32(real_page_table.shape[1]):
                            p8_scan_pages = Int32(real_page_table.shape[1])
                            p8_ok = Int32(0)
                        p8_page = Int32(0)
                        while p8_page < p8_scan_pages:
                            p8_pid = Int32(real_page_table[q_idx, p8_page])
                            if (
                                p8_pid < Int32(0)
                                or p8_pid >= Int32(k_quant_bytes.shape[0])
                                or p8_pid >= Int32(k_scales.shape[0])
                            ):
                                p8_ok = Int32(0)
                            p8_page += Int32(1)
                        s_relay[0] = p8_ok
                    cute.arch.sync_threads()
                    p8_direct_short = Int32(s_relay[0])

                if p8_direct_short != Int32(0):
                    # The deferred final-page append has no trailing rendezvous on
                    # this no-trim regime.  Publish all score-warp carry writes first.
                    cute.arch.sync_threads()
                    p8_begin = page_start * Int32(_PAGE_SIZE)
                    if p8_begin > seq_len:
                        p8_begin = seq_len
                    p8_end = page_end * Int32(_PAGE_SIZE)
                    if p8_end > seq_len:
                        p8_end = seq_len
                    if p8_end < p8_begin:
                        p8_end = p8_begin
                    p8_span = p8_end - p8_begin
                    # Exact valid-table source invariant: carry_count == p8_span.
                    # Masking the impossible mismatch keeps every output initialized
                    # without introducing a CTA-local fallback into a collective arm.
                    i = Int32(tx)
                    while i < p8_span:
                        p8_valid = i < carry_count
                        out_values[group_id, p8_begin + i] = (
                            Float32(s_c0_values[i])
                            if p8_valid
                            else Float32(float("-inf"))
                        )
                        out_indices[group_id, p8_begin + i] = (
                            Int32(s_c0_gindex[i]) if p8_valid else Int32(-1)
                        )
                        i += Int32(_RADIX_THREADS)
                    # One disjoint owner initializes the unused suffix.  Kernel
                    # completion, not a grid barrier, publishes all CTA ranges.
                    if cta_in_group == Int32(0):
                        i = seq_len + Int32(tx)
                        while i < topk_static:
                            out_values[group_id, i] = Float32(float("-inf"))
                            out_indices[group_id, i] = Int32(-1)
                            i += Int32(_RADIX_THREADS)
                else:
'''
    replacement = direct + textwrap.indent(legacy_body, "    ")
    return source[:start] + replacement + source[end:]


def _patch_cache_key(source: str) -> str:
    anchor = '''    cache_key = tuple(_fused_indexer_tensor_key(name, t) for name, t in key_tensors) + (
        (variant,) + tuple(policy),
    )
'''
    replacement = '''    p8_logical_short_v1 = (
        kernel.kv_layout == KV_LAYOUT_PAGED
        and not kernel.paged_output
        and kernel.num_heads_static == 32
        and kernel.topk == 512
        and kernel.merge_in_kernel
    )
    if p8_logical_short_v1:
        variant += "_p8logicalshortv1"
    cache_key = tuple(_fused_indexer_tensor_key(name, t) for name, t in key_tensors) + (
        (variant,) + tuple(policy),
    )
'''
    return replace_once(source, anchor, replacement, "compile-cache variant")


def transform(name: str, raw: bytes) -> tuple[str, dict]:
    if name != MODULE:
        raise ValueError(f"unsupported index-order transform target: {name}")
    original = hashlib.sha256(raw).hexdigest()
    if original != ORIGINAL_SHA:
        raise ValueError(f"pinned fused-indexer source differs: {original}")
    source = raw.decode("utf-8")
    source = _patch_relay(source)
    source = _patch_cache_key(source)
    compile(source, f"<{MODULE}:{MODE}>", "exec", dont_inherit=True)
    emitted = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return source, {
        "schema": "glm53-p8.index-order-source.v1",
        "module": MODULE,
        "mode": MODE,
        "original_sha256": original,
        "emitted_sha256": emitted,
        "cache_suffix": CACHE_SUFFIX,
        "contract": {
            "kv_layout": "paged",
            "paged_output": False,
            "num_heads_static": 32,
            "topk_pools": 512,
            "cross_cta_merge": True,
            "runtime_pool_seq_len_max": 512,
            "negative_or_missing_page": "legacy-relay",
            "positive_page_oob": "unchanged caller-validity precondition; scorer reads first",
            "tail_expansion_changed": False,
        },
    }
