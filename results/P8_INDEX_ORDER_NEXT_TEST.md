# P8 index-order: bounded next-test proposal

Status: proposal only. This document does not authorize a build, GPU run, runtime
change, or qualification claim.

## Evidence being acted on

The two instrumented N128 processes first diverged at causal row 259, sparse
layer 7, TP rank 1. At that seam the recorded scorer query, weights, canonical
pool cache, lengths, page tables, MLA query, and canonical MLA cache were
bitwise equal. The selected pool set was also equal, but its order was
`[64, 0, ..., 63]` in one process and `[0, ..., 64]` in the other, followed by
a different sparse-attention output. An independent replay reproduced the
finding. This is localization evidence, not proof that the proposed change is
correct or sufficient.

The pinned B12X source is
`/opt/infernal-invocation/b12x/b12x/attention/nsa_indexer/fused_indexer.py`,
SHA-256
`69110dcf9d54d4e14ee4d501990245a2cbad7621d0a3d84f035f93d368add7af`.
Its page partition is at lines 1385-1401. On the short serial-relay arm,
lines 2299-2305 atomically reserve each CTA's slab offset, lines 2306-2311
publish its local carry at that arrival-dependent offset, and lines 2313-2327
publish/count arrivals. When `total <= topk`, lines 2356-2365 copy that slab
order directly to the output. Therefore two CTAs containing pool ranges
`[0,64)` and `[64,65)` may legally produce either observed order.

## Proposed one-variable intervention

Add one narrowly guarded branch inside the existing in-kernel merge:

- compile-time contract: paged input, logical (not physical-slot) output,
  `topk == 512`, and cross-CTA merge enabled;
- runtime contract: compressed `seq_len <= 512` pools;
- leave scoring, local carry construction, pool-to-token expansion, incomplete
  tail handling, MLA remapping, and sparse attention unchanged.

For that branch only, bypass `pack_values`, `pack_indices`,
`_STATE_OUTPUT_COUNTER`, and `_STATE_ARRIVAL_COUNTER`. The page partition is
contiguous and deterministic:

```text
logical_base = page_start * 64
logical_end  = min(page_end * 64, seq_len)
```

Before entering this branch, require the source invariant
`carry_count == logical_end - logical_base`. Each CTA writes its existing
`s_c0_values[i]` and `s_c0_gindex[i]` directly to
`out_[group_id, logical_base + i]`. CTA 0 alone fills `[seq_len, 512)` with
`(-inf, -1)`. These ranges are disjoint, so no cross-CTA atomic allocator,
arrival counter, grid barrier, or extra sort launch is needed. Existing
CTA-local synchronization still publishes the completed local carry, and the
kernel-completion boundary publishes all output before the downstream kernel.
The merge state remains zero because this branch never touches it.

The guard is deliberately `<= 512` **pools**, not original tokens. It covers
the observed 65-pool boundary and is valid because every live candidate is
retained; no top-k choice is being made. Empty CTAs write nothing. At 512 pools
the output is exactly full. At 513 pools and above, retain the current atomic
pack, release fence, arrival publication, last-arriver acquire fence, radix
selection/copy, and counter reset verbatim. Do not generalize this first test
to other top-k sizes, physical output, contiguous-K layouts, DCP merge, or the
cooperative long-sequence arm.

This chooses logical pool order, not score order. For `total <= topk`, all
candidates are retained and score order has no selection meaning; logical order
is the stable order already observed in one replay and expected by the later
pool expansion.

## Required gates before any claim

### CPU/static gates

1. Refuse any original fused-indexer source other than SHA-256 `69110dc...` and
   record the emitted-source hash.
2. Structural test: the new branch is reachable only for the exact paged,
   logical-output, K512 contract; the complete pre-existing `>512` relay body
   remains in the `else` arm.
3. Reference geometry for pool lengths
   `{0, 1, 63, 64, 65, 255, 256, 257, 511, 512}` and CTA counts
   `{1, 2, 4, pinned_num_sms}`: every valid logical position has exactly one
   owner, every invalid suffix position has exactly one owner (CTA 0), and no
   ranges overlap or escape `[0,512)`.
4. Reference score/index pairing must survive direct placement. Include
   shuffled physical page tables and a partial last page.
5. Boundary negatives: 513 pools, physical-slot output, non-paged input, and
   top-k other than 512 must select the unchanged legacy arm.
6. Source/interpreter tests must demonstrate that the short branch does not
   read or write either merge counter and does not add a sort/kernel launch.

### Device gates (a separately sealed experiment)

1. Kernel-level equality to a CPU reference at the short lengths above,
   including exact index order, aligned score/index pairs, `-1/-inf` suffix,
   five eager replays, five FULL-CUDA-graph replays, and five fresh processes.
2. Alternate `512 -> 513 -> 512 -> 513` and the reverse order using the same
   caller-owned scratch. This must prove the direct branch leaves state zero
   and the legacy branch still resets it.
3. For 513, 1024, the merge threshold boundary, and one value above it, compare
   candidate versus pinned source using selected `(index, score-bits)` multisets
   and reference top-k membership. Do not demand the old arrival-dependent
   order. Any set or score-bit change fails.
4. Repeat the same N128 forced-decode canary twice. All 2,047 raw-logit rows
   must be bitwise equal; the trace at positions 255-263 must show identical
   logical pool order and identical sparse-attention output at every recorded
   layer/rank. A null or later first divergence is not a pass: localize it.
5. Compare N128 against the N64 fused-small-M candidate on the same forced
   canary, then the full conditional-fit-32 panel. Exact device closure remains
   mandatory before reusing any teacher-KLD score.
6. Five-run bitwise determinism and the already specified cold TP4 serving
   speed gates remain separate final gates. Observer runs are never speed runs.

## Separate tail contract: preserve it in this test

The exact vLLM sources prove a distinct incomplete-pool-tail omission:

- `kpool_compress.py` SHA-256 `01bfba91f667214760e1fe8af8c4151498ae9b0407493a4670e255945ae01a58`,
  lines 1024-1110, writes 2,048 complete-pool columns plus three tail columns;
- `sparse_attn_indexer_kpool.py` SHA-256
  `ac6b64227346e76c5cf7d8c5e5011b4e32d7c2c811982338a6c24513c1304229`,
  lines 936-989, emits that expansion;
- `b12x_mla_sparse.py` SHA-256
  `0499c674b6890266b50fa0d5724dcfbb83cba3917714a6787e5dddc6feb65572`,
  lines 2749-2752, slices only the first 2,048 columns, and lines 2832-2849
  remap only those columns;
- exact-image B12X `api.py` SHA-256 `7ec087b5605eb008fb77cdfbc2845cb5936d4153d0127211f07e27d4afba4ee9`,
  `kernel.py` SHA-256 `ec1cfdbead6e072fc3b2c1a6d4bf3a1cebe577e17d2639b95a85872ec1a97b4c`,
  `io.py` SHA-256 `546dc9c93d6cf6eb2f5e9a9f672a4ff9ac62287b5aac778c8ee782978e57e4a5`,
  and the S3 validity mask show that the resulting `-1` entries are masked;
  there is no separate tail path in the pinned DCP1 C1 decode call.

Thus sequence lengths 261-263 omit their 1-3 incomplete-pool tokens from MLA.
The observed first repeatability divergence is row 259 / sequence length 260,
where the tail count is zero, so tail omission is not its immediate cause.
Tail omission may affect quality, but its KLD impact has not been measured.
Do not repair it in the index-order intervention: changing both variables would
destroy the causal value of the next test. Predeclare a separate repair only
after the deterministic-order result is preserved.
