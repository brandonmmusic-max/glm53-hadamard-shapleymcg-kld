# P8 full-model forced-M1 canary: exact gate failed

Both v3 canary arms completed all 2,047 rows, but their logits are not bitwise
equal. The full 32-window stages did not start. The original decision remains
failed; no tolerance was relaxed and no full-model KLD is claimed here.

## Primary, predeclared comparison

Plan SHA-256:
`7b675117a4a2c8a5abe87056dae6a3b6bcd1db2d44d8d0b5c45ad0f33b219085`.
Source commit: `b519337bd51101ff8487d5d38c37c2f729871d59`.
Image: `sha256:0f1eae9329965d68713857e4a5a12e9c5440c866b532e7ba288dc2ae4067fad9`.
Window: conditional-fit-0056, one fixed 2,048-token sequence yielding 2,047
causal prediction rows over 154,880 vocabulary entries. The first row is
one-token prefill; every later row is true M1 decode. Original BF16 head logits
are stored losslessly as FP32 before sampler masking or forced-token selection.

The arms are N128/no combined scratch and N64/combined scratch, both TP4,
no EP, DCP1, V2/FULL graphs, NVFP4 MLA KV, no speculation, same P8 weights.
Both stage receipts exited 0 and validated all forced output tokens, capture
hashes, 168 layer/rank M1 dispatches, four graph-capture ranks and four completed
warmup scopes. The port preflight fix allowed N64 startup after N128 shutdown.

The exact gate failed:

- Rows 0–258 are bit-exact.
- Every row 259–2046 differs: 1,788 of 2,047 rows.
- Unequal values: 271,221,737.
- Maximum absolute logit difference: 7.37451171875.
- `canary-exact.json` SHA-256:
  `6d6a16679528bbe6890bfb4a1df1b88d7f4a37607e1902a5eae1995310e98bb2`.

The root exited 1 at the predeclared exact-canary gate. Both owned containers
were removed, backend restored active and timer inactive as found; protected
roles remained unopened. This is a failed numerical gate, not an infrastructure
failure and not a measured KLD loss.

The [terminal snapshot](../evidence/opened/codec-v2/p8-forced-m1-v2-v3-gate/snapshot.json)
has SHA-256 `25718eed8bfc87f42ac1cafc9001c11cbb3c12d2fc6f3f2bd1c2fb451e5e7f49`.
It preserves the exact failed comparison, both execution/runtime receipts,
and three-capture diagnostic outputs. All 34 v3 stage files and 17 earlier
N128 stage files authenticated. Raw logits remain local, referenced by hash.

## Post-hoc repeated-N128 diagnostic

The earlier v2 attempt had already completed N128 before its port-preflight
stop. Comparing that preserved N128 capture with v3's N128 provides a repeat
of the control. This comparison is post-hoc and does not replace the primary
gate or authorize either full stage.

The control also fails against itself: rows 0–258 match exactly, then every
row 259–2046 differs. There are 271,384,744 unequal values and maximum absolute
difference 7.9140625. At the first differing row, 128,174 of 154,880 values
differ, with maximum absolute difference 0.15625.

Independent auditing verified identical images, environment, topology,
N128 settings, non-bytecode runtime-patch source files, request token sequence,
returned forced tokens and original logit dtype/width. Differences were run
identities, paths, timestamps, and the launcher port preflight. No numerical
configuration change was found between these controls.

Therefore the N128-versus-N64 failure is confounded by baseline cross-run
non-repeatability beginning at exactly the same boundary. It does not isolate
a bug in N64 or the trellis decoder. This is not a basis to call either arm
correct: repeated full-model bitwise determinism is unproven for the baseline
as well. Raw logit differences are not teacher-KLD measurements.

## Next diagnosis, not a concluded mechanism

Trace the shared serving path at causal position259, after 260 input tokens.
The actual nested model config has `index_topk=2048`, `index_kpool=4`, pooled
compression enabled and tail selection enabled. Thus a claimed top-k-256
transition would be unsupported. The boundary also corresponds to completion
of the 65th four-token pool; audit actual pooled-KV page storage, reads and
dispatch before attributing causality. This is a source-tracing hypothesis,
not evidence that those components caused the difference.

The exact-image source audit then identified a mechanism consistent with the
boundary. Attention allocates 3,840-token blocks; KPool4 storage divides to
960 pools, and the installed worker layout chooses 64-row virtual pool pages.
The fused indexer partitions these pages among CTAs. Its short-sequence path
atomically reserves output offsets and copies candidates in CTA arrival order
when the selected count fits within the 512-pool limit. At pool65, a second
page first becomes nonempty. The pool expansion and physical-slot mapping
preserve that order rather than canonicalizing it. Thus identical selected
sets can reach floating-point attention reductions in different orders.

Audited fused-indexer SHA-256:
`69110dcf9d54d4e14ee4d501990245a2cbad7621d0a3d84f035f93d368add7af`,
installed image path `b12x/attention/nsa_indexer/fused_indexer.py`, page split
lines1385–1398 and atomic relay lines2292–2364. The order-preserving consumers
are `vllm/models/glm5next/nvidia/ops/kpool_compress.py:1047` and
`vllm/v1/attention/backends/mla/b12x_mla_sparse.py:2832`. These are paths inside
the immutable image, not assertions about an arbitrary local donor checkout.
The entire 2,047-row window remains below 512 complete pools, so actual top-k
truncation is unnecessary here. This source-backed explanation is not yet
device-proven causation.

An independent storage/write audit found no contradictory page-allocation
mechanism: position259 maps to compressed slot64, the second virtual page
inside an already allocated manager block, not a newly allocated KV block.
The completed pool is written before scoring; later unwritten slots should
remain masked. Existing writer equivalence coverage exercises only eight
pools and does not test this boundary, so this is not a device-level clearance
of the writer. The trace should also preserve slot64's K bytes and scale,
the first two translated page-table entries, and sorted `(pool_id, score)`
multisets alongside ordered IDs.

The discriminating follow-up should compare same-N128 runs around rows255–263:
ordered pool IDs versus sorted sets, expanded tokens, physical selected slots,
and layer3 pre/post-attention values. Same sets with different order before
the first hidden-state difference supports the ordering mechanism. Different
sets, cache values or an earlier divergence redirects diagnosis. Any tracing
overhead must be kept separate from throughput evidence.

Do not restart the full panel, change numerical thresholds, or resume
allocation until the repeatability issue has been diagnosed under a new
predeclared diagnostic. P8 retains E4M3 `mxf8f6f4` and twice NVFP4's MMA issue
count. Its five-cold speed result remains a speed-only serving-system result,
not numerical qualification. P4 remains a separate, unqualified endpoint.
