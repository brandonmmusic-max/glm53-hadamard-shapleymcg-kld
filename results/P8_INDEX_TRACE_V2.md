# P8 index-order diagnostic v2

Two fresh, same-N128 serving processes reproduced index-order non-repeatability with identical recorded scorer inputs. The differing order was accompanied by different inner-MLA output with identical recorded query, cache content, lengths, tables and selected token sets. This supports the proposed ordering mechanism; it is not an intervention-proven correction, numerical qualification, or KLD result.

## Observed result

Both processes completed the same previously opened `conditional-fit-0056` window: one prompt token and 2,047 exact forced output tokens. Pre-mask logits came from a BF16 head and were preserved by exact conversion to FP32, shape `[2047,154880]`.

The first 259 logit rows matched bitwise. Rows 259–2046 differed: 1,788 rows, 271,308,586 unequal values, maximum absolute difference 9.375. This is **same N128 versus same N128**, not evidence that N64 caused the mismatch.

The earliest observed same-input order divergence was position 259, layer 7, TP rank 1:

- Repeat 1 ordered pool IDs: `[64, 0, 1, …, 63]`.
- Repeat 2 ordered pool IDs: `[0, 1, …, 63, 64]`.
- Quantized index query, index weights, pooled-cache payload/scales, pool lengths and first two pool-page mappings were bitwise identical.
- Pool sets, logical token sets and consumed token sets were equal. Only pool/logical/physical ordering and inner-MLA output differed among the recorded seams.
- Attention query, gathered logical cache contents, lengths and page table were identical. Independent raw-byte review found 2,410 of 16,384 attention-output bytes differed.

The preregistered same-input/order-linked predicates matched four correlated subsamples:

| Position | Layer | TP rank |
| --- | --- | --- |
| 259 | 7 | 1 |
| 260 | 3 | 1 |
| 261 | 3 | 2 |
| 263 | 3 | 1 |

These are not four independent trials: the experimental unit is the serving process, and there were two processes and one window. The first recorded differing query/cache inputs were downstream at position 259, layer 11, rank 0; later-layer differences must not be treated as independent causes.

## Source mechanism and runtime evidence

The actual model uses `index_topk=2048`, KPool4 and 64-row virtual pool pages—not top-k 256. Position 259 corresponds to sequence length 260 and the 65th complete pool. The pinned B12X fused indexer partitions pages across CTAs and atomically appends their candidates to a shared slab. For candidate counts below top-k512, the final output retains that arrival order rather than canonicalizing it. At 65 pools, a second page first contributes candidates. Throughout this 2,047-token window, every complete pool fits within 512 slots; score-based truncation is unnecessary.

Exact image source: `/opt/infernal-invocation/b12x/b12x/attention/nsa_indexer/fused_indexer.py`, page distribution at lines 1385–1398 and atomic merge/direct-copy path at lines 2292–2364; SHA256 `69110dcf9d54d4e14ee4d501990245a2cbad7621d0a3d84f035f93d368add7af`.

Each repeat independently authenticated:

- TP4, no EP, DCP1, NVFP4 MLA KV, max-sequences 1, N128 and separate scratch clearing.
- 168 native forward receipts and 168 actual M1 dispatch receipts.
- Four-rank FULL graph capture, four V2-ready markers and four completed lexical startup-warmup scopes.
- All 11 sparse layers (3, 7, …, 43), positions 255–263, 13 observed seams: 143 base buffers/429 NPZ arrays per rank. Raw per-row counts equal one, error counts zero, graph-recorded flags true, and exact source transformation identities.
- Exact causal request/response token IDs and complete raw-logit hashes.

The observer copied pooled FP8 payload and FP32 scales using their page-wide split layout, and packed 288-byte MLA records through the MLA page table. Invalid future cache rows were masked and saved as zero. DCP1 score values were not copied. Graph evidence includes the inserted observer nodes, not merely an enablement flag.

## Scope and limitations

This diagnostic amended the failed v1 observer loader only to prevent inherited Python future-annotation compiler flags. It did not change the indexer, attention, MoE arithmetic, ordering, or cache contents. Both runs used image `sha256:0f1eae9329965d68713857e4a5a12e9c5440c866b532e7ba288dc2ae4067fad9` with sealed, in-memory observation transforms.

The observers add graph nodes and can perturb CTA scheduling. A positive ordering mismatch is observable evidence; two matching instrumented runs would not have falsified the hypothesis. Internal scratch/reduction state was not exhaustively observed. No order intervention or uninstrumented corrected run has established causal closure or an acceptable fix.

The separate native P8 arithmetic remains procedural trellis-to-E4M3 with K32 `mxf8f6f4` MMA. Covering K64 requires two such MMA steps versus one native NVFP4/E2M1 K64 MMA issue: a 2× instruction-issue requirement, **not a measured 2× end-to-end slowdown**. This correctness diagnostic provides no residual performance attribution; it does not replace a fresh optimized-N64 profile.

No teacher logits were opened or KLD scored. No protected roles were opened. No numerical, speed, or allocation gate advances. A next experiment requires a separately sealed targeted intervention or investigation of the first differing inputs.

## Lifecycle and preservation

The diagnostic ran 2026-09-05 08:58:20–09:02:20 UTC. Repeat thermal maxima were 51°C and 59°C (initial cooldown samples 30/26/27/28°C and 48/38/42/42°C). Both owned containers were removed. The prior backend-active/timer-inactive state was restored with no errors. Terminal unit state: inactive/success, MainPID 0, no remaining matching container.

Plan: `experiments/p8-index-trace-v2.json`, SHA256 `d8e814b5a25ea7d557d2efeb297914aad4c3d98d338e1ef816c68d7c1ecddd2d`.

The [terminal snapshot](../evidence/opened/codec-v2/p8-index-trace-v2/report.json) preserves the [complete replayed comparison](../evidence/opened/codec-v2/p8-index-trace-v2/comparison.json), root/stage execution receipts, sanitized ready/final logs, runtime audits, thermals, capture metadata and trace manifests. Its [manifest](../evidence/opened/codec-v2/p8-index-trace-v2/snapshot.json) binds all 39 sealed source files and all 26 receipted artifacts per stage. Full comparison and sanitized runtime replay passed, independently corroborating the separate reviewer replay.

Large logits and trace NPZ files remain at `/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-smallm-scheduler-v1/index-trace-v2`, referenced by exact hashes; no large arrays were copied. Private launch/container/error/log bytes and request/response content are hash-only. Reproduction helper: `scripts/snapshot_p8_index_trace.py`; it refuses overwrite and requires the original terminal receipts and sealed sources.
