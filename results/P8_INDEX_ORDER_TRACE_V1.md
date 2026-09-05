# P8 pooled-index ordering diagnostic, preregistered

Status: v1 stopped at worker import before evaluation. See
[the preserved failure and v2 amendment](P8_INDEX_TRACE_IMPORT_V2.md).
This does not change the failed v3 closure gate or authorize the full
conditional-fit panel. The text below preserves the predeclared v1 design.

## Decision before result

Two independently started N128/noncombined-scratch processes will replay the
same already-opened conditional-fit-0056 2,048-token input, producing 2,047
forced prediction rows each. Both retain TP4/noEP/DCP1, V2/FULL graphs,
NVFP4 MLA KV and the existing all-42-layer native P8 sidecars. There is no
teacher-logit opening, KLD scoring, speed test, sorting or codec change here.

The observer records positions255–263 for every sparse layer3,7,...43 on all
four ranks. The independent unit is the fresh process. Rows, layers and ranks
are correlated diagnostic observations, not independent statistical samples.
Single-window localization is intentionally not a domain-balanced KLD panel.

Observe, in order: index query/weights, canonical valid pooled-cache bytes,
pool selection, expanded logical tokens, physical selection mapping, actual
MLA query, canonical valid MLA-cache bytes, then inner MLA output. If scorer
inputs and selected sets match but order differs, the indexer ordering defect
is reproduced at that seam. If attention content/set also match and its output
differs, this supports the ordering linkage; it is not an intervention-proven
fix. Upstream differences redirect diagnosis. No difference in two observed
runs is inconclusive and never triggers an automatic retry or qualification.

## Observer and immutable runtime

The image remains
`sha256:0f1eae9329965d68713857e4a5a12e9c5440c866b532e7ba288dc2ae4067fad9`.
Opt-in, fail-closed import transformations insert read-only observers inside
the existing opaque indexer/attention operations and wrap the sampler's
outside-graph request boundaries. Original and emitted Python hashes are
checked; the immutable image is not edited. ExLlamaV3/MCG and B12X algorithms
are unchanged; these new observer copies are instrumentation, not a port of
an alternative codec. Existing repository attributions remain applicable.

The unmodified B12X fused-indexer SHA is
`69110dcf9d54d4e14ee4d501990245a2cbad7621d0a3d84f035f93d368add7af`.
The pinned nested model config SHA is
`16c46a1e0684671791c833662a81e6d2150fcd5e82216d2427cc8ea97e21ca7c`.

Buffers are allocated during eager startup, never first allocated inside a
captured graph or after real-request registration. Copies use device-side
position gates, stable destination pointers, once-per-row counts and invalid
address flags. File publication occurs only after the sampler finishes the
real request. Replay independently checks all429 arrays per rank (143 data,
143 counts,143 error arrays), every graph flag and exact source transformations.

Important layouts:

- Pool cache:64 rows of128-byte payloads, then64 four-byte scales per page.
  First66 logical pools are copied canonically; invalid entries are not read.
- MLA cache: packed byte records, first264 logical tokens, with every needed
  block-table entry. Physical allocator differences are not mistaken for
  differences in canonical cache content.
- The indexer exposes2051 logical columns including three appended tail slots;
  this pinned attention backend consumes2048 physical columns. Both are
  observed distinctly. This is not a tail-handling repair, and the tail
  distinction cannot alone explain row259, whose token length260 has no tail.

This comprehensive observer adds143 copy-kernel launches per token and can
strongly perturb CTA arrival order. Positive evidence remains useful; a null
does not falsify a scheduling race. It is not a throughput measurement.

## Preflight and launch conditions

122 focused CPU tests passed before the final diagnostic seal. The exact-image Triton CPU
interpreter exercised dynamic row gates, multiple copy CTAs, permuted cache
pages, both payload layouts, invalid-row masking and invalid-address flags.
The final source/interpreter receipt is
`evidence/opened/codec-v2/p8-index-trace-preflight-v1/cpu-interpreter-v2.json`.
Earlier pre-seal receipt/log is retained, not used as the final source seal.

A standalone CPU import initially omitted the serving recipe's PYTHONPATH;
after correcting it, importing the indexer first hit the source's model/import
cycle. The actual serving model-first import order passed without GPU drivers;
CPU-only vLLM extension/driver warnings are not GPU qualification.

The original unlaunched plan `p8-index-trace-v1.json` (SHA
`3e3534d9044d1e0d61fb9748beddb6d25cba4bc2569ee3b23aa35295e466d7f1`)
is retained. A final review required raw-byte equality (distinguishing signed
zero) and paired dtype/shape checks in the analyzer. Before any GPU launch,
the reviewed replacement `p8-index-trace-v1-reviewed.json` was sealed with SHA
`7e21d7ad3d1edbb8c74df4cb85d8e82154c57b48d1d45e7754de1aa882ba0dbd`.
That intermediate plan also remains unlaunched: the final audit required the
same-input index divergence and attention linkage to occur in the same row,
not merely somewhere in each comparison. A negative fixture now enforces it.
Only `p8-index-trace-v1-final.json`, SHA
`a900b61096bd377a9430dea87127d2ebf6da6d4209f8b6cbe1b4f3ee7973b265`,
may run. These are explicit pre-result amendments, not silent repairs of a
failed measurement; the filename does not designate the protected final role.

The runner requires a clean sealed checkout, historical v3 failure receipts,
source/image/config/weight identities, sufficient disk, the maintenance lock,
startup temperature <=75C and abort >=90C. It preserves every attempt, stops
after an execution/trace-validity failure, never retries, and restores the
prior backend/timer state after owned-container cleanup. Two full raw captures
require about2.54GB plus bounded traces and a20GiB safety margin. The full32
window KLD stage and allocation remain disabled.

P8 is still E4M3 `mxf8f6f4`, twice NVFP4's MMA issue count. P4 is separate.
