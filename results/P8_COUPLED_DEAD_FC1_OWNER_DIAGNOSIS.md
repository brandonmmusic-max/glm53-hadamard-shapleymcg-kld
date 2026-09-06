# Coupled N128 M1 owner was constructed but not launched

2026-09-05. Source and immutable-v6 CPU-constructor evidence supersede the
compiler-cache hypothesis in the earlier diagnostic-dispatch report.

For the exact coupled M1 specialization, the constructor produces:

```text
w4a8_m1_materialized=true
w4a8_split_materialized=false
p8_small_m=true
p8_fc1_tile_n=128
p8_full_coupled=true
external_materialized_fc1=false
external_materialized_fc2=true
```

The external-FC1 predicate only selected split dense paths or narrow N32/N64.
It omitted N128 scale-sandwich/full-coupled M1. Constructing `P8H128FC1Kernel`
or replacing it with the raw diagnostic subclass did not change the false call
gate. Therefore the external owner was dead and the monolithic FC1 consumer
ran instead. The saved eight-row quantized carrier plus packed scale-tail writes
are consistent with that path. No cache-collision hypothesis is needed.

This also explains why the previously corrected external-owner scale/gather
indices could not repair the observed v4 output: that owner did not execute.
The previous wrong-expert-pointer finding is a separate real defect that becomes
relevant when the external M1 owner is actually enabled; it did not establish
that the old monolithic computation used expert 0 for every route.

## Decision before the next image

Enable external FC1 for `p8_small_m and p8_scale_sandwich`, covering the intended
ordinary coupled candidate as well as its diagnostic. Do not make a diagnostic-
only bypass that leaves the actual requested candidate on the wrong path.
Keep identity N128 unchanged and preserve existing dense/narrow selection.
Prove constructor selection plus external launch and internal suppression in
regression tests. `consumer_live` is zero when external FC1 is enabled, so this
switch selects one FC1 implementation; it does not add a second FC1 GEMM.

The failed coupled synthetic tests were not end-to-end tests of the intended
complete coupled kernel chain. This finding does not invalidate unrelated
identity-P8/EXL3 baseline measurements, and does not establish a KLD gain.
New device closure and later paired model KLD remain required.

Regime: synthetic M1 P8 E4M3/UE8M0-K32, K4 payload4.25bpw plus coupled metadata;
no attention/KV. P8 MMA issue count remains twice NVFP4. No production restore.
