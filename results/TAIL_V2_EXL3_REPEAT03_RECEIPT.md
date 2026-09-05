# Tail V2 EXL3 third cold repetition

Observed 2026-09-05. This is an interim result within the existing five-cold
protocol, not a new selected arm or a changed stopping rule.

All rows below: attention B12X_MLA_SPARSE; KV nvfp4_ds_mla; MoE B12X EXL3
full-expert BF16 path; BF16 activations; EXL3 4.0 bpw; TP4/EP4/DCP4;
Tail V2; CUDA graphs enabled; MTP off. Same 32 conditional-fit windows.
KLD is KL(teacher || student), nats, equal-window mean.

| Cold repetition | Windows | Mean including first prefill row | True-decode mean | Raw hashes equal repetition 1 |
|---|---:|---:|---:|---:|
| 1 | 32 | 0.03349229804040844 | 0.031611840268931456 | 32 (self-reference) |
| 2 | 32 | 0.033124694303131426 | 0.031244056862183633 | 0 |
| 3 | 32 | 0.03292601667370284 | 0.03104528212736431 | 0 |

Third-repetition execution exited zero, cleanup reports `ok=true` and no
errors, and `protected_roles_opened=[]`. Execution receipt SHA256:
`5ef471bd2a26ecffbe8a601c4bd3c2e65bdd4ea59be1ebd4fc3ff4cf2b5c6216`.
Local evidence root:
`/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/tail-v2-p8-exl3-cf32-product-validation-v1b/quality`.
The execution receipt is `round-03-exl3/execution.json`; per-window scores are
`scores/repeat-03/exl3/*.score.json`. The same paths with repeats 01/02 provide
the first two rows. These aggregates were independently recomputed with jq
from all 32 score receipts per repetition; every repeat-03 status is complete.

The score receipt's `raw_retired=false` is deliberately the immutable state
before retirement, not a current raw-file inventory. The runner writes a
separate retirement receipt after unlink/fsync. Read-only inspection after
this repetition found zero `.f32*` files in its capture directory.

The five-cold bitwise determinism gate remains failed; a lower KLD on a later
cold start does not rescue it or establish a new codec improvement. Remaining
predeclared repetitions continue unchanged. This result does not identify the
cause of EXL3 nondeterminism, qualify Tail V2 for shipping, or measure the new
coupled P8 candidate. P8 uses native E4M3/UE8M0 and twice NVFP4's MMA issue
count; no P8 speed inference follows from these EXL3 measurements.
