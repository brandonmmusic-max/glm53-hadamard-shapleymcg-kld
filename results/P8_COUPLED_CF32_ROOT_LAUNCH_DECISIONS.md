# Coupled CF32 launch decisions

1. All three pilot layers (3, 20, 22) have packed sidecars and passing real
   four-rank loader receipts. These are ABI checks, not new KLD measurements.
2. Integrate executor commits 24a5e160, 2c6bb0e and 6bbbdf9, preserving their
   independent review. Root reproduced 30 focused tests before the final
   inventory-only patch. Reviewer passed that patch separately.
3. Preserve the first storage-inventory failure: Python traversal raised
   PermissionError on the unrelated root-owned r17 graph JIT cache. No GPU
   launch occurred. The amended inventory uses read-only privileged find with
   the same cutoff and exclusions; permission errors are not ignored.
4. Fresh ledger p8-coupled-cf32-storage-v1.json projects 29,747,330,832 bytes
   against the user's 30,000,000,000-byte ceiling, including capture allowance
   and reserve. Every capture must still pass prospective and actual checks.
5. Fixed order remains stock, existing identity P8, coupled P8, on the same
   32 conditional-fit windows. Development pass: coupled mean true-decode
   KLD below identity; report paired BCa without making it a blocking gate.
   This compares the whole coupled candidate, not isolated rotation effects.
6. Conditions: TP4/DCP1/noEP, B12X_MLA_SPARSE, nvfp4_ds_mla KV, Tail V2,
   graphs, MTP off. P8 selected projections use E4M3 activations and nominal
   4.25 bpw (coupled tensor payload plus metadata approximately 4.25398 bpw).
   Stock MoE/activation evidence must come from runtime logs. P8's
   mxf8f6f4 instruction has twice NVFP4's MMA issue count. No speed claim.
7. Production remains off. Only scored-and-hashed transient raw logits may
   be retired by the executor; all source chunks remain. No protected
   confirmation logits are opened. New KLD results remain outstanding.
8. Attempt v1 failed before any request: the original source command already
   began with `exec`, and the preparer added another. Bash reported
   `exec: exec: not found`. Zero windows completed, no candidate scores were
   inspected, cleanup passed and production remained off. Preserve v1;
   normalize the leading shell builtin, test it, and use fresh v2 manifests,
   ledger and execution seal. All scientific factors and arm ordering remain
   unchanged. This is a disclosed launch-only protocol amendment.
9. User reprioritized newest-layer KLD ahead of another stock baseline.
   Root canceled v2 during prelaunch input authentication (PID 2651595,
   SIGTERM, session 91337 terminal exit 143). No serving container or
   windows had started. Preserve both v2 seals and all prior evidence.
   Next amendment is coupled P8 first, then matched three-layer identity
   P8; no fresh stock arm is required for this development comparison.
   Historical stock results are contextual until runtime/model/mask
   equivalence is verified. A prior all-42-layer identity P8 result cannot
   substitute for identity on layers 3/20/22 with the rest at stock.
