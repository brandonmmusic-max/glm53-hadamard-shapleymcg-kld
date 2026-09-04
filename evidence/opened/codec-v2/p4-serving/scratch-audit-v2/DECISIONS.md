# P4 scratch audit decisions

1. Before testing this follow-up, scope activation reuse to one explicitly
   identified model namespace/TP rank, stream, capture sequence and shape.
   Separate capture IDs must not share scratch, even if captured on the same
   stream. Pin external-launch resources until worker exit because the adapter
   does not own a graph destruction callback. Reject duplicate model namespaces,
   overlapping host dispatch, DBO and microbatching. Keep the old test receipts
   unchanged. Frozen decision: `experiments/p4-serving-scratch-audit-v2.json`.

2. Require a canonical ascending decimal layer list, so startup readiness is
   byte-consistent with the launcher's literal environment string. Installation
   failure must terminate the process even if traceback reporting fails.
   CPU subprocesses and the exact launcher grep are the acceptance checks.

3. After the first two 147-pass attempts, extend the existing offline link
   symbol assertion to include the newly added `p4_capture_state` C entry.
   No runtime implementation is changed by this coverage addition. Preserve
   attempts 1/2 and run the complete suite again as attempt 3.

These decisions authorize no GPU work or data-role access. They do not qualify
CUDA graph execution, arithmetic closure, KLD or throughput.
