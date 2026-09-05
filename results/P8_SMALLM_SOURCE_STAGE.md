# P8 small-M source stage

Status: structural implementation; not device-compiled, numerically qualified,
timed, or enabled in serving. Governing plan:
`experiments/p8-smallm-scheduler-v1.json`. No role/logit data was opened.

## Changed path

Explicit `P8NativeTPMoE(..., small_m_scheduler=True)` selects the candidate only
for M1, E288, H4096, I512, topk8, deterministic GLM TP4. M2/M3 and prefill
retain baseline selection. Cache identity includes the candidate flag.

The M1 arm uses B12X's direct route-order M16 preparation and route-by-K128
FC1 schedule, producing 32 FC1 tasks. The existing P8 FC1 body performs the
procedural MCG decode, physical UE8M0 scaling, BF16 projection/activation
boundaries and E4M3 intermediate quantization. Only quantized intermediate
activations and their scales are materialized; no decoded weights are stored.

The candidate sets `external_materialized_fc2`, so the old E2M1/atomic M1 FC2
helper cannot run. `P8SmallMPhase2Kernel` receives original route ids and assigns
128 route-by-N256 tasks. Each task computes two N128 halves using the inherited
P8 compressed-stream staging, M16 E4M3 `mxf8f6f4` MMA and repacked physical
UE8M0/32 scales. It restarts the FC2 accumulator at every K128 slice, rounds the
scaled slice result to BF16, applies the route weight, and updates the BF16
running sum in ascending slice order. The existing fixed-order top-k kernel
then reduces the eight route rows. Invalid route rows are explicitly zeroed.

This is two stream-ordered compute launches plus fixed-order top-k reduction.
FC1 is fused with preparation/activation and cooperative; FC2 is a separate
noncooperative kernel. It does not claim a single fused FC1+FC2 launch. Launch
geometry is host-shape-only; no routing readback or synchronization was added.
The wrapper still allocates scratch during eager calls/capture; graph replay
and poisoned reusable storage are outstanding device gates.

## Provenance and integration

Base image: `sha256:5da4ef3e814a71c6bcc47a7eb409a02fe4e3d5d867261f0b8e2e9b2d6ebb8ef8`.
The FC2 source is adapted from its B12X `w4a8_phase2.py`, SHA256
`be317f7f76153ff5f60d2d1cb76f15dae14e6252a109b2d448e7dd8852f66194`.
The inherited staging/launch methods remain from that pinned base. The new
Dockerfile copies only the changed B12X modules to both site-packages and the
editable `/opt/infernal-invocation/b12x/b12x` tree that Python actually imports.
Both base copies have the hashes stated above. A first CPU constructor probe
mounted only site-packages and correctly exposed the untouched editable tree
(`unexpected keyword argument p8_small_m`); the mount/build paths were corrected.
Supply both host wrapper files
`p8_native_kernel.py` and `p8_smallm_schedule.py` on the runtime import path.
No image build or GPU execution is part of this source stage.

## Required next gates

1. Compile the exact candidate and inspect generated MMA/rounding instructions.
2. Compare identical synthetic payload/routes at full E288 on M1, M2, M3 against
   the frozen baseline, requiring finite and bit-exact BF16 output.
3. Five repeats with poisoned reusable scratch, invalid routes and repeated
   experts; verify stable graph addresses and no replay allocation.
4. Captured kernel A/B on every selected GPU, then integrated four-rank trace.
5. Freeze a new product protocol before any serving throughput claim.

CPU tests establish geometry coverage, fail-closed selection, source syntax,
arithmetic call paths and a counterexample showing why full-K summation would
violate baseline rounding. They do not establish CUDA compilation, device
correctness, reproducibility, performance or KLD preservation.

Validation command: pinned image without `--gpus`, read-only worktree mount,
`/opt/venv/bin/python -m pytest -q -p no:cacheprovider /work/tests/test_p8_smallm_scheduler.py`.
Result: 11 passed. The local campaign venv did not contain pytest; no packages
were installed and the existing pinned image's pytest was used instead.
CPU import/construction against the actual editable B12X source path also
passed: `P8SmallMPhase2Kernel True False True` for the selected class,
M1-materialized flag, external-FC1 flag and external-FC2 flag respectively.
This constructor check does not compile a CUDA kernel.
