# P8 H128 scale sandwich: legal N128 owner

Date: 2026-09-05
Branch: `p8-coupled-scale-kernel-v1`
Scope: CPU/static implementation; no GPU, image build, service, KLD, or speed
run was performed while the sealed CF32 campaign was active.

## Decision

Use one N128 CTA as the smallest legal owner of every H128 block.  The existing
N64 arm assigns adjacent 64-column halves to independent CTAs and cannot place
`svh` after H128 without a new grid synchronization boundary.  N128 already
has four warps, the same total threads and MMA work as two N64 CTAs, and its
existing shared-memory pipeline reservation is larger than the two FP16
N128xM16 transform tiles required by the epilogue.  It therefore avoids an
extra kernel launch, global intermediate, or cooperative-grid barrier.

The frozen N32/N64 source remains unchanged.  The scale arm is isolated in
`p8_h128_fc1.py`; static regression continues to compare the narrow kernel AST
to its pinned candidate.

## Implemented ordering

For M1 only:

1. One warp owns each input H128. It loads BF16 channels, multiplies the shared
   signed `gate_up_suh` in FP32, rounds to FP16, performs normalized H128, then
   computes K32 amax and E4M3/UE8M0 bytes from the transformed values.
2. One N128 FC1 CTA accumulates gate/up with the unchanged K4 procedural-MCG
   decode and native E4M3/UE8M0 MMA. Physical results round to FP16. Each warp
   owns one canonical row, performs gate/up H128, applies private signed
   `gate_svh`/`up_svh`, rounds to FP16, activates, rounds to FP16, applies
   private `down_suh`, rounds to FP16, performs down-input H128, and only then
   recomputes K32 amax and E4M3/UE8M0 bytes.
3. FC2 retains the full K512 accumulation for the scale arm, rounds the physical
   down result to FP16, performs output H128 in its owning N128 CTA, applies
   shared signed `down_svh`, then route-weights and writes deterministic route
   output for the existing fixed-order top-k reducer.

No weight stream, weight scale plane, MMA instruction, or extra GEMM changes.
The sidecar still declares `full_coupled=false`: H512 and coupled procedural
signs remain separate work and this arm must not be reported as full coupling.

## Fail-closed envelope

- Exact schema, rank/layer, three tensor hashes, FP16 dtype and shapes required.
- Signed values accepted; zero/non-finite tensors rejected.
- Scale arm requires TP4 GLM H4096/E288/I512, K4, M1 small-M, FC1 N128,
  deterministic output, and the existing direct materialized P8 schedule.
- M other than 1 and requested N32/N64 scale ownership reject before execution.
- Identity sidecars retain the existing path and frozen N32/N64 implementation.

## Static cost estimate

There is no extra GEMM and no added global phase boundary.

- Shared metadata: 901,120 FP16 bytes per TP rank/layer, approximately 0.0040
  bits per routed weight.
- Input prologue, once per token and shared across eight routes: 32 H128 blocks,
  4,096 FP16 scale multiplies/casts, plus warp re-gather before the existing
  128 K32 quant blocks.
- FC1 epilogue/down prologue, per route: 12 H128 transforms and 1,536 scale
  multiplies across gate, up, and down-input roles.
- FC2 epilogue, per route: 32 H128 transforms and 4,096 output-scale multiplies.
- N128 replaces two N64 FC1 CTAs per output block. MMA count is unchanged;
  likely cost is lower CTA-level parallelism plus H128 shuffle/scale/cast work.
  A numerical speed claim requires the later five-cold device benchmark.

## CPU/static gates

The tests establish source syntax, frozen narrow-kernel preservation, strict
sidecar/hash validation, signed-scale admission, multiply-FP16-H128 ordering,
post-H128 E4M3 payload and UE8M0 byte construction, complete scale-sequence
bit identity to an explicit reference, N128 ownership, and M/shape rejection.

Device compilation, captured activation-byte closure, output closure, CUDA-graph
parity, and performance remain mandatory before serving qualification.

CPU/static regression command:

```text
PYTHONPATH=. pytest -q tests/test_p8*.py
```

Result: `680 passed in 24.37s`.
