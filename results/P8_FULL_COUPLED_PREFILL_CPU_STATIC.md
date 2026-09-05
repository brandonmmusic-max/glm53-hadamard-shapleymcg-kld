# P8 full-coupled prefill CPU/static implementation receipt

Date: 2026-09-05  
Branch: `p8-coupled-prefill-v1`  
Implementation commit: `4c73139964ad84ca18b9581a4a1cd1ac47536584`  
Plan commit: `63b486bee8862528d4117c1573ef160c42c5a64b`  
Decode/full-coupled base: `fe1df7a2695a975bce632a613a12756405ed2c84`

Status: **structural implementation only**. No image build, CUDA compile, GPU
run, service action, KLD capture, throughput measurement, or large allocation
was performed. This receipt does not establish device arithmetic or speed.

## Implemented path

The wrapper now has one full-coupled implementation per serving regime:

```text
M=1: existing direct-route M16/N128 decode owner
M>1: forced grouped/materialized M64/N128 path
     BF16 -> FP16 -> normalized H512 -> signed suh -> normalized H128
     -> E4M3/UE8M0/32 activation materialization
     -> native K4 procedural-MCG FC1 MMA
     -> physical gate/up FP16
     -> joint H128 -> gate_svh/up_svh -> H128 -> draw0 pre signs
     -> capped SiLU (gate max10, up +/-10) -> draw0 post signs
     -> H128 -> down_suh -> H128 in FP32
     -> E4M3/UE8M0/32 intermediate
     -> native K4 procedural-MCG FC2 MMA
     -> physical output FP16 -> H128 -> shared down_svh
     -> unweighted FP32 deterministic route scatter
     -> router-order weighted FP32 top-k sum -> normalized H512 -> BF16
```

The two GEMMs remain `mxf8f6f4` with E4M3 A/B operands and UE8M0 scales per
32 elements. The transforms/scales execute in producer or epilogue/prologue
code; no additional GEMM and no BF16 dequantized weight matrix was added.

New exact owners:

- `p8_coupled_prefill_fc1.py`: grouped M64/N128 FC1 traversal over the exact
  N128 full-coupled owner; four M16 accumulator blocks and sixteen four-row
  ownership rounds cover M64 and tails.
- `p8_coupled_prefill_fc2.py`: grouped M64/N128 FC2 traversal; ordered K=512
  FP32 accumulation, physical FP16 boundary, CTA-owned H128/down_svh and
  deterministic pair-index FP32 scatter.

`dynamic.py` now materializes the full H512+suh->H128 input with two warps per
token. Each warp owns alternating complete H512 units, so no transform unit
crosses a warp or CTA boundary.

## Correctness repair found during widening

The inherited full-coupled FC1 used the same shared base for physical FP16
gate/up slabs and transformed FP32 output. One FP32 row spans the bytes of two
FP16 rows, so a warp could overwrite a neighboring row before that warp had
read its physical values. This was a genuine source-level race.

The implementation now keeps the transformed FP32 tile disjoint from both
physical slabs. The repair also applies to the M1 full-coupled owner. It does
not alter scale-only semantics.

Resource consequence:

- M64 FC1 pipeline: 35,328 bytes.
- M64 FC1 safe full-coupled phase: 65,536 bytes shared memory.
- M64 FC2: 34,816 bytes shared memory; its 16,384-byte physical tile aliases
  the retired pipeline safely because there are no later shared writes.
- Procedural MCG uses no 4 KiB shared LUT. A 4 KiB T12 table is not allocated
  in this candidate.

The 64 KiB FC1 bound is expected to permit one CTA/SM on SM120, but actual
register allocation, occupancy and launchability are **Not tested**.

## Exact scale-plane bound and allocation

For shared-input materialization at H=4096:

- producer index: `token * 128 + block`, `block in [0,127]`;
- FC1 consumer reads four bytes starting at
  `token * 128 + 4*k128`, `k128 in [0,31]`;
- both inclusive maxima are `M*128 - 1`;
- required storage is exactly `M*H/32 = M*128` bytes.

Grouped physical route rows never index this scale plane; FC1 first maps a
physical row to a token (deterministic pair index divided by top-k), then reads
the token scale row. Parameterized tests prove identical producer/consumer
bounds for M in `{2,63,64,65,1296,4096}`.

The wrapper therefore right-sizes only the full-coupled, materialized, M>1
arm. Scale-only, M1 and all non-coupled allocations remain on their prior
paths. At M=4096 this changes `scale_flat` from 1,083,211,776 bytes to 524,288
bytes and reduces the plan's conservative per-rank scratch upper bound from
1,891,409,920 to 808,722,432 bytes. The FP32 route plane remains 536,870,912
bytes and is now the dominant cost.

## Source identities

```text
dynamic.py                         aa73d9cd9a813daf22969a69b77fb1603a4eb105e0abc30422bd075e7dc06266
p8_h128_fc1.py                     893acefe29fe96719da5ca52a015927e84278d60f23687ce82ba29c237b5d525
p8_small_m.py                      a92ca1f0c5bee88d51038af5347d7ee265386fb9b96589a2e64442e03016ea33
p8_coupled_prefill_fc1.py          7cac0e5eeeee79afb3f903ea7d71552b60f9835e92cebf032ca951d3153a26c1
p8_coupled_prefill_fc2.py          7aed6e7c6f2b4195982e876d932891d4ed5fabd1ee04d8177d032328721d889c
p8_native_kernel.py                644fd28a1ae3efb2c2e6bb69d3de16a5f16e2fdb0c61553e8070fa4fa94d5370
p8_coupled_prefill_plan.py         441fcf1f854899d39da14e5de07e250b0bdfe66cdf5d8de3802df394d4abecaa
```

The local materialized donor checkout does not contain the expanded
`trellis_codebook`/`trellis_scaled` constructor ABI used by the patched
`dynamic.py`; those base phase classes are inherited from the pinned runtime
image. This is recorded as an unresolved exact-image identity, not papered
over with a different local source. Before build, the candidate Dockerfile
must hash and inspect the actual parent-image phase1/phase2 constructors, then
copy all five changed B12X kernel modules to both active Python trees.

## CPU/static validation

```text
python3 -m py_compile \
  runtime_patch/p8_native_kernel.py \
  runtime_patch/p8_coupled_prefill_plan.py \
  runtime_patch/b12x_h16/b12x/moe/_shared/kernels/dynamic.py \
  runtime_patch/b12x_h16/b12x/moe/_shared/kernels/p8_h128_fc1.py \
  runtime_patch/b12x_h16/b12x/moe/_shared/kernels/p8_small_m.py \
  runtime_patch/b12x_h16/b12x/moe/_shared/kernels/p8_coupled_prefill_fc1.py \
  runtime_patch/b12x_h16/b12x/moe/_shared/kernels/p8_coupled_prefill_fc2.py
PYTHONPATH=. pytest -q tests/test_p8*.py
git diff --check
```

Result: `712 passed in 24.71s`; syntax and diff checks passed. The CPU
reference is bit-exact when the same 65 rows are evaluated as one batch or an
M64 tile plus one-row tail. This checks row separability of the frozen
transform/cast contract; it is not GPU closure.

## Remaining device gates

1. Exact parent-image constructor/source hash and no-GPU import/instantiation.
2. SM120 compile; record registers, shared bytes, achieved occupancy and any
   spill. Reject if FC1 cannot launch at 65,536 bytes.
3. Bit-exact activation payload and UE8M0 closure for M `{2,63,64,65}`.
4. Bit-exact FC1 intermediate payload/scales and FC2 FP32 route values,
   including expert-tail tiles and deterministic pair indexing.
5. Final BF16 closure after router-order sum/H512 against pseudoquant.
6. Five-run determinism with CUDA graph parity.
7. Same-path full-model KLD, prefill throughput and peak-memory measurement.
   Transform, FC1, FC2 and reducer costs must be reported separately.

Until these pass, the supported conclusion is only that an exact source-level
prefill route exists and passes CPU/static invariants. It is not yet a serving
or performance result.
