# P8 full-coupled prefill source map and bounded implementation plan

Date: 2026-09-05
Branch: `p8-coupled-prefill-v1`
Base: `fe1df7a2695a975bce632a613a12756405ed2c84`
Scope: CPU/source audit and immutable geometry only. No CUDA build, GPU run,
service action, model artifact, or large allocation was performed.

Implementation update: the source plan was implemented at commit
`4c73139964ad84ca18b9581a4a1cd1ac47536584`; see
`P8_FULL_COUPLED_PREFILL_CPU_STATIC.md`. Statements below that the M>1 guard
"remains" describe the pre-implementation baseline preserved by plan commit
`63b486bee8862528d4117c1573ef160c42c5a64b`.

## Verdict

M>1 support is feasible without another GEMM and without changing the native
P8 K4/E4M3/UE8M0/32 MMA contract, but it is not a one-line extension of the
M1 kernels. The safe implementation is:

```text
M=1  -> existing direct-route N128 full-coupled path
M>1  -> force split materialized M64/N128 path
        integrated two-warp/token H512 input packing
        new exact full-coupled M64 FC1
        new exact full-coupled M64 FC2
        existing generic FP32 top-k + final-H512 reducer
```

The current M>1 rejection remains in place. Enabling prefill before the two
new M64 kernels exist would silently run identity/ordinary scale arithmetic
against coupled-basis weights.

## Current dispatch and blockers

`runtime_patch/p8_native_kernel.py:451-452` rejects every scale-bearing call
whose M is not 1. Without that guard, the default threshold at `:456-460`
would send M=2..1295 through the monolithic M16 kernel and only M>=1296 through
the split M64 path. Full coupling would consequently need two independent
implementations.

The full-coupled wrapper should instead force `materialized=True` for every
M>1 call. This makes the existing dynamic kernel a route/input-pack front end
and invokes external M64 FC1 and FC2 grids (`dynamic.py:1063-1072,2699-2749`).

The full schema currently trips the M1-only constructor guard at
`dynamic.py:1030-1033`, and `dynamic.py:1135-1139` replaces every scale-bearing
FC1 with the M16-only `P8H128FC1Kernel`. Both selections must become explicit:

```text
full_coupled && p8_small_m -> current P8H128FC1Kernel/P8SmallMPhase2Kernel
full_coupled && split M64  -> new P8CoupledPrefillFC1/P8CoupledPrefillFC2
scale_only                 -> retain current M1-only restriction
```

## Legal ownership

### Input H512 and activation quantization

Dense materialization already quantizes one input row per token, not one per
routed expert. The routing front end assigns two warps to each W4A8 token at
`dynamic.py:3517-3526,3568-3575`, and stores `[token,H]` E4M3 payload plus
`[token,H/32]` UE8M0 at `:3643-3740`.

One warp can legally own a complete 512-wide transform unit:

1. Load four adjacent FP16-rounded input elements from each of four H128
   quarters.
2. Apply four warp-local normalized H128 transforms.
3. Apply normalized H4 across the four register quarters.
4. Multiply shared signed `suh` in FP32 and apply the inner H128.
5. Form all sixteen K32 amax/payload/scale records for that H512 unit.

With two warps per token, `token_partition` owns H512 blocks
`partition, partition+2, ...` through the eight blocks in H=4096. No cross-warp
data exchange is required; the existing pair barrier remains only for route
metadata publication. This is the same register topology as the validated M1
source, generalized from one row to `a_input[token_idx,*]`.

### M64 FC1

The inherited dense phase owns M64xN128, so every H128 stays within one CTA.
The pinned donor is:

- `b12x-coupled-glm52-20260813/.../w4a8_phase1.py`
- SHA-256 `cb72c50dab933ee866103caf7c32f1a6cfb15df991fcdd8b753ac765551946b1`

It contains a prior coupled M64 epilogue, but cannot be used unchanged: it
uses an expert-private 6I rotation table, BF16 physical/output scratch, and the
archived activation/cast contract. The new class must port only its M64 MMA
pipeline and use the exact current ABI:

```text
physical gate/up FP16
gate32/up32 interleave
H128 -> [E,gate_svh|up_svh] -> H128 -> shared draw0 pre signs
capped SiLU -> shared draw0 post signs
H128 -> [E,down_suh] -> H128 in FP32
E4M3/UE8M0/32
```

Four warps each own 16 routed rows. Gate/up FP16 slabs require 32,768 bytes;
the final M64xN128 FP32 activation slab also requires 32,768 bytes and can
alias them after the pipeline retires. The donor pipeline is 35,328 bytes;
with its 4 KiB procedural decode table the upper shared allocation is 39,424
bytes. Existing MMA accumulators are already the dominant register cost
(approximately 128 FP32 gate/up accumulator values per thread before compiler
allocation); the transform adds a small warp-local register tuple but likely
leaves one CTA/SM. This must be confirmed by the eventual build receipt.

### M64 FC2

The pinned donor is:

- `b12x-coupled-glm52-20260813/.../w4a8_phase2.py`
- SHA-256 `ce31085628a8423468a453275f18c3322f8029d35afccb35f5e7060cbab11a7e`

The inherited implementation applies ordinary weighted BF16 output semantics.
The new class must retain the ordered full K=512 FP32 accumulator, then execute
`physical FP16 -> H128 -> shared down_svh`, and write unweighted FP32 route
values using deterministic `token_map` pair indices. Its stage allocation is
34,816 bytes, or 38,912 bytes with the 4 KiB decode table. A 16,384-byte
M64xN128 FP16 physical slab fits by aliasing the retired pipeline.

### Output H512

`p8_coupled_topk.py` already parameterizes its grid by `active_m` and assigns
one 128-thread CTA per `(token,H512 block)`. It accepts route-major FP32 values,
sums eight routes in router order, applies H128 tensor H4, and stores BF16.
No new output kernel is needed. The M64 FC2 scatter index must be proven to be
the original deterministic pair index before this reducer is enabled.

## Exact source-level implementation sequence

1. Add `p8_coupled_prefill_fc1.py` by porting the pinned M64 phase-1 pipeline,
   not by widening the M16 kernel with runtime conditionals.
2. Add `p8_coupled_prefill_fc2.py` from the pinned M64 phase-2 pipeline with
   full-K accumulation and unweighted FP32 deterministic scatter.
3. In `dynamic.py`, admit full coupling only for either the existing exact M1
   shape or `(materialize_intermediate, share_input, deterministic, M64,N128)`;
   select the new classes only for the latter.
4. Replace the dense direct K32 input loads with the warp-owned H512 sequence
   only under `p8_full_coupled`. Preserve the ordinary path byte-for-byte.
5. In `p8_native_kernel.py`, force the split M64 arm for full-coupled M>1,
   retain the scale-only rejection, and keep the FP32 route scratch/reducer.
6. Prove source index bounds for packed input, scale, intermediate, route
   scratch and deterministic token-map scatter before shrinking allocations.
7. Add CPU reference tensors for M in `{2,63,64,65,1296}` and tail expert
   tiles. Device closure then compares every input/intermediate E4M3 byte,
   UE8M0 byte, route FP32 value and final BF16 output before KLD or speed.

## Memory bounds

`runtime_patch/p8_coupled_prefill_plan.py` reproduces the current wrapper's
conservative capacity formulas. Important exact examples per TP rank/layer:

| M | FP32 route scratch | current scratch upper bound |
|---:|---:|---:|
| 2 | 0.25 MiB | 91.71 MiB |
| 512 | 64 MiB | 304.80 MiB |
| 1,296 | 162.0 MiB | 632.71 MiB |
| 4,096 | 512 MiB | 1.761 GiB |
| 8,192 | 1 GiB | 3.434 GiB |
| 32,768 | 4 GiB | 13.472 GiB |

At M=4096 the exact current components are:

| allocation | bytes |
|---|---:|
| packed A capacity | 209,715,200 |
| inherited `scale_flat` allocation | 1,083,211,776 |
| intermediate payload/scales | 27,033,600 |
| FP32 route output | 536,870,912 |
| BF16 output | 33,554,432 |
| conservative controls | 1,024,000 |
| total upper bound | 1,891,409,920 |

Only 524,288 bytes are logically required for shared-input `[M,H/32]` scales
at M=4096, versus the inherited 1.083 GB allocation. This discrepancy should
be resolved by an index-bound audit before performance testing. It is a larger
memory opportunity than transform metadata and does not change mathematics.

The unavoidable full-coupled delta versus BF16 deterministic route scratch is
2 bytes x M x topk x H: 256 MiB/rank at M=4096 and 2 GiB/rank at M=32768.
If exact buffer shrinking cannot close the long-prefill footprint, prefill must
be chunked at a fixed graph-visible token count; 4096 is the first concrete
candidate, not a preapproved value.

## Acceptance boundary

No prefill implementation is accepted merely because it compiles. Required
device gates are:

1. bit-exact input E4M3 and UE8M0/32 closure for M64 plus 1/63-row tails;
2. FC1 intermediate bit closure against the target capped-SiLU CPU reference;
3. FC2 route-output closure before reduction;
4. final BF16 output closure versus pseudoquant within the predeclared CI;
5. five-run determinism with CUDA graph parity;
6. prefill throughput and peak-memory comparison on the same FP8-compatible
   path, reporting the transform and reducer costs separately.

Until those gates pass, the truthful status is: decode source implemented and
CPU/static-validated; prefill mapped but still fail-closed.

CPU/static receipt:

```text
PYTHONPATH=. pytest -q tests/test_p8*.py
701 passed in 25.33s
python3 -m py_compile runtime_patch/p8_coupled_prefill_plan.py
git diff --check
```
