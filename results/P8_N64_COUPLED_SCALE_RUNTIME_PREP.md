# P8 N64 coupled-scale runtime preparation

Date: 2026-09-05
Branch: `p8-coupled-scale-kernel-v1`
Scope: CPU/static implementation only; no GPU run, image build, service action,
KLD measurement, or throughput measurement was performed.

## Result

The five-role FP16 scale component now has a fail-closed sidecar schema,
byte-hash validation, TP4 shape validation, signed-scale support, a contiguous
runtime carrier, exact Luke-order CPU reference operations, and named CuTe
scale hooks at the required native P8 seams.  The K4 trellis stream and its
E4M3/UE8M0-per-32 physical MMA operands are unchanged, and the implementation
adds no GEMM.

This is **not a runnable full coupled path yet**.  The current N64 FC1 phase
assigns the two halves of an H128 output block to independent CTAs with no grid
barrier.  Applying `gate_svh`/`up_svh` directly to their raw accumulators would
put the scales before, rather than after, H128.  The wrapper consequently
rejects a validated scale-component sidecar before compile/launch until the
H512/H128/sign implementation supplies a correct cross-half owner.  Reverting
that arm to N128 ownership, or adding a materialize-and-transform phase with a
legal synchronization boundary, are the two correct integration choices.

## Frozen component ABI

Schema: `glm53-p8-coupled-scale-component-tp4-rank.v1`

Packed FP16 carrier:

```text
gate_up_suh[H]
| intermediate_scales[E,3I] = gate_svh | up_svh | down_suh
| down_svh[H]
```

For GLM-5.3 TP4, `H=4096`, `E=288`, and TP-local `I=512`.
Every tensor has a required SHA-256 in metadata.  Zero and non-finite values,
wrong shapes/dtypes/rank/layer, positive-only declarations, missing hashes,
or `full_coupled=true` fail validation.  The component declares its composition
target as `coupled-h512-h128-suh-svh-v1`; it does not claim to implement it.

## Exact cast/order contract

The CPU reference and CuTe seam helpers freeze:

```text
input:       FP16(FP32(x) * FP32(gate_up_suh)) -> H128 -> E4M3 amax/quant
FC1 output:  FP16 physical output -> H128 -> FP32 * gate_svh/up_svh
down input:  activation -> FP16(FP32 * FP32(down_suh)) -> H128 -> E4M3 amax/quant
down output: FP16 physical output -> H128 -> FP32 * down_svh -> route sum
```

Scales remain signed.  H512 and procedural signs are separate transform
components and must execute at the audited non-commuting locations.

## Validation

Command:

```text
PYTHONPATH=. pytest -q tests/test_p8_coupled_scale_runtime.py tests/test_p8_narrow_fc1.py tests/test_p8_smallm_scheduler.py tests/test_p8_k5_preparation.py
```

Result: `27 passed in 0.89s`.

The new tests cover signed-value admission, exact shapes and packed offsets,
physical byte hashes, zero/non-finite rejection, false full-coupled rejection,
the multiply-then-FP16-then-H128 cast order, source syntax, named seam helpers,
and rejection before `_compile`.

Broader static/CPU regression command:

```text
PYTHONPATH=. pytest -q tests/test_p8*.py
```

Result: `678 passed in 24.85s`.

## Changed files

- `runtime_patch/p8_coupled_scales.py`
- `runtime_patch/p8_native_kernel.py`
- `runtime_patch/b12x_h16/b12x/moe/_shared/kernels/dynamic.py`
- `runtime_patch/b12x_h16/b12x/moe/_shared/kernels/p8_narrow_fc1.py`
- `runtime_patch/b12x_h16/b12x/moe/_shared/kernels/p8_small_m.py`
- `tests/test_p8_coupled_scale_runtime.py`
- `results/P8_N64_COUPLED_SCALE_RUNTIME_PREP.md`
