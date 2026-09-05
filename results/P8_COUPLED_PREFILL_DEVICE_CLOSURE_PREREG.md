# P8 coupled M64/N128 prefill device-closure preregistration

Status: **CPU harness prepared; no GPU execution performed**.  
Protocol SHA-256: `7c6db9db4728ed6c6e626c976f8bfe77a9729a8cc48a4c04f98fcf734a5c0fb8`.

The executable protocol is frozen in
`scripts/run_p8_coupled_prefill_device_closure.py`. Its default invocation
prints the protocol and performs no CUDA or Docker action. Future device work
requires explicit `--execute`, an immutable image ID, an idle SM120 device, a
fresh output directory, and exact sidecar/design/transform/runtime-manifest
hashes.

## Decision question

For the existing synthetic layer-3/rank-0 full-rank sidecar, does an actual
`P8NativeTPMoE` call at M>1 select the coupled split-materialized M64/N128
wrapper path and close its observable activation carriers and numerical outputs
for small, boundary, and partial-tail geometries?

Primary hypothesis: actual M>1 wrapper dispatch reaches `small_m=false`,
`materialized=true`, M64/N128, preserves every supplied route through grouped
compaction, reproduces the CPU E4M3/UE8M0 carrier bytes, and satisfies the
unchanged M1 route/final numerical thresholds.

Credible rivals frozen before execution:

- the wrapper or selector silently enters M1, monolithic, identity, or an
  inherited phase path;
- grouped histogram/prefix/scatter state loses, duplicates, or associates a
  route with the wrong expert;
- the complete M64 tile works but the one-row tail is omitted or reads stale
  carrier bytes;
- carriers close exactly while the M64 FP32/FP16 boundary or coupled reducer is
  numerically wrong;
- source/image ABI compiles structurally but fails on SM120.

Any such observation falsifies this closure. There is no fallback acceptance
and no post-result threshold relaxation.

## Frozen matrix and experimental hierarchy

The device matrix is M in `{2,64,65}` at real E288/H4096/local-I512/top-k8.
Every token routes once to each of experts
`{0,1,17,63,127,191,255,287}`:

- M2: smallest supported M>1 and two-row partial tiles;
- M64: exact M64 boundary for all eight active experts;
- M65: one complete M64 tile plus a one-row tail for all eight active experts.

Inputs alternate between two seeded BF16 prototypes. This keeps the independent
CPU reference bounded: eight experts are decoded once, and only two unique
input rows are evaluated before reference results are expanded back into all
M×8 routes. The real wrapper still executes every physical route and output at
full shape. Per-token router weights are independently seeded and remain fixed
across repeats.

The prepared experimental unit is one image/sidecar/runtime construction. The
three M cases are within-process conditions; the five same-input launches per
case are determinism repeats, not five independent experiments. This is a
developmental synthetic device-closure role, not fit, selection, confirmation,
or final quality evidence. No teacher logits or protected role are opened.

## Actual-wrapper and schedule gates

Class construction alone cannot pass. For each of the 15 future calls, the
harness invokes the actual `P8NativeTPMoE` wrapper and requires exactly:

```text
small_m=false
materialized=true
fused_scratch_zero=false
fc1_tile_n=128
tile_m=64
```

The device's `row_counts` must equal a CPU bincount of the supplied route IDs.
`expert_tile_base` must be a monotone prefix whose span for each expert is
exactly `ceil(row_count/64)`. Every active physical row's `token_map` entry must
be in range, point to a route for the same expert, and form a bijection over all
M×8 routes. Carrier comparisons are reordered through this observed mapping;
they do not assume atomic compaction order.

## Exact carriers versus numerical outputs

The following gates are deliberately distinct.

Exact, bitwise CPU/device gates:

1. all input E4M3 bytes after the frozen within-K32 permutation;
2. all input UE8M0/32 bytes;
3. all routed post-FC1/down-input E4M3 bytes in route order;
4. all routed post-FC1/down-input UE8M0/32 bytes in route order;
5. source, fixture, design, transform, and runtime-manifest identities;
6. schedule counts/prefix/mapping invariants.

Numerical CPU/device gates:

- route-output cosine strictly greater than `0.995` and relative L2 strictly
  less than `0.12`;
- final-output cosine strictly greater than `0.995` and relative L2 strictly
  less than `0.12`.

These equal the existing M1 protocol at
`9556885c32385574530acf7b07c1d5636b46852ca9714c27883f98f17368efe1`;
the prefill harness does not weaken M1. Route and final outputs are numerical,
not claimed bit-exact to the CPU reference. Separately, all four observable
carrier hashes plus route-output and final-output bits must be identical across
the five same-input device repeats for each M case.

The CPU reference preserves the existing physical path:

```text
BF16 -> FP16 -> H512 -> signed suh -> H128
     -> E4M3/UE8M0-K32
     -> procedural-MCG E4M3*UE8M0 FC1 -> FP16
     -> coupled H128/svh/sign/capped-SiLU/sign/suh/H128
     -> E4M3/UE8M0-K32
     -> procedural-MCG E4M3*UE8M0 FC2 -> FP16
     -> H128 -> shared down_svh
     -> ordered top-k FP32 reduction -> H512 -> BF16
```

## Identity, storage, and failure policy

The harness reuses the streamed full-rank synthetic fixture produced by
`scripts/generate_p8_coupled_m1_synthetic_fixture.py`; it does not generate or
copy another 0.963 GB sidecar. The sidecar is mounted read-only. The maximum
conservative wrapper scratch forecast is 123,834,624 bytes at M65, below the
frozen 130 MiB bound. New evidence is capped at 64 MiB.

The installed runtime manifest must name and hash the wrapper, scale code,
schedule, dynamic kernel, H128 FC1, small-M FC2 base, coupled top-k reducer,
coupled-prefill planner, and both M64 prefill kernels. Actual imported module
paths are hashed; a matching file elsewhere in the image cannot substitute.

Failure receipts preserve exception type, message, traceback, stdout, stderr,
launch command, and exit status. A missing carrier, fallback dispatch, mapping
error, nonfinite value, numerical miss, hash drift, over-budget evidence, or
device failure is a failed closure.

## Future opt-in command shape

```bash
python3 scripts/run_p8_coupled_prefill_device_closure.py --execute \
  --image sha256:<PIN_COUPLED_IMAGE_ID> \
  --gpu-device <IDLE_SM120_GPU> \
  --sidecar /absolute/p8-synthetic-layer-003-tp4-rank-0.safetensors \
  --sidecar-sha256 <64-hex> \
  --design /absolute/synthetic-design.json --design-sha256 <64-hex> \
  --transform /absolute/p8-coupled-transform-draw0-silu10-v1.json \
  --transform-sha256 093d219b18ba32471adcee746442b1481c7ba5659bbea62b94f1a665d4343a12 \
  --runtime-manifest-sha256 <64-hex> \
  --output /absolute/fresh/p8-prefill-device-closure-v1
```

The outer process validates image and manifest identities with a CPU-only
network-disabled container, requires the selected GPU to be idle and at most
75 C, and launches one network-disabled container without touching services.

## Preparation receipts

- Runner SHA-256:
  `6375e79dbd03efdc36b52092e453d3db64aa532546f22102c62f8dbdac00938b`.
- CPU tests SHA-256:
  `517ff29620e48887f1aac58afb361085c87e0ba7a49984e1de52d906a3e53b32`.
- Reused M1 helper SHA-256:
  `a98b6f5069919b4da700f1e7e46949ac49f804a178913e6931f0c444f9384430`.
- Reused fixture generator SHA-256:
  `c41986025866ccd2fbb2874c36e351e86d54fb88a517bb6222727c839c5c48c8`.
- Transform SHA-256:
  `093d219b18ba32471adcee746442b1481c7ba5659bbea62b94f1a665d4343a12`.
- `python3 -m py_compile` for the new runner and tests: pass.
- Combined new/M1/fixture/prefill-plan/launch-ABI CPU tests: **44 passed**.
- GPU execution, image build, service mutation, model KLD, CUDA-graph parity,
  prefill throughput, and all-rank/full-model closure: **Not tested**.

This preparation can establish reproducibility of the frozen harness only.
Successful future execution would be a synthetic GPU smoke/numerical closure,
not production prefill qualification or independent reproduction.
