# P8 coupled three-layer CF32 runtime/capture preflight

Date: 2026-09-05  
Scope: CPU/read-only runtime inspection plus preparation code. No GPU, image
build, serving container, production unit, teacher tensor, or large artifact
was opened or changed.

## Result

The exact matched runtime path is recoverable, but it is not execution-ready
yet.

All three arms must use one new **numerical-capture-only child image** of the
immutable coupled v9 image
`sha256:ad6b26bf6d1f265d99b09383485ddef82a4acfaea43e28af46341ebb41da24e3`.
The child installs only the already-frozen forced-decode sampler/warmup patches
and capture package. This is necessary because coupled v9 contains the native
P8/Tail-V2 runtime but not the forced-decode capture installation. It also
prevents the old `/runtime-patch` bind from masking v9 with stale all-42 N64
sources.

The stock, identity, and coupled arms then use the same image, stock NVFP4
carrier, TP4/DCP1/no-EP/no-MTP/graphs, B12X_MLA_SPARSE attention,
NVFP4-DS-MLA KV, and CF32 forced-decode protocol. Only these P8 settings differ:

| arm | selected P8 layers | sidecars | transform | exact kernel mode |
|---|---|---|---|---|
| stock | none | none | none | stock ModelOpt NVFP4 |
| identity P8 | 3,20,22 | uniform identity P8 | none | K4, E4M3, UE8M0/32, N128 |
| coupled P8 | 3,20,22 | V3 coupled P8 | draw0 transform | same MMA/stream plus full coupled H512/H128/suh/svh |

Every other layer remains stock because `sitecustomize.py` only intercepts the
explicit `GLM53_P8_NATIVE_LAYERS=3,20,22` set. The coupled launch must use
`GLM53_P8_FC1_TILE_N=128` and `GLM53_P8_FUSED_SCRATCH=`. The earlier all-42
product recipe's N64/fused-scratch values are not this candidate.

## Added fail-closed preparation path

- `runtime_patch/p8_decode_capture/Dockerfile.coupled-v9-control` composes the
  frozen capture patch directly on coupled v9 and validates v9 manifest,
  sampler, warmup, and patched output hashes.
- `glm53_nvfp4/p8_coupled_three_layer_runtime.py` validates all 12 sidecars per
  P8 arm, headers, design/transform identity, postwrite closure, capture-image
  receipt, exact arm environments, and the 12 layer/rank ready+forward log
  inventory.
- `scripts/prepare_p8_coupled_three_layer_cf32_runtime.py` authenticates CF32
  teacher bytes, the stock carrier config/index, the source launch recipe, and
  user-scope production-off state before writing a sealed launch manifest. It
  explicitly removes the stale `/runtime-patch` bind. It never starts or
  restores production and does not call `tail_v2_product_runner._campaign`.
- Coupled sidecars currently live under per-layer directories. The launch
  mounts that tree read-only and creates a container-local flat symlink farm;
  no sidecar copy or extra model storage is required.

The runtime evidence gate requires exactly one `WEIGHTS_READY` and one
`NATIVE_FORWARD` record for each Cartesian pair
`{3,20,22} x {rank0,rank1,rank2,rank3}`, with `boundary=identity` for the
identity arm or `boundary=coupled-h512-h128-suh-svh-v1 full_coupled=true` for
the coupled arm. Missing pairs and identity fallback fail the arm.

## Current blockers, observed rather than inferred

1. The coupled capture child has not been built, so no complete
   `glm53.p8-coupled-cf32-capture-image.v1` receipt exists. The unbuilt
   Dockerfile is preparation, not image qualification.
2. Coupled layer 3 is complete: four chunks, four TP4 sidecars, source-exact
   postwrite receipt, and real-loader result `decision=pass`.
3. Coupled layer 20 currently has two of four chunks, no sidecars, no postwrite
   receipt, and no loader closure.
4. Coupled layer 22 currently has no chunks, sidecars, postwrite receipt, or
   loader closure.
5. The execution seal still needs the future capture image receipt plus fresh
   stock-arm runtime logs proving the actual stock MoE/activation selection.
   The proposal explicitly left those runtime pins unresolved; this preflight
   does not manufacture them.
6. Teacher bytes are intentionally not rehashed by this inspection. The
   manifest builder performs that check before it can emit a seal.

At inspection time, the real user units `klc-backend.service` and
`klc-model-stack.timer` were both inactive and port 8000 was unbound. This is
state evidence only; the builder rechecks the **user** systemd scope and never
contains restoration logic.

## Validation

`PYTHONPATH=$PWD pytest -q tests/test_p8_coupled_three_layer_runtime.py`
passed 7 tests. The tests cover missing layer/rank, coupled-to-identity fallback,
wrong/missing runtime log pairs, stock arm explicit disablement, N128 coupled
settings, and capture parent-image drift. Python compilation and `git diff
--check` also passed.

No KLD, speed, full-model, or execution result is claimed by this work.
