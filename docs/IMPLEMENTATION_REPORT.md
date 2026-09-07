# TrellisMX reproducibility implementation report

Date: 2026-09-07

Source baseline: `6c7664a96809c2f4af1fedf6c26b7152e47fac0a`

Task branch: `trellismx-package-v1`

## Executive result

This branch turns the TrellisMX research implementation into an installable,
CPU-testable package without silently weakening the real encoder/runtime
boundary. It now has:

- a `trellismx` distribution, import package, typed API, and CLI;
- TrellisMX-first README and citation metadata;
- P8 K3/K4/K5 typed protocol validation;
- CPU P8 edge packing/unpacking, state reconstruction, E4M3 state table,
  UE8M0 scales, dense reference decode, and an all-rate synthetic fixture;
- a portable model-independent safetensors container for named K3/K4/K5 P8
  tensors;
- a production full-coupled TP4 sidecar validator;
- GLM checkpoint config/index inspection with zero tensor-payload reads;
- portable P4 matrix codec and deterministic fixture;
- a fail-closed GLM-5.3-Flash architecture adapter;
- portable uniform-K4 and upgrades-only K4/K5 workflow manifests;
- provenance overlays and a disabled-by-default encoder-backend descriptor;
- a workflow stage map bound to the committed research source;
- a separate source-only SM120 mixed-rate runtime overlay wheel;
- CPU-only CI and qualification documentation;
- explicit external-encoder and GPU-boundary reporting.

This is not yet an arbitrary-model production converter. Real BF16-to-P8
encoding still requires the CUDA Viterbi/Hessian path and the separately pinned
KQuant/QSRT backend whose redistribution terms are unresolved.

## Package surface

### TrellisMX P8 reference codec

Source: `trellismx/protocol.py`, backed by committed research implementations
in `glm53_nvfp4/trellis_mxf.py` and `glm53_nvfp4/trellis_nvfp4.py`.

Supported CPU operations:

- K3/K4/K5 edge packing and unpacking;
- cyclic state reconstruction;
- 65,536-state procedural-MCG alpha-2 table;
- UE8M0/K32 pack/unpack;
- dense pseudoquant reference decode;
- tensor shape/Hessian request validation.

The real encoder wrapper exists but fails closed unless:

1. the caller explicitly authorizes device execution, and
2. the tensor is on CUDA, and
3. the external KQuant/QSRT Viterbi backend is supplied by the operator.

No ordinary CLI command invokes that encoder.

### Portable P8 tensor container

Source: `trellismx/p8_format.py`

Schema `trellismx.p8-tensors.v1` stores named logical matrices as paired
trellis and UE8M0/K32 tensors. The loader enforces exact inventory, metadata,
dtypes, shapes, tensor hashes, and optional whole-file hash. It reconstructs
the deterministic procedural codebook instead of storing duplicate tables.

### Full-coupled TP4 sidecar validator

Source: `trellismx/sidecar.py`

The validator accepts the current full-coupled schema at K3/K4/K5 and checks:

- layer/rank/world-size/rate identity;
- E4M3, procedural MCG alpha2, UE8M0/K32, and no-LDLQ contracts;
- complete H512/H128/suh/svh/sign metadata;
- exact eight-tensor inventory, shapes, and dtypes;
- per-tensor SHA-256 values;
- regenerated coupled-sign identity;
- production `K + 0.25` payload accounting.

It explicitly reports runtime loader closure as **not tested** and imports no
CUDA/B12X/CUTLASS/vLLM dependency.

### Architecture adapter

Source: `trellismx/adapters.py`

The only qualified adapter is GLM-5.3-Flash:

- routed layers 3–44;
- 288 experts;
- hidden size 4096;
- global intermediate size 2048;
- TP4 local intermediate 512;
- boundary `coupled-h512-h128-suh-svh-v1`;
- gate/up/down projections;
- P8 rates K3/K4/K5.

Unknown architectures are rejected before checkpoint I/O.

The GLM adapter also inspects `config.json` and
`model.safetensors.index.json`, validates all 42 × 288 × 3 routed expert
weight names, checks relative shard paths, and records config/index hashes.
It reads zero safetensors payload bytes.

### Portable workflows

Source: `trellismx/workflow.py` and `configs/`

- `glm53-flash-coupled-p8-uniform-k4-v1.json`: all 42 layers K4.
- `glm53-flash-coupled-p8-upgrades-k4k5-v1.json`: 25 K4 and 17 K5 layers,
  matching the measured upgrades-only assignment.

Portable configs omit private artifact URIs. A workstation provenance overlay
must pin the source checkpoint, exact scale source, calibration capture, and
role manifest before any authorized execution.

### Provenance and encoder backend

Source: `trellismx/provenance.py`

Provenance overlays exactly match workflow input inventory and record each
artifact kind, URI, SHA-256, and optional byte count. The package does not read
artifact bytes or store credentials.

`configs/kquant-qsrt-encoder-backend.disabled.json` declares the only real
encoder backend, KQuant/QSRT Viterbi, as disabled, unpinned, unlicensed for
redistribution, and unauthorized. Enabling it requires an operator-pinned URI,
source hash, resolved license status, and separate execution authorization.

### Runtime source map

The reusable stage sources remain explicit:

- checkpoint indexing: `glm53_nvfp4/shard_index.py`;
- calibration: `glm53_nvfp4/capture.py`;
- Hessians: `glm53_nvfp4/output_aware.py`;
- GLM coupled transform: `glm53_nvfp4/p8_coupled_scale.py`;
- tensor encoder: `glm53_nvfp4/trellis_mxf.py`;
- layer encoder: `glm53_nvfp4/quantize_p8_coupled_rate_layer.py`;
- sidecar packer: `glm53_nvfp4/build_p8_coupled_rate_tp4_sidecars.py`;
- postwrite verifier: `glm53_nvfp4/verify_p8_coupled_rate_tp4_sidecars.py`;
- manifest/ledger: `glm53_nvfp4/full_coupled_build_support.py`;
- mixed-rate runtime: `runtime_patch/p8_mixed_rate_image/p8_native_kernel.py`;
- full-model executor: `glm53_nvfp4/p8_full_coupled_cf32_executor.py`.

### SM120 source-only overlay

`runtime_overlay/` builds the separate distribution
`trellismx-runtime-sm120`. Its wheel contains the v11 manifest and all 15
hash-pinned runtime source files. It imports no runtime dependency, is not
executable, and still requires the separate immutable image for device work.
`trellismx runtime-info` verifies the overlay by SHA-256 without using CUDA.

## Current measured KLD snapshot

The source campaign's latest completed developmental measurements are:

| Arm | Stored bpw | True-decode KLD |
|---|---:|---:|
| Uniform coupled K4 | 4.253979859528719 | 0.036967452439595275 |
| Same-size K3/K4/K5 allocation | 4.230170335719194 | 0.05048433119198257 |
| Upgrades-only K4/K5 | 4.658741764290623 | 0.03418114591027796 |

The stock NVFP4 value `0.03978855156037106` is contextual only: it used FP8
KV, DCP4, eager execution, and prefill scoring rather than the current
`nvfp4_ds_mla` forced-decode path. No matched stock arm has been run.

## Validation receipts

Focused package suite:

```text
72 passed in 1.23s
```

Wider CPU/static research and package suite:

```text
125 passed, 3 skipped in 3.12s
```

The three skipped tests are CUDA encoder closures; no CUDA was exposed to the
process.

Final installed-wheel smoke tests:

- main wheel: `trellismx-0.1.0-py3-none-any.whl`
- main SHA-256:
  `887f22d35bb2dd6018ee76e6867d2c346e79b507f326c244560330fb9693dc09`
- SM120 overlay wheel:
  `trellismx_runtime_sm120-0.1.0-py3-none-any.whl`
- SM120 overlay SHA-256:
  `7e9321b112e0bfc8eeb8dab9458d529a70318a19736525bdd5c89a01a7388a3d`
- K3/K4/K5 portable tensor fixture: passed
- K5 full-coupled synthetic sidecar: passed
- SM120 overlay: 15/15 source hashes passed
- device execution: false

## Explicitly not implemented or claimed

- No arbitrary-model adapter.
- No CPU Viterbi/Hessian encoder for real BF16 weights.
- No redistribution of the external KQuant/QSRT encoder backend.
- No standalone portable full-model P8 checkpoint writer; the new portable
  container currently covers named tensors, not full model architecture and
  runtime composition.
- No ordinary-install vLLM/B12X/CUTLASS runtime.
- No matched stock-NVFP4 forced-decode KLD arm.
- No protected-final or independent reproduction claim.
- No native NVFP4/FP4 compute claim for P8.
- No speed claim from CPU package operations.

## Smallest next milestones

1. Integrate an explicitly pinned KQuant/QSRT backend plugin without changing
   its default-disabled state.
2. Add typed calibration and evaluation-role schemas to the portable workflow.
3. Extend the model-independent tensor container into a full architecture
   checkpoint manifest and runtime composition.
4. Add resumable, atomic real-model encoding orchestration.
5. Run separately authorized device closure and KLD gates on any new runtime.
6. Qualify one additional architecture end-to-end before advertising generic
   ModelOpt-style model support.
