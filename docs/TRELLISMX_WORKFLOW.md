# TrellisMX toolkit workflow

TrellisMX follows the workflow category of a quantization toolkit: inspect,
validate, plan, calibrate, encode, export, verify, integrate, and qualify. It
does not copy or claim compatibility with ModelOpt. Every model family still
requires a qualified architecture adapter and an authorized execution backend.

## TrellisMX encoder boundary

The primary encoder is the TrellisMX coupled P8 encoder used by the sealed GLM
campaign: coupled H512/H128/sign/scale transform, route-weighted Hessians,
GPTQ-style full-Hessian feedback, and K3/K4/K5 procedural MCG trellis encoding.
It is CUDA-only research source. Ordinary package installation does not enable
real-model encoding; execution remains disabled until an operator pins the exact
workstation source/provenance and explicitly authorizes device execution. Any
alternate trellis-search implementation is an optional plugin detail, not the
product encoder identity.

CPU package operations remain honest and useful:

- typed P8 K3/K4/K5 protocol validation;
- architecture planning;
- stream packing/reference-decode components;
- deterministic K3/K4/K5 CPU reference fixture;
- portable model-independent P8 tensor containers;
- full-coupled TP4 sidecar validation;
- checkpoint config/index inspection;
- provenance overlays;
- portable P4 byte codec;
- sidecar verifier source mapping;
- workflow/provenance manifests;
- future QAD evaluation-reference validation with explicit non-use provenance.

## Portable GLM workflow configs

```bash
CUDA_VISIBLE_DEVICES= trellismx workflow \
  --config configs/glm53-flash-coupled-p8-uniform-k4-v1.json

CUDA_VISIBLE_DEVICES= trellismx workflow \
  --config configs/glm53-flash-coupled-p8-upgrades-k4k5-v1.json
```

These commands parse and summarize configs only. They do not inspect a
checkpoint, read calibration data, encode tensors, invoke CUDA, or start a
runtime.

The all-rate CPU fixture can be written and replayed without a model:

```bash
CUDA_VISIBLE_DEVICES= trellismx p8-fixture /tmp/trellismx-p8-fixture --write
CUDA_VISIBLE_DEVICES= trellismx p8-fixture /tmp/trellismx-p8-fixture
```

The upgrades-only config records the exact measured rate assignment: 25 layers
at K4 and 17 layers at K5. Artifact URIs remain null in portable configs; a
workstation-specific provenance overlay must pin source checkpoint, scale
source, capture, and role manifest before authorized execution.

A provenance overlay is a separate JSON object whose artifact inventory exactly
matches the workflow. The package validates names, kinds, URIs, and SHA-256
fields, but does not read artifact bytes or store credentials:

```bash
CUDA_VISIBLE_DEVICES= trellismx workflow \
  --config configs/glm53-flash-coupled-p8-upgrades-k4k5-v1.json \
  --provenance /path/to/provenance.json \
  --encoder-backend configs/trellismx-coupled-encoder.cuda-disabled.json \
  --evaluation-reference configs/glm53-flash-qad-future-evaluation-v1.json
```

The checked-in TrellisMX encoder descriptor is intentionally disabled. Enabling
it requires an operator-pinned source URI/hash and separate execution
authorization.

## Calibration and future evaluation roles

Current TrellisMX measurements use the sealed CF32 conditional-fit
forced-decode role campaign. The package also validates the five-role
`fit`, `conditional-fit`, `selection`, `confirmation`, and `final` contract and
loads the repository's legacy sealed role manifest.

The preferred future evaluation reference is the Local Inference Lab QAD-aligned
GLM token suite:

- suite `glm-5.3-flash-kimi-k3-source-fidelity-1024x-max2048-v1`;
- analysis: 768 contexts / 1,571,435 positions;
- qualification: 256 contexts / 524,020 positions;
- source clusters cannot cross partitions;
- full-vocabulary forward KL under natural and exact-BF16-route modes;
- analysis decisions frozen before qualification.

This QAD reference is future-only. It was not used for the existing TrellisMX
KLD results and must not be presented as their provenance.

## Stage-to-source map

| Stage | Research source | Package status | Device |
|---|---|---|---|
| Inspect indexed checkpoint | `glm53_nvfp4/shard_index.py` | source available | CPU |
| Load calibration captures | `glm53_nvfp4/capture.py` | source available | CPU reader |
| Fit route-weighted Hessians | `glm53_nvfp4/output_aware.py` | research source | authorized device |
| Apply coupled GLM transform/scales | `glm53_nvfp4/p8_coupled_scale.py` | GLM-only | authorized device |
| Encode P8 tensor | `trellismx/protocol.py`, `glm53_nvfp4/trellis_mxf.py` | typed wrapper; external CUDA backend required | authorized device |
| Pack TP sidecars | `glm53_nvfp4/build_p8_coupled_rate_tp4_sidecars.py` | source available | CPU |
| Verify sidecars/postwrite closure | `glm53_nvfp4/verify_p8_coupled_rate_tp4_sidecars.py` | source available | authorized device |
| Build manifest/storage ledger | `glm53_nvfp4/full_coupled_build_support.py` | source available | CPU |
| Runtime integration | `runtime_patch/p8_mixed_rate_image/p8_native_kernel.py`, `glm53_nvfp4/p8_full_coupled_runtime.py` | GLM research runtime | authorized device |
| Device closure and full-model KLD | `scripts/`, `glm53_nvfp4/p8_full_coupled_cf32_executor.py` | not run by package | authorized device |

## SM120 source-only overlay

`runtime_overlay/` builds a separate `trellismx-runtime-sm120` wheel. It
contains the v11 manifest and the 15 hash-pinned source files, but imports no
CUDA/B12X/CUTLASS/vLLM dependency and is not executable by installation.
`trellismx runtime-info` hash-checks the source-only overlay and reports:

- `executable: false`;
- `dependencies_imported: false`;
- `cuda_used: false`;
- `requires_separate_pinned_image: true`.

## Required extraction for additional models

1. move checkpoint indexing and tensor naming behind architecture adapters;
2. make calibration capture and Hessian construction adapter-specific;
3. make the coupled transform optional rather than a hidden GLM assumption;
4. persist model-independent P8 tensor payloads in a portable format;
5. generate runtime contracts from capability descriptors;
6. execute the downstream-owned device closure and full-model KLD gates.

Unsupported architectures must fail before checkpoint I/O rather than falling
back to an untested GLM-style mapping.

## Generic adapter contract

`configs/architecture-adapter-contract.template.json` defines the downstream
implementation surface:

```bash
CUDA_VISIBLE_DEVICES= trellismx validate-adapter-contract \
  configs/architecture-adapter-contract.template.json
```

The contract requires exact checkpoint inspection, named weight-family mapping,
role-separated calibration, the TrellisMX coupled encoder, portable P8 output,
E4M3 `mxf8f6f4` runtime capabilities, rate-keyed dispatch, fail-closed shape
handling, and downstream-owned device gates. TrellisMX does not claim those
gates for an adapter that has not run them.
