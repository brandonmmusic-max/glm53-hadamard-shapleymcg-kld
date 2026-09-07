# TrellisMX toolkit workflow

TrellisMX follows the workflow category of a quantization toolkit: inspect,
validate, plan, calibrate, encode, export, verify, integrate, and qualify. It
does not copy or claim compatibility with ModelOpt. Every model family still
requires a qualified architecture adapter and an authorized execution backend.

## Real encoder backend boundary

The proven P8 Viterbi/Hessian encoder is CUDA-only and delegates its trellis
search to an external KQuant/QSRT encoder snapshot. That snapshot is not
licensed for redistribution here. Ordinary package installation therefore does
not enable real-model encoding. The API fails closed unless the operator
supplies a separately pinned backend and explicitly authorizes device
execution.

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
- workflow/provenance manifests.

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
  --encoder-backend configs/kquant-qsrt-encoder-backend.disabled.json
```

The checked-in encoder-backend descriptor is intentionally disabled. Enabling
it requires an operator-pinned backend URI, source hash, and resolved license
status; even then, execution needs a separate explicit authorization.

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
6. qualify one new architecture end-to-end before advertising generic support.

Unsupported architectures must fail before checkpoint I/O rather than falling
back to an untested GLM-style mapping.
