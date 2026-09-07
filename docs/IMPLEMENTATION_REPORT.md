# TrellisMX reproducibility implementation report

Date: 2026-09-07

Source baseline: `6c7664a96809c2f4af1fedf6c26b7152e47fac0a`

Task branch: `trellismx-package-v1`

## Executive result

The private TrellisMX research implementation is now an installable, CPU-testable
toolkit with:

- a `trellismx` distribution, import package, typed API, and CLI;
- TrellisMX-first README and citation metadata;
- the primary TrellisMX coupled P8 encoder contract;
- K3/K4/K5 protocol validation and CPU reference decoding;
- portable model-independent P8 tensor containers;
- full-coupled TP4 sidecar validation;
- GLM checkpoint config/index inspection without tensor reads;
- role-separated calibration and future evaluation schemas;
- a future Local Inference Lab QAD-aligned evaluation reference with explicit
  non-use provenance;
- a generic downstream architecture-adapter contract;
- full checkpoint manifest planning;
- resumable, atomic encoding orchestration state;
- portable P4 matrix codec and fixture;
- a separate source-only SM120 runtime overlay wheel;
- CPU-only CI and qualification documentation.

The existing TrellisMX KLD results come from the CF32 conditional-fit
forced-decode campaign. QAD was **not** used for those measurements.

## Primary encoder

The primary encoder is the TrellisMX coupled P8 encoder used by the sealed GLM
campaign:

- layer encoder: `glm53_nvfp4/quantize_p8_coupled_rate_layer.py`;
- coupled H512/H128/sign/scale transform:
  `glm53_nvfp4/p8_coupled_scale.py`;
- K3/K4/K5 tensor codec: `glm53_nvfp4/trellis_mxf.py`;
- routed captures: `glm53_nvfp4/capture.py`;
- route-weighted Hessians: `glm53_nvfp4/output_aware.py`.

It uses GPTQ-style full-Hessian feedback, procedural MCG alpha2, E4M3 operands,
UE8M0/K32 scales, and no LDLQ or BlockLDLQ. It is CUDA-only research source.
The package names this as the product encoder and keeps execution disabled by
default. An alternate trellis-search implementation may be an optional plugin
detail, but KQuant/QSRT is not the required TrellisMX encoder identity.

## Calibration and evaluation roles

`trellismx.roles.v1` requires globally disjoint:

- fit;
- conditional-fit;
- selection;
- confirmation;
- final.

The loader also accepts the repository's legacy sealed role manifest and CF32
development manifest. It rejects duplicate token/input identities across roles.

The preferred future evaluation reference is
`glm-5.3-flash-kimi-k3-source-fidelity-1024x-max2048-v1`, derived from
`festr2/kimi-k3-distribution-fidelity-1024x2048-v1` at pinned revision
`402919ae70d61396087571b63fe9185d95491afb`:

| Partition | Contexts | Scored positions |
|---|---:|---:|
| analysis | 768 | 1,571,435 |
| qualification | 256 | 524,020 |

The contract requires source clusters not to cross partitions, full-vocabulary
forward KL, natural and exact-BF16-route modes, and frozen analysis decisions
before qualification. It records:

```text
used_for_existing_trellismx_measurements = false
```

## Generic architecture support

`trellismx.architecture-adapter.v1` is the generic extension contract. A
downstream implementer declares:

- checkpoint inspection;
- named weight-family mapping;
- parallelism and sharding;
- calibration and Hessian interfaces;
- TrellisMX coupled encoder use;
- portable P8 output;
- E4M3 `mxf8f6f4` runtime capabilities;
- K3/K4/K5 rate-keyed dispatch;
- fail-closed unsupported-shape behavior;
- downstream-owned device gates.

GLM-5.3-Flash is the bundled qualified reference adapter. Additional
architectures do not require this repository to run their GPUs; their implementer
owns and records adapter qualification. A valid contract alone is not a passed
device gate.

## Checkpoint and encoding orchestration

The full manifest planner covers:

- 42 GLM routed layers;
- four TP ranks per layer;
- 168 expected sidecar files;
- exact per-layer rate assignments;
- primary encoder contract;
- runtime capability requirements.

The resumable encoding plan contains 130 tasks:

- checkpoint inspection and provenance validation;
- per-layer encode, pack, and verify dependencies;
- final manifest;
- separately authorized future evaluation.

State files use atomic replacement and an integrity hash. Completed tasks
require a 64-digit receipt hash. Dependencies cannot start before their
predecessors complete. No orchestration command executes a device task.

## Current measured KLD snapshot

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
85 passed in 1.19s
```

Wider CPU/static research and package suite:

```text
138 passed, 3 skipped in 3.08s
```

The three skipped tests are CUDA encoder closures; no CUDA device was exposed.

Installed-wheel smoke tests:

- main wheel SHA-256:
  `7b7dbca60b21b6ce8ac69644edae1b7b11b935fb2f3ffa6a11802249675991a8`
- SM120 source-only overlay SHA-256:
  `34208b6b42e2014e7ba04bfa5837cbed95c932799c756b3e8cc63b773f017514`
- primary encoder contract: TrellisMX coupled P8;
- QAD prior-use claim: false;
- expected checkpoint sidecars: 168/168;
- encoding tasks: 130;
- SM120 source hashes: 15/15 passed;
- CUDA used: false.

## Explicitly not implemented or claimed

- No CLI execution of the real CUDA encoder.
- No CPU Viterbi/Hessian encoder for real BF16 weights.
- No automatic arbitrary-model conversion; downstream adapters are explicit.
- No substitute of synthetic fixture quality for model quality.
- No matched stock-NVFP4 forced-decode KLD arm.
- No protected-final or independent reproduction claim.
- No native NVFP4/FP4 compute claim for P8.
- No production-serving readiness claim.

## Remaining implementation work

1. Add a no-launch launcher adapter around the existing TrellisMX layer encoder
   for separately authorized workstation execution.
2. Convert portable per-tensor outputs into the full GLM sidecar packing flow.
3. Connect orchestration receipts to the existing encoder/packer/verifier CLIs.
4. Add a QAD evaluation runner contract that consumes the future pinned token
   suite without claiming prior TrellisMX use.
5. Keep device closure, full-model KLD, and performance gates separate and
   downstream-authorized.
