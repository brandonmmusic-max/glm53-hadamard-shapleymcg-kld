# TrellisMX

TrellisMX is the name of this GLM-5.3-Flash quantization research implementation
and its reproducible CPU-side toolkit. It stores procedural MCG trellis streams
and decodes them to Tensor Core operands. The research path currently called
**P8** decodes to E4M3 and consumes `mxf8f6f4` MMAs with UE8M0/K32 scales.

This repository is private and evidence-bounded. It contains research code,
sealed campaign records, and a small installable CPU package. It is **not yet a
general production converter**, and no command in the new package launches CUDA
or a model server.

## Current snapshot

| Area | Current state | Boundary |
|---|---|---|
| Product identity | **TrellisMX** | ShapleyMCG is attribution/allocation lineage, not the codec name |
| Current measured endpoint | TrellisMX P8 K3/K4/K5 routed-MoE sidecars | Research implementation, not arbitrary-model conversion |
| Latest KLD | Upgrades-only K4/K5, `0.0341811459` true-decode mean | Already-opened CF32 development evidence |
| Within-endpoint result | 7.5% below uniform coupled K4; interval excludes zero | Not protected final qualification |
| Stock NVFP4 context | `0.005607` below a same-window contextual arm | Different KV/execution/scoring path; no matched claim |
| EXL3 4.0 bpw | EXL3 remains lower at `0.031300` | Separate runtime comparator, not stock NVFP4 |
| Speed | Five-cold-run system gates passed versus EXL3 | Different topology; not codec-only causality |
| Portable package | P4 matrix codec, CLI fixture, and focused CPU tests | P8 model conversion is intentionally not exposed |
| GPU qualification | Prepared/unexecuted by this package | No CUDA, serving, or model job is launched |

### Terminology

| Term | Meaning | Do not infer |
|---|---|---|
| TrellisMX P8 | MCG trellis decoded to E4M3 with UE8M0/K32 scales and `mxf8f6f4` MMA | Native NVFP4 compute |
| K3/K4/K5 | Stored branch rate used by a P8 layer | Whole-model bpw |
| Stock NVFP4 carrier | Comparator weight carrier | Same execution/scoring path |
| `nvfp4_ds_mla` | KV-cache dtype | The routed-weight codec |
| EXL3 | Separate serving checkpoint/runtime comparator | Stock NVFP4 or codec-only control |
| ShapleyMCG | Attribution/allocation method and license lineage | Layer damages do not sum to model KLD |
| P4 | Separate native-E2M1 research endpoint | Qualified P8 quality or speed |
| CF32 | Conditional-fit 32-window role | Protected final holdout |
| True-decode KLD | `KL(teacher‖student)` over forced rows, excluding row 0 | Prefill-scored KLD |
| C1 | Concurrency-one decode | Aggregate server throughput |

## Current KLD status

The most recent completed measurements are in
[`results/P8_FULL_COUPLED_CAMPAIGN_RESULTS_20260906.md`](results/P8_FULL_COUPLED_CAMPAIGN_RESULTS_20260906.md).
All TrellisMX arms below used the sealed 32-window conditional-fit protocol,
the same teacher, `nvfp4_ds_mla` KV, the same attention backend, TP4/DCP1/no
EP, CUDA graphs, and forced decoding. Each window has 2,047 prediction rows;
row 0 is the one-token prefill row and is excluded, leaving 2,046 true-decode
rows per window and 65,472 rows across all 32 windows. KLD means
`KL(teacher || student)`, computed in FP64.

| Arm | Stored rate | True-decode KLD | Paired result vs uniform K4 |
|---|---:|---:|---|
| Uniform coupled TrellisMX K4 | 4.2540 bpw | 0.0369674524 | reference |
| Same-size K3/K4/K5 allocation | 4.2302 bpw | 0.0504843312 | +0.0135168788 (+36.6%); 1/32 wins |
| Upgrades-only K4/K5 | 4.6587 bpw | **0.0341811459** | −0.0027863065 (−7.5%); 25/32 wins |

The upgrades-only result is the best current TrellisMX measurement. Its paired
95% percentile interval versus uniform K4 is `[−0.004723, −0.000961]`, so this
particular development comparison excludes zero. It is not a protected final
qualification claim.

### Stock NVFP4 comparison

There is **no stock-carrier measurement on this exact forced-decode
`nvfp4_ds_mla` serving path**. The pilot's stock arm failed before scoring, and
the planned stock-only serving run was canceled. Do not treat the table below as
a matched A/B result.

The only available stock run used the same 32 windows and teacher but a
different path: stock NVFP4, FP8 KV, DCP4, eager execution, and scoring from a
single 2,048-token prefill pass rather than 2,047 forced-decode rows.

| Arm | Path | Mean KLD | Difference from contextual stock |
|---|---|---:|---:|
| Stock NVFP4 | FP8 KV, DCP4, prefill-scored | 0.039789 | reference |
| Uniform coupled TrellisMX K4 | NVFP4 KV, DCP1, forced decode | 0.0369674524 | −0.002821; 12/32 worse than stock |
| Upgrades-only TrellisMX K4/K5 | NVFP4 KV, DCP1, forced decode | 0.0341811459 | −0.005607; 6/32 worse than stock |

The contextual comparison looks favorable, but it changes carrier, KV dtype,
topology, execution mode, and scoring path. A true stock NVFP4 comparison still
requires the planned no-sidecar arm on the same `nvfp4_ds_mla` forced-decode
harness. That GPU measurement is **not run by this CPU package**.

EXL3 4.0 bpw is a separate comparator, not stock NVFP4. On the same sealed
windows, teacher, KV cache, and attention backend, EXL3 measured `0.031300`
mean KLD; it remains ahead of the upgrades-only TrellisMX arm at 4.6587 stored
routed-sidecar bpw. See the full report's Phase 1 section for exact intervals.

## Current serving-speed context

The passing five-cold-run system comparison is:

| Metric | TrellisMX P8 | EXL3 | Result |
|---|---:|---:|---|
| C1 32K decode median | 100.5501 tok/s | 91.2200 tok/s | +10.23% |
| Server-validated 32K prefill median | 6,959 tok/s | 6,239 tok/s | +11.54% |

This is a serving-system comparison: TrellisMX P8 used TP4/no-EP/DCP1 while
EXL3 used TP4/EP4/DCP4. It does not isolate codec causality or establish
protected numerical quality or production superiority. See the
[five-cold-run report](results/P8_FC1_COLD_COMPARISON_V1.md).

## What is not established

- No matched stock-NVFP4 KLD arm on the latest forced-decode/`nvfp4_ds_mla` path.
- No protected-final or independent reproduction claim.
- No arbitrary-model conversion support.
- No native P4/NVFP4-speed-class quality or speed claim.
- No codec-only causality from the EXL3 system-speed comparison.
- No claim that Shapley routed-output damage is full-model KLD.
- No production-serving readiness claim.
- No interchangeability between the old eager/FP8-KV KLD and latest forced-decode KLD.

## What is installed and tested

The distribution is named `trellismx`; the import package is also `trellismx`.
Historical `glm53_nvfp4` and `bmxfp4` modules remain importable for sealed
receipts and research code.

The portable CPU surface includes:

- typed P8 K3/K4/K5 protocol validation;
- P8 trellis stream packing/unpacking, state reconstruction, E4M3 state table,
  UE8M0 scales, and dense reference decode;
- a portable model-independent safetensors container for multiple named
  K3/K4/K5 P8 tensors;
- a production full-coupled TP4 sidecar validator and tiny structural fixture;
- GLM-5.3-Flash config/index inspection with zero safetensors payload reads;
- provenance overlays and a disabled-by-default external encoder-backend
  descriptor;
- a separate source-only SM120 mixed-rate runtime overlay package;
- a versioned P4 matrix interchange:

- procedural MCG K4 trellis stream;
- E2M1 operands with positive E4M3/K16 scales;
- deterministic packing and random-access decoding;
- safe safetensors loading with metadata, dtype, shape, offset, and hash checks;
- exact storage accounting;
- a deterministic synthetic fixture and independent scalar oracle.
- a separate deterministic safetensors fixture covering K3, K4, and K5 P8
  reference decoding.

These commands deliberately do **not** claim that a synthetic fixture is a full
model, that CPU floating-point accumulation is byte-identical to Tensor Core
accumulation, or that the current research encoder can convert arbitrary
checkpoints.

Install into an isolated environment and run the CPU fixture:

```bash
python3 -m venv /tmp/trellismx-venv
/tmp/trellismx-venv/bin/pip install -e .
rm -rf /tmp/trellismx-fixture
CUDA_VISIBLE_DEVICES= /tmp/trellismx-venv/bin/trellismx fixture /tmp/trellismx-fixture --write
CUDA_VISIBLE_DEVICES= /tmp/trellismx-venv/bin/trellismx fixture /tmp/trellismx-fixture
CUDA_VISIBLE_DEVICES= /tmp/trellismx-venv/bin/trellismx validate /tmp/trellismx-fixture/fixture.p4.safetensors --expected-shape 32x128
CUDA_VISIBLE_DEVICES= /tmp/trellismx-venv/bin/trellismx p8-fixture /tmp/trellismx-p8-fixture --write
CUDA_VISIBLE_DEVICES= /tmp/trellismx-venv/bin/trellismx p8-fixture /tmp/trellismx-p8-fixture
CUDA_VISIBLE_DEVICES= /tmp/trellismx-venv/bin/trellismx p8-tensor-fixture /tmp/trellismx-p8-tensors --write
CUDA_VISIBLE_DEVICES= /tmp/trellismx-venv/bin/trellismx validate-p8-tensors /tmp/trellismx-p8-tensors/p8-tensors.safetensors
CUDA_VISIBLE_DEVICES= /tmp/trellismx-venv/bin/trellismx sidecar-fixture /tmp/trellismx-sidecar --write
```

Architecture planning and portable workflow summaries are also CPU-only:

```bash
CUDA_VISIBLE_DEVICES= trellismx plan --architecture glm53-flash --uniform-rate 4
CUDA_VISIBLE_DEVICES= trellismx workflow \
  --config configs/glm53-flash-coupled-p8-upgrades-k4k5-v1.json
CUDA_VISIBLE_DEVICES= trellismx inspect-checkpoint \
  --architecture glm53-flash --checkpoint /path/to/GLM-5.3-Flash-BF16
CUDA_VISIBLE_DEVICES= trellismx validate-sidecar \
  /path/to/p8-layer-003-tp4-rank-0.safetensors \
  --expected-layer 3 --expected-rank 0 --expected-bits 4
CUDA_VISIBLE_DEVICES= trellismx runtime-info
```

`CUDA_VISIBLE_DEVICES=` is a defensive CPU boundary for environments with a
CUDA-enabled PyTorch build. The fixture itself uses NumPy and does not issue
CUDA calls.

## TrellisMX family boundaries

- **P8** decodes stored trellis bits to E4M3 and performs E4M3×E4M3
  `mxf8f6f4` compute with UE8M0/K32 scales. For equivalent K it has twice the
  MMA issue count of native NVFP4. It is not native FP4 compute and must not be
  advertised as native FP4 speed.
- Current measured P8 stored rates are K4 uniform and mixed K3/K4/K5. The
  K3/K5 runtime is measured only in the sealed research path; the new CLI does
  not expose a general P8 model converter.
- **P4** is a separate native E2M1 research codec. Its portable matrix format
  is independently specified and CPU-tested. It is not the P8 production path
  and does not inherit P8 measurements.
- File bpw and GPU residency are different quantities. Reports distinguish
  payload rate, metadata, headers, total file bytes, and resident memory.

## Documentation map

- [Codec and checkpoint contracts](docs/TRELLISMX_FORMAT.md)
- [Toolkit workflow and encoder-backend boundary](docs/TRELLISMX_WORKFLOW.md)
- [Implementation status and validation receipts](docs/IMPLEMENTATION_REPORT.md)
- [CPU and GPU qualification gates](docs/QUALIFICATION.md)
- [Reproducibility guide](docs/REPRODUCIBILITY.md)
- [Full historical research ledger](docs/RESEARCH_LEDGER.md)
- [Third-party notices and attribution](THIRD_PARTY_NOTICES.md)
- [License](LICENSE)

The historical ledger is intentionally separate from this README. It preserves
campaign chronology and older measurements; this page is the concise entry point.

## Support boundary

The supported architecture is GLM-5.3-Flash with the exact shapes and runtime
contracts in the sealed P8 campaign. Other architectures are rejected. This is
not yet a ModelOpt-style arbitrary-model converter.

The path to that broader goal is explicit:

1. separate a model-independent P8 K3/K4/K5 tensor codec from GLM campaign I/O;
2. define typed architecture adapters and weight-mapping contracts;
3. implement GLM-5.3-Flash first and reject every unsupported architecture;
4. add deterministic calibration and evaluation schemas;
5. expose resumable, atomic encoding with provenance;
6. qualify one additional architecture only after the GLM adapter is portable.

Real-model encoding and serving require separately authorized artifacts,
calibration roles, and GPU qualification. No GPU gate is silently executed.

The SM120 runtime is separately packaged as source-only metadata and pinned
source files:

```bash
python3 -m pip wheel --no-deps ./runtime_overlay
python3 -m pip install trellismx_runtime_sm120-0.1.0-py3-none-any.whl
CUDA_VISIBLE_DEVICES= trellismx runtime-info
```

Installing that wheel does not import CUDA, B12X, CUTLASS, or vLLM, and does
not make the runtime executable. Device execution still requires the separate
immutable image and explicit authorization.
