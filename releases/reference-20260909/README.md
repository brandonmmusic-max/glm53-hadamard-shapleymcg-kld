---
license: other
license_name: shapleymcg
license_link: LICENSE
pipeline_tag: text-generation
base_model:
  - zai-org/GLM-5.3-Flash-BF16
tags:
  - trellismx
  - glm
  - mixture-of-experts
  - quantization
  - vllm
  - sm120
  - custom-runtime
  - shapleymcg
---

# GLM-5.3-Flash TrellisMX-MXFP8

A compressed GLM-5.3-Flash checkpoint whose routed experts reconstruct directly
into **FP8 Tensor Core operands**. The September 9 reference stack is the selected
runtime: **222.100 tokens/s C1 decode at 8K** and **8,457 tokens/s prefill at 32K**
in separate `llm_decode_bench` runs on four RTX PRO 6000 Blackwell GPUs at 300W each.

## Recommended reference stack — September 9, 2026

[Docker Hub](https://hub.docker.com/r/verdictai/trellismx) image:

```text
verdictai/trellismx:glm53-flash-p8-r27-reference-20260909
```

Immutable image reference:

```text
verdictai/trellismx@sha256:ca6b80188dce154b91f49108b7d87792d2ba6328935afc71b44d1c0e6f6a1adf
```

Use the supplied [Docker Compose configuration](compose.yaml) and
[serving script](serve.sh). The [reference runtime recipe](runtime-reference-20260909/Dockerfile)
and [source manifest](runtime-reference-20260909/source-manifest.json) record the
inference overlays. The checkpoint has not been re-encoded for this update.

| Setting | Selected reference |
| --- | --- |
| GPUs / measured power | 4 × RTX PRO 6000 Blackwell 96GB (SM120), 300W per GPU |
| Parallelism | TP4 / DCP4 |
| Routed-expert math | E4M3 FP8, native UE8M0 scales per 32 elements |
| MLA KV cache | NVFP4 (`nvfp4_ds_mla`) |
| Speculative decoding | Probabilistic MTP3, standard rejection |
| Execution | CUDA graphs, B12X MLA attention and PCIe collectives |
| Scheduler | 24 sequence slots, 4,096-token batch |
| Memory / maximum request length | 0.97 GPU memory utilization / 1,000,000 tokens |

**FP8 math and NVFP4 KV cache describe different parts of the model.** Compressed
K4/K5 expert weights decode to FP8 operands for multiplication. NVFP4 stores the
MLA attention cache. Choosing FP8 KV instead changes the cache representation;
it does not turn this into an FP4-math checkpoint. The P4 design is separate.

Weight upload is complete: the remote inventory and all **168 sidecar SHA-256
identities** were verified. Local serving and measurements use the same checkpoint.
A separate clean download-to-GPU test of the HF directory layout has not been run;
see [release status](release-status.json).

## Measured speed and KV capacity

Rates below are **tokens/s from `llm_decode_bench`**. C1 is one request;
C4 and C8 are aggregate throughput across concurrent requests, not per user.
Context labels are nominal benchmark cells; raw logs retain actual prompt sizes.

| Context | Prefill | C1 decode | C4 decode, aggregate | C8 decode, aggregate |
| --- | ---: | ---: | ---: | ---: |
| 0K | — | 204.611 | 327.670 | — |
| 8K | — | 222.100 | 318.560 | 594.639 |
| 16K | — | 216.636 | 319.157 | 602.965 |
| 32K | 8,457 | 204.930 | 329.343 | — |
| 64K | 8,443 | 199.446 | 330.096 | — |
| 128K | 8,323 | 204.445 | 334.086 | — |

This table combines the short-context screen with the separate expanded-context
pass. The earlier comparison screen measured **8,407 tokens/s prefill at both
32K and 64K**. All runs are retained rather than selecting one figure as a repeated-run average.
The reference was selected for balanced use: it had the highest observed C1
throughput at 0K and 8K among the three finalists. Task-count performed better
at 16K C4 (**353.532 tokens/s**), but neither alternative was measured at C1
32K–128K. Reference is not established as the fastest candidate for every workload.

The engine reported **23,562,091 aggregate effective KV tokens** in the expanded
reference run; separate short-context sessions reported **23,568,627**. This is
allocated capacity across requests, not a tested single-request context length
or a full-capacity stress result. The configured request limit is 1,000,000 tokens.
Use the engine's split-cache metric, not the benchmark client's generic
block-count-times-DCP estimate.

These are exploratory measurements, with one server preparation per configuration.
Later screens used a minimum 90-second idle period and required all GPUs at or below
55°C for 30 continuous seconds before each cell; earlier runs retain their original
protocols. MTP acceptance, generated output and clocks can vary. No independently
repeated speed qualification or long-context accuracy claim is implied.

[All September 9 results, candidate comparison and protocol](results/speed-20260909/README.md)
include the full benchmark index, JSON, logs, negative results and diagnostic runs.
Profiled diagnostics are identified separately from serving speed measurements.

## Reference-stack KLD: matched FP8 and NVFP4 MLA cache

Lower KLD means the student's next-token distribution is closer to the BF16
teacher on the measured inputs. It is not a direct score for general answer quality.

| Cache on the September 9 reference | Mean true-decode KLD | Status |
| --- | ---: | --- |
| FP8 KV | **0.0319451732** | 32/32; receipt audit passed |
| NVFP4 MLA KV | **0.0354562238** | 32/32; receipt audit passed |

FP8 has lower observed KLD in **22/32 windows**. The paired mean difference
(FP8 minus NVFP4) is **−0.0035110506**, with a window BCa 95% interval
[−0.0081718772, −0.0013675941]. [Full results and audit](results/kld-reference-20260909/README.md).

Both arms use the **same 32 previously opened conditional-fit development windows**
as the prior measurement, with the same token arrays and BF16 teacher. Each window
contains 2,048 input tokens and 2,047 prediction rows; row zero is excluded, leaving
**2,046 true-decode rows per window**. Scores use CPU FP64 KL(teacher || student)
and the equal mean of the 32 window means.

The matched setup is TP4/DCP4, **MTP off**, one sequence, a 4,096-token batch and
zero prefix-cache hits. FP8 runs first, then NVFP4. The logits-capture image is
`sha256:0405a1c0dc128b51069798a5d00b346257bbd006c0deb7e6530a3d005d75de71`;
it adds the capture/warmup seam to the selected serving image and is not the public
serving tag. This quality measurement therefore does not measure the MTP3 speed
configuration's complete output behavior.

The paired result and confidence interval passed the receipt/score audit. These
already-opened windows provide a development comparison, **not untouched final qualification**.
No matched claim against EXL3, TR3 4bpw or another weight format is established by
this cache comparison.

The archived historical `.0341811459` cache label is under provenance review:
the prior card labels it NVFP4, while the owner's historical account identifies
FP8. The prior text and linked receipts are retained below without silently
changing either number. Do not use that disputed historical label as a matched
cache comparison. New reference results are recorded separately above.

## Matched-window comparison with TR3 4bpw

| Cache | TrellisMX reference KLD | TR3 / EXL3 4bpw KLD |
| --- | ---: | ---: |
| FP8 KV | 0.0319451732 | 0.0281899278 |
| NVFP4 MLA KV | 0.0354562238 | 0.0304785381 |

Lower is better on this panel. These runs use the same BF16 teacher, 32 previously
opened conditional-fit windows, exact token histories and 2,046 true-decode rows
per window. Both use MTP off, one sequence and CPU FP64 KL(teacher || student).
The TR3 and EXL3 Hub names identify the same uniform 4bpw checkpoint; this is not
the stock NVFP4-weight model. FP8 and NVFP4 in the table label the MLA KV cache.

This is a matched-data **system comparison**. TR3 uses its compatible r10
TP4/EP4/DCP4 runtime; TrellisMX uses r27 TP4/DCP4 with EP off. Expert math,
nonrouted-weight policy and cache layout also differ. The result does not isolate
the codec or establish general answer quality. One server per cache mode, FP8
then NVFP4; window intervals do not measure server-run variability.

[Per-window scores, paired intervals and audit](results/kld-tr3-20260909/README.md).
The first TR3 startup failed before any measured windows because kernel metadata
was unavailable in the shared compiler cache. The retained retry used process-
separated compiler caches on tmpfs; model weights and kernel math were unchanged.

## Run the model

Requires the custom runtime, four supported GPUs and NVIDIA Container Toolkit.
This is a sidecar checkpoint used with a separate carrier model; stock Transformers
or unmodified vLLM cannot load it as a standalone model.

```bash
hf download brandonmusic/GLM-5.3-Flash-TrellisMX-MXFP8 --local-dir ./trellismx-p8
hf download local-inference-lab/GLM-5.3-Flash-NVFP4 \
  --revision 520de24eabf507659eaef7c70f14fd584527facc --local-dir ./glm53-carrier
cd trellismx-p8
export MODEL_ROOT=/absolute/path/to/glm53-carrier
docker compose -f compose.yaml config --quiet
docker compose -f compose.yaml up -d
```

The default endpoint is port 8000. The scripts do not set GPU power limits or
memory clocks. The measured 300W per GPU setting is part of the benchmark setup,
not a guarantee that a new host will reproduce its speed. The 1,000,000-token
launch limit is within the model's declared 1,048,576 positions; it is not a
measured 1M-context accuracy result. API credentials are not baked into the image.

## Required runtime and model layout

This repository contains **168 TP4 routed-expert sidecars (177,269,057,440
bytes)** plus serving metadata, scripts, license and evaluation receipts.
They replace the carrier's routed experts at load time. They are not an
additional expert ensemble, and this is **not a standalone Transformers
checkpoint**. Do not point stock Transformers or unmodified vLLM at it.

The working runtime additionally requires the stock carrier:
[`local-inference-lab/GLM-5.3-Flash-NVFP4`](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4/tree/520de24eabf507659eaef7c70f14fd584527facc),
revision `520de24eabf507659eaef7c70f14fd584527facc`. Its attention, shared
experts, embeddings, output head and MTP components remain part of the served
model. Keeping this exact dependency preserves the existing loader contract;
the reported logical model payload is not a claim that the two download
directories together occupy only that many bytes.

## Checkpoint and implementation details

This is the **17-K5 / 25-K4 coupled checkpoint**, covering all 42 routed layers
(3–44). Routed tensors occupy **4.6587417643 bits/weight including metadata**;
this is not a uniform 4.25 bpw model.

<details>
<summary>How the compressed weights, FP8 math and prior work fit together</summary>

## What TrellisMX does

**TrellisMX separates the compressed weight format from the native format
used for multiplication.** Its trellis encoder chooses weight reconstructions
jointly, rather than independently rounding every weight. The serving kernel
decodes the compressed stream in registers into E4M3 FP8 operands, paired with
native UE8M0 scales per 32 elements, for `mxf8f6f4` Tensor Core MMA.

```text
K4/K5 compressed trellis weights + UE8M0/32 block scales
    → procedural MCG decode in the expert kernel
    → native E4M3 operands and block scales → Tensor Core MMA
```

This brings **trellis/vector weight quantization into a native block-scaled
compute path**: joint encoder decisions, scalar hardware operands. It is
**not VQ compression of the block-scale array itself**. The current format
stores `w13_scale_ue8m0` and `w2_scale_ue8m0` separately from the trellis
streams. The coupled input/output `suh`/`svh` scales are another distinct
part of the transform, not that native block-scale array.

The engineering distinction is the combination of procedural MCG-to-E4M3
decoding, native block-scaled expert MMA, coupled H512/H128 transforms with
input/output scales, and layerwise K4/K5 storage allocation in a measured
GLM serving implementation. The same E4M3 compute alphabet is retained when
the stored trellis rate changes. This release allocates K5 to 17 layers and
K4 to 25; it is not a uniform four-bit model.

### Storage, compute, and tradeoffs

| Format/path | Weight storage | Expert multiplication and tradeoff |
| --- | --- | --- |
| TrellisMX P8, this checkpoint | 4.65874 routed bits/weight including metadata | Compressed streams decode to E4M3 + UE8M0/32; avoids a full FP16/BF16 weight expansion before expert MMA, but pays decoder and transform costs |
| Native MXFP8 weight storage | Nominal 8.25 bits/weight | Stores FP8 codes directly with an 8-bit scale per 32; no trellis decode, larger weight payload |
| NVFP4 weight storage | Nominal 4.5 bits/weight | E2M1 + E4M3 scale per 16; smaller reconstruction alphabet and FP4-rate MMA |
| EXL3 reference W4A16 path | Compressed trellis codes plus recipe-dependent metadata | Reconstructs higher-precision operands; already uses Tensor Cores, but is not this native E4M3 block-scaled expert path |

Nominal native-format figures count payload plus block scales only, not all
model metadata or non-routed tensors. They are not identical accounting
denominators to this checkpoint's measured routed bpw.

Relative to storing native MXFP8 weights, the benefit is a smaller compressed
weight payload while retaining FP8 reconstruction and compute. Relative to
the EXL3 W4A16 reference path, the distinction is decoding directly into
low-precision native MMA operands, not introducing Tensor Cores to EXL3.
Relative to NVFP4, P8 offers a larger reconstruction alphabet, **not FP4
math throughput**: `mxf8f6f4` needs twice the NVFP4 MMA issue count for equal
dimensions. That is not a prediction of exactly twice the wall-clock time.
Fused weight decoding does not make the whole MoE layer one kernel launch,
nor establish that every decoder/transform cost is hidden by computation.

The receipts below establish this checkpoint's KLD and serving speed.
They do **not** establish a matched numerical quality or speed advantage
over EXL3 or stock NVFP4; no such percentage is claimed here.

### How it relates to SQG, KQuant, QSRT, and EXL3

These names describe different levels of the system, not four interchangeable
hardware formats:

- **SQG** is a codebook/decoder-law lineage used in related trellis paths.
  This P8 path selects a procedural **MCG** reconstruction law instead; it
  reuses SQG-derived packing/lane organization rather than claiming that
  organization as a new invention.
- **KQuant** is quantization/integration tooling supporting multiple recipes.
  TrellisMX names this particular codec/runtime/checkpoint combination, not
  a demonstrated replacement for every KQuant recipe or model converter.
- **QSRT** supplies related trellis and coupled-transform lineage. Native
  W4A8 reconstruction and coupled Hadamards already appear in the public
  [QSRT/B12X integration](https://github.com/local-inference-lab/vllm/pull/566).
  TrellisMX builds on that lineage with the MCG/E4M3 block-scaled P8 path and
  this mixed-rate GLM deployment; native reconstruction alone is not a new
  priority claim. The runtime bundle includes a port of the SQG-XOR-Cheb-T12
  decoder and reused `w4a8_trellis` pipeline/lane machinery.
- **EXL3/QTIP** establish procedural-codebook and trellis-quantization prior
  art. See the [EXL3 format description](https://github.com/turboderp-org/exllamav3/blob/6ff3a17ea7f3d0026b273d43239398d57f71b788/doc/exl3.md)
  and [QTIP](https://arxiv.org/abs/2406.11235). TrellisMX credits that lineage;
  its distinction here is the specific native reconstruction/runtime and
  measured coupled mixed-rate checkpoint, not inventing trellis VQ.

**No “first publicly available VQ codec to Tensor Cores” claim is made.**
Existing prior art makes that broad wording inappropriate. The distinctive
combination above describes this implementation without asserting an
unverified first-in-history claim or general superiority over its sources.

</details>

## Earlier measurements and provenance

<details>
<summary>September 8, RC5 and DCP1 results — archived runtime settings and receipts</summary>

The text below preserves the prior release's results and links. References to
“current” or “production” inside this archive describe the earlier release,
not the September 9 reference above. In particular, the old 16-sequence recipe
and old image digest are superseded. The historical `.0341811459` cache label
is disputed as described above; retaining the original text is not a resolution.

## September 8 r27 DCP4 results (superseded runtime)

| Measurement | Result | Conditions |
| --- | --- | --- |
| C1 nominal zero-context decode | **202.45 tokens/s** | 20-second streaming window, output cap 512, MTP3 |
| C2 nominal zero-context decode | **264.83 aggregate tokens/s** | Same run; not per-user throughput |
| Cold prefill, actual 8,200 tokens | **7,692 tokens/s** | TTFT 1.066 s, 3 samples |
| Cold prefill, actual 16,227 tokens | **7,940 tokens/s** | TTFT 2.044 s, 3 samples |
| Cold prefill, actual 32,316 tokens | **8,006 tokens/s** | TTFT 4.037 s, 2 samples |
| Engine-reported NVFP4 MLA KV capacity | **23,424,836 aggregate effective tokens** | Split-cache accounting; 4 GPUs, TP4/DCP4 |

The capacity is across requests, not the per-request context limit. The engine
reported 23.42 theoretical million-token slots, but this recipe schedules at
most 16 sequences and permits at most 1,000,000 tokens per request. Hybrid
KDA state has its own cache layout. Capacity is an allocation report, not a
long-context accuracy or successful full-capacity stress result.

These are existing quick observations, not independently repeated qualification.
The measured image and public rebuild have byte-matched inference sources;
the public package bakes in the same production launcher defaults. No new GPU
speed run is implied by publication. See [protocol, full cells, raw JSON/logs,
client snapshot and redaction hashes](results/r27-20260908/README.md). The
benchmark client's larger generic capacity estimate is not valid for this
split-cache layout.

### September 8 r27 KV-cache KLD comparison

| Runtime / measurement image | KV dtype | Attention backend | Mean true-decode KLD | Window BCa 95% interval |
| --- | --- | --- | ---: | --- |
| Historical DCP1 / `c121590d3371…` | NVFP4 (`nvfp4_ds_mla`) | B12X_MLA_SPARSE | **0.0341811459** | [0.0291483518, 0.0409784257] |
| Historical DCP1 / `c121590d3371…` | FP8 (`fp8_ds_mla`) | FLASHINFER_MLA_SPARSE_SM120 | **0.0318077613** | [0.0267419020, 0.0387478446] |
| Current r27 DCP4 / `a7fde8169ec2…` | NVFP4 (`nvfp4_ds_mla`) | B12X | **0.0350078183** | [0.0292403806, 0.0430987171] |
| Current r27 DCP4 / `a7fde8169ec2…` | FP8 (`fp8`) | B12X | **0.0310574767** | [0.0263411936, 0.0374509346] |

**Both historical KV modes were measured.** The earlier card omitted the full
32-window historical FP8 run. Its exact mean is **0.03180776125099182**;
**0.03418114591027796** belongs to the separate NVFP4/B12X run. The recovered
FP8 runtime manifest and startup log identify `fp8_ds_mla`, FlashInfer, TP4/DCP1
and MTP off. Its retained score arrays and hashes were audited; all 32 input
and token hashes and window order match the current panel. Both historical
and current rows use the same true-decode scoring mask.
See the [historical FP8 audit](results/r27-kv-cf32-20260908/historical-fp8-audit.json).

**Historical and current measurements use different images and runtimes:**

- Historical measurement image, local Docker image ID:
  `sha256:c121590d3371b406c1e456076b1fc9cf9b596aa425b4401c692d142cbc425cd4`.
- Current measurement image, local Docker image ID:
  `sha256:a7fde8169ec24fff3d7f1a7a8a50375a90e6c9e8cf71254680a281f54be37ca6`.
  Both current KV arms use this same image, which adds the logits-capture and
  warmup wrapper to the published r27 serving image; it is not the public serving tag.
- Published r27 serving image, registry manifest digest:
  `verdictai/trellismx@sha256:1c8a10d2b21bd6ed5a7ca4a29bcc3900d29acc3ce42e1d722b9ebaa74357de3f`.

The historical-to-current FP8 comparison changes attention backend (FlashInfer
to B12X), image, DCP topology and runtime settings. The lower current score does
**not** establish an accuracy benefit from DCP4 alone. The current NVFP4-versus-FP8
pair holds the measurement image and DCP4 configuration fixed. Historical FP8
confidence intervals use NumPy `default_rng(20260902)`; the current pair uses
legacy `RandomState(20260902)`, each with 20,000 BCa resamples. The recovered
historical mean replays exactly and its interval endpoints within 1e-15.
The original [NVFP4 receipt](evidence/k5-only-cf32-analysis.json) remains unchanged.

FP8 minus NVFP4 is **-0.0039503416**, paired-window BCa 95%
interval [-0.0092510457, -0.0018107907]; FP8 has lower observed KLD in
26/32 windows. Both runs use the same previously opened
conditional-fit windows and BF16 teacher as historical DCP1: 2,048 input tokens,
2,047 prediction rows, and **2,046 true-decode rows per window**. Current runs
use TP4/DCP4, MTP off, one sequence and zero prefix hits. These are development
measurements from one server preparation per arm in fixed order; the intervals
do not measure run-to-run variability, MTP quality or long-context accuracy.
Historical DCP1 differs in runtime and topology and is not a KV-only control.
See [protocol, all 64 window scores and receipt audit](results/r27-kv-cf32-20260908/README.md).


### C16 scaling investigation

On the unchanged server, balanced C16/4K tests measured **377.88 aggregate
tokens/s** with a 512-token output cap and **633.38** with an 8,192-token cap.
Short responses repeatedly introduce prefill; the long-response central windows
had no new prompt tokens. Nominal zero-context, long-response cells measured
**252.60 at C2** and **651.02 at C16**. These are streaming windows, not completed
8,192-token requests or a configuration improvement.

Four subsequent instrumented C2/C16 traces identify the residual costs: larger
target batches switch to grouped TrellisMX FC1/FC2 and two-shot PCIe collectives.
The repeated rank-local kernel spans average 19.41 ms/step at C2 and 65.40 at C16.
All tensor fills together account for about 1.99 ms of summed C16 GPU time per
step, so compacting input buffers alone is not established as a major fix.
Kernel durations overlap and collective time includes synchronization; no
optimization speedup is claimed. Production settings remain the tested recipe.
See the [report, per-rank attribution and uninstrumented cells](results/r27-scaling-20260908/REPORT.md).

Rebuild the inference-only overlay from the included `runtime/Dockerfile`:

```bash
docker build --pull=false -t trellismx-r27-rebuild runtime
```

## Historical RC5 and DCP1 measurements

| Measurement | Result | Boundary |
| --- | --- | --- |
| Mean true-decode KLD | **0.0341811459** | 32 opened CF windows; BF16 teacher to student |
| Window BCa 95% interval | [0.0291483518, 0.0409784257] | Historical checkpoint measurement, not fresh RC5 KLD |
| Uniform coupled K4 KLD | 0.0369674524 | K4/K5 is 7.54% lower at a larger bit budget |
| Estonia | **30/30 PASS** | 30 requests, C10, reasoning effort max |
| LAVD | **28/30 credited** | 22 exact, 6 near, 2 truncated at 100k output tokens |
| Hotel Lights | **28/30 exact** | 2 scored failures, neither token-limited |
| Sustained decode C1 | **178.336464 tokens/s** | Aggregate, MTP3, context zero |
| Sustained decode C2 / C4 | 235.605573 / 300.874345 tokens/s | Aggregate, not per user |
| Client prefill, 8k / 32k cells | 8063 / 7998 tokens/s | Separate prefill run |

All 28 finished LAVD answers were credited by the official scorer, including
six NEAR answers. The two capped answers remain incomplete: **28/28 credited
among finished answers is not 30/30 exact accuracy**. Hotel's two failures
are retained as scored. All three selected reasoning runs had zero request
errors. Their output-token p99 values were 100,000 (LAVD), 8,766.38 (Estonia),
and 97,012.24 (Hotel); these are not p99 KLD or speed values.

Speed comes from final `llm_decode_bench` JSON and matching logs, not vLLM's
rolling throughput. Decode used a separate cache-disabled launch with zero
prefix-hit delta; prefill enabled prefix caching. Four GPUs at 300W each,
+6000 memory offset. One run per configuration: no independently repeated
speed qualification or 200-token/s claim. Reasoning-profile generation
statistics are not substituted for sustained-decode speed.

KLD uses CPU FP64 teacher-to-student KL, 2,047 prediction rows per window (2,046 true-decode rows),
excluding row zero (prefill). It is already-opened development evidence,
not untouched final qualification or a matched stock-NVFP4 comparison.
The historical KLD image differs from RC5. Top-1 agreement and p99 KLD are
not reported by this release; no values are inferred for missing metrics.

See the [full GitHub results and receipts](https://github.com/brandonmmusic-max/glm53-hadamard-shapleymcg-kld/blob/03527b1f092cfbf274ae3ca78cabb23077c37057/results/RC5_RELEASE_RESULTS_20260907.md),
and the included `results/` and `evidence/` files.

</details>

## Codec and limitations

Procedural MCG trellis streams decode in registers to E4M3 for native
`mxf8f6f4` tensor-core MMA. Coupled boundary:
`coupled-h512-h128-suh-svh-v1`, sign draw zero, SiLU cap 10, UE8M0/32 scales.
P8 uses **twice the NVFP4 MMA issue count for the same dimensions**; it is
not native FP4-rate math. The separate P4 design is not this checkpoint.

The current runtime is a custom TrellisMX overlay on digest-pinned Jovian
Judgement GLM r27; it is not a claim that unmodified upstream supports this checkpoint. A general TrellisMX adapter or an
arbitrary-model BF16 converter is not included in this model upload.

## License and attribution

The exact [SHAPLEYMCG LICENSE](LICENSE) applies to this work. It is
source-available, **not an OSI-open-source license**. Base-model and component
licenses remain applicable; this license does not replace them.
Credit: Brandon M. Music; Z.ai for GLM; Local Inference Lab for vLLM/B12X
integration; ExLlamaV3 for procedural trellis lineage; KQuant, QSRT and
`w4a8_trellis` for the coupled/reference and native decoding lineage.
See `CITATION.cff`, `LICENSE.exllamav3` and `LICENSE.carrier`.

[Runtime support](https://github.com/brandonmmusic-max/glm53-hadamard-shapleymcg-kld/issues/5)
is for the custom experimental release; no general hardware/support guarantee
is implied. The original research versions remain available on GitHub.
