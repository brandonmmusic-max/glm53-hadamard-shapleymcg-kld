# NVFP4 MLA scale-counter preparation

Status: **CPU/static prepared, not built or run.** No GPU, image build, service,
model, or existing result was touched.

The counter now targets the exact active static writer input: 512 values after
the calibrated per-layer outer-scale division and any DCP gather, immediately
before the unchanged B12X 288-byte NoPE writer. It preserves the writer call
signature and does not write the latent or cache. It refuses 304-, 368-, 432-
or 656-byte records.

Prepared artifacts:

- `runtime_patch/nvfp4_mla_scale_counter.py`: opt-in observer, request control,
  per-layer/rank GPU counters and atomic receipts.
- `glm53_nvfp4/nvfp4_mla_counter_control.py`: atomic reset/dump-reset/dump and
  all-rank receipt-wait CLI.
- `runtime_patch/nvfp4_mla_scale_counter/Dockerfile`: fail-closed derivation
  from the validated tail-V2 image digest; it verifies the active MLA wrapper,
  B12X backend and writer source hashes without modifying those files.
- `experiments/nvfp4-mla-scale-counter-cf32-v1.json`: prereg-ready canary,
  CF32 collection, exclusions, decision rule and claim boundary.

The mandatory first GPU result is an instrumented-versus-uninstrumented
bitwise exact-logit canary on a fit-role window. Only after it passes may the
counter collect the fixed, domain-balanced 32 conditional-fit requests. Any
refitted scale bank is a later encoder/runtime calibration candidate and must
be assessed by a separate counter-disabled KLD A/B.

Every future numerical row must carry attention backend, KV dtype/ABI, MoE
backend, activation precision, stored bpw, topology, graph/speculation status,
checkpoint/image identity and scale-bank hash. Instrumented speed is
inadmissible by construction.

