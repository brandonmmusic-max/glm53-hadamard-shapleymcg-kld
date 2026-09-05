# Uniform native P8: all-42-layer developmental KLD

Status: demonstrated on the already-opened conditional-fit panel, not protected
qualification or independent reproduction. Completed 2026-09-05 00:14 EDT.

Mean teacher KLD is **0.04002139481676188**, with window-bootstrap BCa 95% CI
**[0.03377816974746322, 0.04825257361678302]**. All 32 windows and 65,504
ordered causal positions passed the completion audit. Every layer 3–44 on
every TP4 rank reported both weight readiness and native forward dispatch:
168/168 pairs for each event. No LDLQ or BlockLDLQ was used.

## What was measured

- Uniform K4 procedural MCG alpha2, identity boundary, E4M3 reconstruction
  and physical UE8M0 scales per 32 weights on all 42 routed layers.
- 304,405,807,104 routed weights; 161,715,585,024 payload bytes, exactly
  4.25 payload bpw. This is not whole-model bpw including unchanged tensors,
  file headers, or resident-memory overhead.
- Native `mxf8f6f4` execution at TP4, EP disabled, DCP1, eager mode, no MTP,
  FLASHINFER_MLA_SPARSE_SM120 attention and FP8 MLA KV. Four RTX PRO 6000 GPUs.
- Domain-balanced conditional-fit n=32: eight windows in each of four domains,
  2,047 predictions per window. One checkpoint and one full-model evaluation;
  windows are sampling units, not independent quantization-pipeline repeats.
- REAP fit-only calibration, GPTQ-style feedback, static in-group ordering,
  jointly refitted scales. The 28 protected confirmation logits were not opened.

P8 is a quality product: `mxf8f6f4` has **twice the MMA issue count of NVFP4**
for equal K work. This is not the P4/NVFP4 speed product, not “one native FP4
matmul,” and not proof that decode overhead is hidden or serving is faster.

## Contextual comparison, not a matched product win

The predeclared historical decoded-GPTQ comparator measured 0.05333141760014658
on the same 32 windows. P8 is 24.957189% lower, wins 28/32 windows, and has
paired delta -0.013310022783384704 with BCa 95% CI
[-0.02010226777603074, -0.007578923405310998]. The frozen bootstrap uses
50,000 window replicates and seed 20260965.

| Domain (8 windows each) | P8 KLD | Context KLD | Relative decrease |
|---|---:|---:|---:|
| General | 0.033219 | 0.050669 | 34.44% |
| Legal | 0.057224 | 0.079681 | 28.18% |
| Code/agentic | 0.039926 | 0.050025 | 20.19% |
| Reasoning/termination | 0.029717 | 0.032950 | 9.81% |

The comparator is not equal-rate, does not transform the same all-layer set,
and uses TP4/EP4/DCP4 rather than TP4/no-EP/DCP1. Runtime topology, execution
path, and quantization coverage remain rival explanations. Therefore the
supported statement is “24.96% lower than this historical contextual row,”
**not** “the codec improves matched NVFP4 KLD by 24.96%.” This run also does
not prove whole-model pseudoquant/kernel equivalence, Shapley benefit,
protected generalization, or five-run bitwise determinism.

## Speed gate outcome

The original speed comparison failed before its first measured request. The
EXL3 baseline rejected `checkpoint=2, runtime=4` in `create_weights`.
The failed attempt is retained in `speed-v1-failed/`; neither arm has a speed
result. Allocation remains stopped under the preregistered incomplete-arm
rule. Repair requires a disclosed compatibility amendment and a fresh run,
not relabeling this trial or silently changing TP. The P8 KLD result is intact.

Source diagnosis: the stored experts are unsliced; the reader synthesizes the
TP2 restriction. The existing four-GPU EXL3 service uses its explicit EP path.
[Speed plan v2](../experiments/p8-uniform-all42-tp4-vs-exl3-speed-v2.json)
therefore preregistered P8 TP4/no-EP/DCP1 versus EXL3 TP4/EP4/DCP4,
target-only in both, retaining the same five-run thresholds. V2a3 completed
all ten cold runs: P8 gained **11.00%** on median 32K prefill but lost
**77.33%** on median C1 32K decode. The two-metric gate failed and allocation
stopped. This is explicitly a different-topology product comparison; it does
not repair v1 or identify codec causality. See the
[complete speed report](P8_SPEED_V2A3.md). The
[speed-attempt ledger](P8_SPEED_ATTEMPTS.md) preserves every prior failure.

## Reproduction and receipts

Evidence lives in [the exact-copy and projected-data snapshot](../evidence/opened/codec-v2/uniform-p8-all42-v1/).
Its `snapshot.json` binds every copied source hash. Original Parquet hashes
are preserved; the CSV projections retain all causal KLD values but omit
token IDs and logits. The original records and teacher inputs remain local.

- [Checkpoint manifest](../evidence/opened/codec-v2/uniform-p8-all42-v1/manifest.json),
  SHA-256 `9521dae70165d5ba930ca447a37fef97801a6aa8638aef2417a4b206bdc8f191`.
- [Execution receipt](../evidence/opened/codec-v2/uniform-p8-all42-v1/fullmodel-kld-execution.json)
  binds the original sealed plan, both amendments, runtime, image and roles.
- [Frozen analysis](../evidence/opened/codec-v2/uniform-p8-all42-v1/fullmodel-kld-vs-decoded-gptq-context.json)
  includes all window values, domain means, paired intervals and source hashes.
- [Completion audit](../evidence/opened/codec-v2/uniform-p8-all42-v1/fullmodel-completion-audit.json)
  and `server-final.log` prove complete native dispatch and aligned records.

The executed follow-on worktree is commit
`7840c980c40c7ca9ab89af911eafc15a9baa821d` and runtime worktree is
`efd250b002e8da5a0e603253e0cd07ba6249c7b4`;
image ID is `sha256:5da4ef3e814a71c6bcc47a7eb409a02fe4e3d5d867261f0b8e2e9b2d6ebb8ef8`.
The local launch and analysis commands are preserved in
`scripts/run_uniform_p8_all42_kld_followon.sh` and `scripts/run_kld_v3.sh`.
They deliberately refuse to overwrite completed results. Reproduction needs
the recorded local model/teacher inputs and a newly declared run destination.

Run `python3 scripts/snapshot_uniform_p8_fullmodel.py` to reconstruct this
snapshot from those local receipts; then `python3 scripts/verify_public_evidence.py`
checks the repository evidence hashes. Existing ExLlamaV3, KQuant, QSRT and
w4a8_trellis attribution remains in force; this report adds no new codebook port.

Snapshot validation: two idempotent runs copied 52 exact artifacts and exported
32 causal-KLD CSV files. Independent CSV aggregation reproduced
`0.040021394816761875` from all 65,504 values; the numerical difference from
the frozen analyzer is floating-point summation order, not a changed metric.
