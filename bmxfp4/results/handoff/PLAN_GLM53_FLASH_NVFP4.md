# Plan: a better NVFP4 for GLM-5.3-Flash (rotation + calibrated GPTQ + optional multi-tier), 2026-09-02

Goal: an NVFP4 checkpoint of GLM-5.3-Flash that serves on the existing marlin/humming NVFP4 path and lands well
below the stock NVFP4 KLD (0.0492 on the sealed 25-window panel) — projected ~0.035 at the same 188 GB — and, as a
research arm, a multi-tier 5.03 bpw variant projected ~0.027 (EXL3-4bpw runtime = 0.0305).

## 0. What is and is not usable from what is already on disk

| asset | usable as | why |
|---|---|---|
| LibertAIDAI GLM-5.3-Flash-NVFP4 (188 GB, ModelOpt RTN) | baseline + scoring-path validation | its expert weights are already 4-bit; re-quantizing them (rotation or not) can only ADD error vs the teacher. Not a source. |
| brandonmusic GLM-5.3-Flash-EXL3-4bpw (164 GB) | comparator only | 4-bit trellis; same argument. |
| KLC_SANDBOXES/glm53-flash-kld-eval/data/teacher/logits (30 GB) | THE teacher for scoring | FP32 teacher logits of the sealed 25 x 2048 panel; KL(teacher‖student), FP64 log-softmax — same definition as the Qwen3 harness. |
| REAP calibration corpus (reap_recall_calib.jsonl) | calibration windows | re-tokenize with the GLM tokenizer (build_calib.py, one flag). |
| zai-org/GLM-5.3-Flash official (328 GB, 62 shards, FP8 e4m3 block-scaled experts, BF16 attention/embeddings/lm_head/hyper-connections) | THE source | there is no BF16 release of Flash; FP8 is the official numerics and is what the teacher logits were produced from. NOT on disk (HF cache dir is empty). |

So the honest answer to "can I do H16 rotations and Shapley without re-downloading the BF16": the rotation + GPTQ
pass needs the source weights, and for Flash the source is the official FP8 (328 GB), not a BF16. Nothing already
downloaded can replace it. The Shapley attribution is not needed (below) and is not feasible on this rig anyway.

## 1. Decisions carried over from the Qwen3-30B-A3B pilot (measured, not assumed)

1. Drop the Aumann–Shapley attribution. Without the sparse tier the Shapley-calibrated allocation tied the proxy
   allocation (B7n 0.00900 vs B7un 0.00883, CI includes 0), and the attribution needs full-model gradients that
   OOM'd even on the 8xB300 pod for GLM-5.3. Use the per-unit output-residual proxy + exact-byte knapsack.
2. Drop the 4:8 sparse tier. It cost 34% of the KLD under the best allocation and 5x at 4 bpw.
3. Keep: block-16 Hadamard + GPTQ (static in-group act-order) on REAP-corpus Hessians, causal layer-by-layer
   re-encode, MXFP6/FP8/BF16 upgrade tiers for the 5.03 bpw arm, FP8 embeddings (free), BF16 lm_head (FP8 costs +15%).
4. Do not use the full-width Hadamard (W4A4 +34% worse on NVFP4).

## 2. Phase 0 — deployability questions, answered on Qwen3 first (today, ~2 GPU-hours, no download needed)

The runtime NVFP4 kernels compute W·x. A rotated weight W·R needs R^T·x at runtime. There are two ways to get it
without touching the MoE kernel, and Phase 0 measures what each keeps:

- P0.a **In-side-only rotation** (`--rotation had16-in`: rotate gate/up inputs, leave down_proj unrotated).
  Tells how much of the Had16 gain (-15% vs identity on Qwen3) lives on the foldable side.
- P0.b **Fold R16 into the residual stream** (QuaRot-style: embeddings·R, attention/dense in-projections R^T·W,
  out-projections W·R, expert down_proj outputs W·R, lm_head·R; RMSNorm weights folded first). Verify on Qwen3
  that the folded checkpoint scores identically to P0.a. Then export it in ModelOpt NVFP4 format (weight /
  weight_scale / weight_scale_2 / input_scale, gate and up sharing one global scale) and score it THROUGH vLLM
  with the stock marlin path against the same teacher. That is the end-to-end proof that the recipe is a
  checkpoint swap. If P0.a keeps most of the gain and P0.b matches it, GLM gets the same treatment.
- P0.c **Per-layer tiering** (whole layers in MXFP6 instead of per-unit): re-run the knapsack on the existing
  Qwen3 candidates with a per-layer constraint, one causal arm. If it keeps most of the multi-tier gain, the
  5.03 bpw GLM arm becomes a kernel-light target (uniform format per layer); if not, per-unit mixed formats
  need a grouped-GEMM kernel and the multi-tier arm stays research-only.
- P0.d GPTQ-only (identity rotation) is the fallback recipe if the fold fails on GLM; on Qwen3 it is already
  -20% vs stock RTN NVFP4 at equal bytes (B3' 0.01150 vs stock 0.01438 on selection).

Stop rule: if P0.a loses more than half of the rotation gain AND P0.b cannot be folded, plan the GLM run as
GPTQ-only + REAP calibration (still projected ~0.039 vs stock 0.0492).

## 3. Phase 1 — GLM-5.3-Flash source and engine (1–2 days, mostly waiting on the download)

- P1.a Disk: klcstore has 328 GB free; the source needs 328 GB. Free ~120 GB first — candidates, none deleted
  without your say: exl3-recon (113 GB, Qwen3 EXL3 reconstructions), teacher/fit (38 GB, off-plan fit-window
  teacher), Qwen3 weight dumps B7/B7a (110 GB, arms that lost). hessians-calib (116 GB) and B7n/B3p/B8c stay.
- P1.b Download zai-org/GLM-5.3-Flash (328 GB; at the 20–50 MB/s seen this morning that is 2–4.5 h).
- P1.c Engine = transformers 5.16 `glm5_next` (present on main: modeling_glm5_next.py, mHC hyper-connections,
  linear-attention (gated delta-net) + deepseek-sparse-attention layer mix, NoPE MLA, sigmoid noaux_tc router,
  swiglu_limit 10) in a NEW venv (klc-env is 4.52 and cannot load it). Load the full text model resident across
  the 4 GPUs (`device_map`), vision tower skipped: FP8 experts stay FP8 with dequant-on-the-fly (transformers
  FineGrainedFP8), ≈330 GB of 384. A custom `NVFP4Linear` (packed u8 + E4M3 scales, dequant in forward — the
  logic already in stock_nvfp4.py) replaces each quantized expert as it is produced, so memory falls from 1.0
  to 0.56 B/param layer by layer and the pseudo-quant is exact (no FP8 carrier).
- P1.d Port hessian.py's router hook to GLM routing (sigmoid + e_score_correction_bias, top-8, norm_topk_prob,
  routed_scaling_factor 2.5, k-major dispatch order must be re-derived from modeling_glm5_next). Hessians are
  captured per layer in 4 chunks of 72 experts (4.8 GB fp32 each) with the early-exit forward.
- P1.e Gates before any quantization (fail-closed, same style as the pilot):
  1. Teacher reproduction: forward 2 sealed windows through the resident source; KLD vs the stored teacher
     logits must be ~1e-4 or the engine/source is not the teacher of record.
  2. Stock reproduction: load the LibertAIDAI NVFP4 experts through `NVFP4Linear` and score the 25-window panel
     in-process; must land at their measured 0.0492 within CI. This makes every in-process number directly
     comparable to the vLLM harness numbers.
  3. REAP windows rebuilt with the GLM tokenizer (64 x 2048, 16/axis; receipt).

## 4. Phase 2 — quantize and measure (~3 h per arm, arms sequential because one resident model uses all 4 GPUs)

- Arm G-U (deliverable): uniform NVFP4, Had16 in-side (or folded per P0.b), GPTQ on REAP Hessians, causal
  re-encode, FP8 embeddings, BF16 lm_head. Routed bytes identical to stock (171 GB; 188 GB total).
- Arm G-M (research): multi-tier {NVFP4, MXFP6, FP8, BF16} at 5.029 bpw routed (+20 GB), proxy allocation from
  per-unit residual candidates, causal. Only if P0.c says a kernel-light layout keeps the gain.
- Scoring: sealed 25-window panel in-process (FP64), paired vs stock NVFP4 (0.0492) and EXL3-4bpw runtime
  (0.0305) using their per-window teacher; plus the calibration-clean 17 windows. Confirmation re-anchors every
  4 layers as in the pilot.
- Export G-U to ModelOpt NVFP4 (static input_scale from the calibration amax; shared gate/up global scale —
  the exact bug their bring-up hit), serve on the cstechdev NoPE image with marlin, run THEIR kld_eval harness
  (`scripts/gpu_session_nvfp4.sh`) paired against EXL3-4bpw. That number is the deliverable.

## 5. Expected outcome (from the Qwen3 ratios; projection, not measurement)

| arm | projected panel KLD | vs stock 0.0492 | vs EXL3-4bpw 0.0305 | size | serves today |
|---|---|---|---|---|---|
| G-U uniform 4.5 bpw | 0.035 (0.031–0.045) | -29% (-9% to -36%) | 1.15x (1.03–1.47) | 188 GB | yes, checkpoint swap |
| G-M multi-tier 5.03 bpw | 0.027 (0.024–0.035) | -45% | 0.89x (0.77–1.15) | 207 GB | no (mixed-format MoE kernel) |

The range ends are WikiText (out of calibration domain) and selection (in domain). GLM's stock NVFP4 damage is
legal-heaviest (1.97x EXL3 on legal), and the REAP corpus is legal-heavy, so the in-domain end is the more likely.

## 6. Risks, in order

1. mHC hyper-connections may not commute with a residual-stream rotation fold (4 mixed streams, sinkhorn
   coefficients). If so: in-side rotation needs an online R^T·x (kernel prologue) or the recipe falls back to
   GPTQ-only (-20% class instead of -29%). Decided by reading modeling_glm5_next before Phase 2.
2. Memory: 330 GB resident + Hessian chunks; mitigations are 36-expert chunks and quantizing early layers
   first (they shrink as they go).
3. transformers-5.16 numerics vs the vLLM teacher path (gate P1.e.1 catches it).
4. Time: download 2–4.5 h; engine port 1 day; each arm ~3 h; export + vLLM harness ~1 h.
