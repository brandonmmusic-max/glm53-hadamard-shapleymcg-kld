# Prompt for Sol / Codex (or a local GLM-5.3-Flash agent): BMXFP4 → GLM-5.3-Flash NVFP4 campaign

You are picking up a quantization campaign on Brandon Music's rig (Pop_OS, 4x RTX PRO 6000 Blackwell 96 GB SM120,
no NVLink, 512 GB RAM; data volume `/media/brandonmusic/klcstore`). Everything below was measured on 2026-09-02
and is written up in the files listed in section 1. Read those files before doing anything. Then execute the plan
in section 5, gate by gate, and write decisions BEFORE results the way DECISIONS.md does.

## 1. Files of record (read first)

| file | what |
|---|---|
| `/media/brandonmusic/klcstore/bmxfp4/REPORT.md` | full Qwen3-30B-A3B results: pilot, round 2, round 2b (REAP calibration), stock anchors, FP8 heads |
| `/media/brandonmusic/klcstore/bmxfp4/DECISIONS.md` | numbered decision log, items 1–23; every deviation and post-hoc arm is declared there |
| `/media/brandonmusic/klcstore/bmxfp4/PLAN_GLM53_FLASH_NVFP4.md` | the GLM-5.3-Flash plan (phases 0–2, gates, risks, expected outcome) |
| `/media/brandonmusic/klcstore/bmxfp4/REPORT-ladder.md` | every arm x panel x activation mode with paired CIs |
| `/home/brandonmusic/KLC_SANDBOXES/bmxfp4-qwen3-30b-a3b/bmxfp4/` | code (git; HEAD 2d01a99); `results/` mirrors the reports and per-arm JSON receipts |
| `/media/brandonmusic/klcstore/bmxfp4/arms/<ARM>/` | per-arm `kld-<panel>-<a16|a4>.json`, per-token KLD arrays, `receipt.json` |

## 2. What the Qwen3-30B-A3B pilot established (final role 25 x 2048, W4A16 KLD(teacher‖student), experts-only scope)

| arm | routed bpw | KLD | note |
|---|---|---|---|
| EXL3 5.0 (turboderp) full / experts-only | 5.029 | 0.00707 / 0.00523 | control |
| stock NVFP4 (nvidia ModelOpt = RedHatAI, RTN) experts-only | 4.5 | 0.0163 | baseline (as shipped incl. attention: 0.0316) |
| **B5r: NVFP4 + block-16 Hadamard + GPTQ on REAP corpus** | 4.5 | **0.0115** | −29% vs stock (selection −36%, WikiText −9%); + FP8 embeddings: free |
| B7n: multi-tier {NVFP4, MXFP6, FP8, BF16}, proxy or Shapley allocation, causal | 5.029 | 0.0090 | −45% vs stock; 1.27x EXL3; Shapley = proxy (tie) |
| B7: same with the 4:8 sparse tier | 5.029 | 0.0137 | sparse tier costs 34% — never use it |
| B5h: full-width (QuaRot) Hadamard | 4.5 | 0.0112 A16 / 0.0331 A4 | A4 34% WORSE than block-16 — do not use |
| FP8 lm_head | | +14% | do not quantize the lm_head; FP8 embeddings are free |

Rules that came out of it: calibration data = Brandon's REAP corpus only (never the sealed evaluation windows);
Aumann–Shapley attribution is unnecessary without the sparse tier (and OOMs on the full GLM anyway); 4:8 tier dead.

## 3. Hard rules in this shop (violations end the run)

1. Calibration is baseline and the corpus of record is `/home/brandonmusic/klc-linux/reap_recall_build/work/calibration/reap_recall_calib.jsonl` (12,228 samples, 4 axes). Sealed panels are evaluation only.
2. Never modify a benchmark or a sealed panel; every number carries a receipt (sha256 of inputs, config, seal).
3. Declare arms and decision rules in DECISIONS.md before looking at their results; post-hoc arms are labeled post-hoc.
4. Never delete prior experiment data; scratch (Hessians, per-layer causal scratch, weight dumps of losing arms) may be cleared only when Brandon says so.
5. Brandon owns "done". Report what was measured, with CIs; do not round a loss into a win.
6. Never change GPU power limits. Kill processes by PID only (never pattern-kill; never touch `codex-remote-control.service`). No external pushes/PRs without explicit OK. Pinned serving configs are not changed unilaterally.
7. The daily driver (GLM-5.3-Flash EXL3 4bpw on :8000) is production: any GPU session that needs the whole rig goes through `KLC_SANDBOXES/glm53-flash-kld-eval/scripts/gpu_session*.sh` (flock, stops/restores production via trap) and is announced to Brandon first.

## 4. GLM-5.3-Flash assets and facts

- Architecture (from `~/models/GLM-5.3-Flash-NVFP4/config.json`): `Glm5NextForConditionalGeneration`, 45 layers (first 3 dense, 42 MoE), 288 routed experts x (gate/up/down 2048x4096), 1 shared expert, 8 active, sigmoid `noaux_tc` router with correction bias, `routed_scaling_factor` 2.5, `norm_topk_prob`, `swiglu_limit` 10, hybrid `layer_types` (linear_attention gated-delta-net + deepseek_sparse_attention), mHC hyper-connections (`hc_mult` 4, sinkhorn), NoPE MLA, vocab 154,880, 1 MTP layer, vision tower. Routed experts = 304.4B params; non-routed 7.4B; embeddings + lm_head 1.27B.
- Source of record: `zai-org/GLM-5.3-Flash` (HF, 328 GB, 62 shards; FP8 e4m3 block-scaled experts, BF16 attention/embeddings/lm_head/hyper-connection params). NOT on disk; there is no BF16 release. Needs ~330 GB free on klcstore (328 GB free now; Qwen3 scratch that may be cleared with Brandon's OK: `bmxfp4/exl3-recon` 113 GB, `bmxfp4/teacher/fit` 38 GB, `bmxfp4/arms/B7`+`B7a` 110 GB).
- Baseline: `~/models/GLM-5.3-Flash-NVFP4` (LibertAIDAI, ModelOpt RTN, 188 GB) measured 0.0492 (panel) / 0.0496 (clean 17) on the sealed 25 x 2048 panel; EXL3-4bpw runtime (`~/models/GLM-5.3-Flash-EXL3-4bpw`) 0.0305 / 0.0293; cloud EXL3 0.02455. NVFP4 = 1.70x EXL3 [1.47, 1.91]; legal domain worst (1.97x).
- Teacher: `KLC_SANDBOXES/glm53-flash-kld-eval/data/teacher/logits` (30 GB FP32, sealed 25 windows); harness = `kld_eval` CLI (KL(teacher‖student), FP64 log-softmax, full vocab, padded lm_head masked). Runtime capture via `VLLM_KLD_CAPTURE_DIR`; `scripts/gpu_session_nvfp4.sh` ≈ 6 min.
- Serving traps (memory, verified 2026-08-27): NVFP4 on SM120 only works on `klc/glm53-ref:w13fix` (cstechdev NoPE image) with `--moe-backend marlin` or `humming`; CUTLASS-family NVFP4 MoE emits garbage; `--no-enable-flashinfer-autotune` mandatory; `NCCL_P2P_LEVEL=SYS`; gate and up experts MUST share one NVFP4 global scale (vLLM applies `w13_weight_scale_2[:,0]` to both halves) — `~/glm53_port/patch_w13_gscale.py` exists for the stock checkpoint.
- transformers 5.16 has `glm5_next` (needed for the in-process engine; `klc-env` is transformers 4.52 and cannot load it — make a new venv).

## 5. Execute, in this order

### Phase 0 — deployability, on Qwen3 (no download; code in the repo; ~2 GPU-hours)
0.1 Add `--rotation had16-in` (rotate the "in" kind only; `make_rotation` returns None for kind "mid") and run
    `run_arm.py --arm B5r-in --method gptq --rotation had16-in --panels selection,final,wikitext --a4 --save-weights`
    with `BMXFP4_HESS_DIR=/media/brandonmusic/klcstore/bmxfp4/hessians-calib`. Compare to B5r (0.01152) and B3p-class GPTQ-only. Decision rule: if B5r-in keeps ≥ half of the rotation gain, the GLM recipe is in-side rotation.
0.2 Write `fold_rotation.py`: fold R16 into the residual stream (embeddings·R; attention q/k/v in-projections R^T-folded; o_proj and expert down_proj outputs ·R; RMSNorm weights folded into the following linears first; lm_head·R) so the runtime needs no online transform. Verify KLD == B5r-in (bitwise-close) on selection.
0.3 Write `export_modelopt_nvfp4.py`: pack the folded experts as ModelOpt NVFP4 (`weight` u8 low-nibble-first e2m1, `weight_scale` e4m3 [N, K/16], `weight_scale_2` f32, `input_scale` f32 from calibration amax; gate/up share `weight_scale_2`), config `quantization_config` like `nvidia/Qwen3-30B-A3B-NVFP4` (on disk under `bmxfp4/models/`). Serve with vLLM (marlin) and score the sealed Qwen3 panels through vLLM logits against `bmxfp4/teacher/`. Gate: vLLM number == in-process number within CI. This is the proof that the recipe is a checkpoint swap.
0.4 Per-layer tiering: `allocate.py` with a per-layer constraint on `arms/cand-calib-had16/candidates` at E1b bytes, one `run_causal.py` arm. Decision: if it keeps ≥ 70% of B7n's gain over B5r, the GLM 5.03 bpw arm is per-layer (kernel-light); else multi-tier stays research-only.

### Phase 1 — GLM source + engine
1.1 Get Brandon's OK on what to clear, then download `zai-org/GLM-5.3-Flash` to `/media/brandonmusic/klcstore/bmxfp4-glm53/models/` (detached; ~2–4.5 h).
1.2 New venv with transformers ≥ 5.16 (glm5_next), torch matching the rig; load the text model resident on 4 GPUs (`device_map`, vision skipped), FP8 experts dequant-on-the-fly. Implement `NVFP4Linear`/`MXFP6Linear` modules (packed + scales, dequant in forward) to replace experts as they are quantized (memory shrinks per layer).
1.3 Port `hessian.py`'s router hook to the GLM router (sigmoid + bias, top-8, norm, scaling 2.5) and derive the expert dispatch order from `modeling_glm5_next.py`; Hessians per layer in 4 chunks of 72 experts with the early-exit forward on the 64 REAP windows (rebuilt with the GLM tokenizer via `build_calib.py`).
1.4 Gates (all fail-closed, receipted): (a) teacher reproduction on 2 sealed windows (KLD ~1e-4 vs stored teacher logits); (b) stock reproduction: LibertAIDAI experts through `NVFP4Linear` score 0.0492 within CI in-process; (c) calibration receipt.

### Phase 2 — quantize, measure, ship
2.1 Arm G-U: uniform NVFP4, Had16 in-side (folded if 0.2 passed), GPTQ on REAP Hessians, causal re-encode with confirmation re-anchors every 4 layers, FP8 embeddings, BF16 lm_head. Same 188 GB as stock. Score the sealed panel in-process (FP64), paired vs stock and EXL3.
2.2 Arm G-M (only if 0.4 passed): multi-tier at 5.029 bpw routed, proxy allocation, causal.
2.3 Export G-U to ModelOpt format, serve on `klc/glm53-ref:w13fix` + marlin, run `scripts/gpu_session_nvfp4.sh` and `cli paired` vs EXL3-4bpw and stock. That paired number, with CI and McNemar, is the deliverable.

### Expected (projection from the Qwen3 ratios; say so in the report)
G-U ≈ 0.035 (0.031–0.045) = −29% vs stock, 1.15x EXL3-4bpw, drop-in. G-M ≈ 0.027 (0.024–0.035) = 0.89x EXL3 at +20 GB, needs a mixed-format kernel.

## 6. Reporting

Append to `DECISIONS.md` (numbered, timestamped, decision before result), extend `REPORT.md` with the same table
style, keep every arm's receipts, and end with a short message to Brandon: what was measured, the paired
comparisons with CIs, what was skipped and why, and what is running. No claims without a receipt.
