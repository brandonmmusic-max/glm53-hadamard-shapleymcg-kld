# Autonomous run: a better NVFP4 for GLM-5.3-Flash (Codex/Sol or Claude as the harness)

You are running this end to end on Brandon Music's rig without him. He will not answer questions during the run.
Every decision you might otherwise ask about is settled below; where something is not settled, apply the
decision rules in section 6, log the choice in DECISIONS_GLM53.md before acting on it, and continue. Stop only
on a failed gate (section 5) — then write the report and end.

Rig: Pop_OS, 4x RTX PRO 6000 Blackwell 96 GB (SM120, no NVLink), 512 GB RAM, `/media/brandonmusic/klcstore`
(1.3 TB free). Working root for this campaign: `/media/brandonmusic/klcstore/bmxfp4-glm53/` (create it).
Code root: `/home/brandonmusic/KLC_SANDBOXES/bmxfp4-qwen3-30b-a3b/bmxfp4/` (git) — the Qwen3 reference
implementation you port; make a sibling repo `bmxfp4-glm53/` for the GLM code and commit as you go.

## 1. Files of record — read before anything else

- `/media/brandonmusic/klcstore/bmxfp4/REPORT.md` — all Qwen3-30B-A3B measurements (stage zero of this campaign; DONE, do not repeat any of it)
- `/media/brandonmusic/klcstore/bmxfp4/DECISIONS.md` — decision log 1–23 (the style to follow: numbered, timestamped, decision written BEFORE the result)
- `/media/brandonmusic/klcstore/bmxfp4/PLAN_GLM53_FLASH_NVFP4.md` — the plan this prompt executes (its "Phase 0 on Qwen" is superseded by section 3 here)
- Reference code (Qwen3): `nvfp4.py` (NVFP4/MXFP6/FP8 fake-quant, MSE scale search, byte accounting), `gptq.py` (batched GPTQ over experts, static in-group act-order), `rotation.py` (Had16, Cayley), `hessian.py` (route-weight^2 Hessians + 256 routed samples/expert, router-hook dispatch order), `run_arm.py` (arm/candidates modes, `process_layer`), `run_causal.py` (causal re-encode + confirmation re-anchors), `allocate.py` (exact-byte MCKP), `stock_nvfp4.py` (ModelOpt / compressed-tensors NVFP4 dequant, nibble order resolved by reconstruction), `score_saved.py` (FP8 heads), `kld.py` (per-window KL(teacher‖student), paired bootstrap), `build_calib.py` (REAP windows)

## 2. What stage zero (Qwen3-30B-A3B, 2026-09-02) settled — carry these, do not re-derive

| finding | number | consequence for GLM |
|---|---|---|
| NVFP4 + block-16 Hadamard + GPTQ on the REAP corpus vs stock RTN NVFP4, same 4.5 bpw | −29% final, −36% selection, −9% WikiText (all CIs exclude 0); W4A4 −8% | the deliverable recipe |
| multi-tier {NVFP4, MXFP6, FP8, BF16} at 5.029 bpw, proxy allocation, causal | −45% vs stock; Shapley allocation = proxy (tie) | research arm; NO Shapley attribution (it also OOMs on the full GLM) |
| 4:8 pair-structured sparse tier | costs 34% of KLD under the best allocation, 5x at 4 bpw | never on the ladder |
| full-width (QuaRot) Hadamard | W4A16 −3%, W4A4 +34% worse | never; block-16 only |
| calibrated learned rotation (Cayley) | lost/tied vs Had16 twice | Had16 only |
| FP8 embeddings / FP8 lm_head | free / +14% | FP8 embeddings yes, lm_head stays BF16 |
| calibration data | sealed eval windows are NOT calibration data | REAP corpus only |

## 3. Decisions that replace the Qwen "Phase 0" (made now, autonomously applicable)

- Rotation for the deliverable checkpoint: the runtime NVFP4 MoE kernels compute W·x, so a rotated expert needs R^T·x. Test the residual-stream fold ON GLM (section 5, gate G4): fold R16 into embeddings, attention/linear-attention/dense/shared-expert in- and out-projections, expert down_proj outputs, MTP layer, lm_head (RMSNorm weights folded into the following linears first; hyper-connection mixing must commute — verify numerically). If the folded unquantized model reproduces the teacher logits (gate G4 passes), the deliverable uses in-side Had16 (gate/up inputs rotated, down_proj input unrotated). If G4 fails, the deliverable is GPTQ-only (identity rotation), and the rotated variant is still run as an in-process research arm so its value on GLM is measured.
- Multi-tier: per-unit tiers, in-process score only (no servable kernel exists); run after the deliverable.
- Panels: the sealed GLM-5.3-Flash 25-window panel (teacher logits already on disk) is the final role; the calibration-clean 17-window subset is reported alongside; confirmation re-anchors use 8 of the 25 windows chosen by the harness's manifest (log which).

## 4. Assets and facts (verified 2026-09-02)

- Source of record: `zai-org/GLM-5.3-Flash` (HF, public, 328 GB, 62 shards; FP8 e4m3 block-scaled experts; BF16 attention, embeddings, lm_head, hyper-connection params; `transformers_version` 5.16.0, arch `Glm5NextForConditionalGeneration`). NOT on disk. There is no BF16 release; FP8 is the official numerics.
- Architecture (`~/models/GLM-5.3-Flash-NVFP4/config.json`, text_config): 45 layers (first 3 dense, 42 MoE), 288 routed experts x (gate/up/down 2048x4096), 1 shared expert, 8 active, sigmoid `noaux_tc` router with correction bias, `routed_scaling_factor` 2.5, `norm_topk_prob`, `swiglu_limit` 10, `layer_types` hybrid (linear_attention gated-delta-net + deepseek_sparse_attention), mHC hyper-connections (`hc_mult` 4, sinkhorn), NoPE MLA (`qk_rope_head_dim` 0), vocab 154,880, 1 MTP layer, vision tower (skip). Routed experts = 304.4B params (171.2 GB at 4.5 bpw; 191.4 GB at 5.029); non-routed 7.4B; embeddings + lm_head 1.27B.
- Baselines already measured by Brandon's harness on the sealed panel: stock NVFP4 `~/models/GLM-5.3-Flash-NVFP4` (LibertAIDAI, ModelOpt RTN, 188 GB) 0.0492 panel / 0.0496 clean; EXL3-4bpw runtime `~/models/GLM-5.3-Flash-EXL3-4bpw` 0.0305 / 0.0293 (cloud 0.02455); NVFP4/EXL3 = 1.70x [1.47, 1.91]; legal domain worst (1.97x).
- Teacher: `/home/brandonmusic/KLC_SANDBOXES/glm53-flash-kld-eval/data/teacher/logits` (30 GB FP32, 25 sealed windows). Harness: `kld_eval` in that repo — KL(teacher‖student), FP64 log-softmax, full vocab, padded lm_head rows masked; runtime capture via `VLLM_KLD_CAPTURE_DIR`; `scripts/gpu_session_nvfp4.sh` (~6 min) and `cli paired`.
- Calibration corpus of record: `/home/brandonmusic/klc-linux/reap_recall_build/work/calibration/reap_recall_calib.jsonl` (12,228 samples, 4 axes). Tokenize the raw `text` field with the GLM tokenizer, EOS between samples, no truncation, per-axis packing into 64 x 2048 windows (16 per axis; first 32 = attribution-style subset, not needed here). `build_calib.py` does this for Qwen — add a `--model` argument.
- Serving facts for NVFP4 GLM-5.3-Flash on SM120: only `klc/glm53-ref:w13fix` (cstechdev NoPE image) with `--moe-backend marlin` (or `humming`) is correct; CUTLASS-family NVFP4 MoE emits garbage; `--no-enable-flashinfer-autotune` mandatory; `NCCL_P2P_LEVEL=SYS`; `--kv-cache-dtype fp8`; gate and up experts MUST share one NVFP4 global scale (vLLM applies `w13_weight_scale_2[:,0]` to both) — write the checkpoint that way from the start.
- Production: GLM-5.3-Flash EXL3-4bpw serves on :8000 (compose in `~/KLC_SANDBOXES/glm-5.3-flash-exl3-4bpw-release/runtime`). Any step that needs the GPUs goes through `KLC_SANDBOXES/glm53-flash-kld-eval/scripts/gpu_session.sh` (flock `model-stack.lock`, stops timer/backend/production, restores via trap). THIS PROMPT IS BRANDON'S AUTHORIZATION to take the GPUs that way; production must be back up after every session and after any failure (verify with a request to :8000 before moving on).
- Environment: `klc-env` is transformers 4.52 — it cannot load glm5_next. Create `/home/brandonmusic/glm53-quant-env` with torch for cu13x and transformers ≥ 5.16 (glm5_next is on transformers main: `src/transformers/models/glm5_next/`).

## 5. Gates (fail-closed; a failed gate ends the run with a report, never a silent fallback)

- G1 Source download complete and byte-verified against the HF file list (sizes + sha256 where published).
- G2 Engine loads the text model resident across the 4 GPUs (FP8 experts dequant-on-the-fly, vision skipped), and a forward on 2 sealed windows reproduces the stored teacher logits: mean KLD ≤ 1e-3 (report the exact value).
- G3 Stock reproduction: the LibertAIDAI NVFP4 experts installed through your `NVFP4Linear` (exact E2M1 x E4M3 x FP32 dequant; nibble order resolved by reconstruction error like `stock_nvfp4.py`) score the 25-window panel in-process at 0.0492 within its CI. From here on, in-process numbers are comparable to the harness numbers.
- G4 Rotation fold (section 3): folded, unquantized model vs teacher ≤ 1e-3 mean KLD. Pass → in-side Had16 deliverable; fail → GPTQ-only deliverable (log DECISIONS entry with the failure mode: which module family broke commutation).
- G5 Calibration receipt: 64 windows, per-axis sample rows, source sha256, tokenizer id.
- G6 Export round-trip: the exported ModelOpt NVFP4 checkpoint reloaded through `NVFP4Linear` scores identically (bitwise on logits, or within 1e-5 KLD) to the in-process arm it came from.
- G7 Serving: vLLM (marlin) serves the exported checkpoint coherently (the harness's canary-loader + a 200-token generation that is not degenerate), and the harness-measured panel KLD is within CI of the in-process number.

## 6. Decision rules for anything not covered

1. Any OOM: halve the Hessian expert chunk (72 → 36 → 18) before anything else; then reduce calibration windows to 32 (log it); never reduce the sealed panel.
2. Any kernel/library failure inside vLLM: try `humming` after `marlin`; do not try CUTLASS backends.
3. Download stalls > 30 min: resume with `hf_hub_download`/`snapshot_download` (they resume); do not restart from zero.
4. Time budget: 36 h wall from start. If the deliverable arm (G-U) is scored and exported by then, ship it and skip the research arms; if not, write the report on whatever is sealed.
5. Never delete anything outside `/media/brandonmusic/klcstore/bmxfp4-glm53/` and your venv; inside it, delete only your own per-layer Hessian scratch after each layer is quantized (keep stats), exactly as `run_causal.py` does.
6. Do not upload anything to Hugging Face and do not open PRs; the checkpoint stays on klcstore for Brandon.
7. Never change GPU power limits; kill only by PID; do not touch `codex-remote-control.service` or the Claude/Codex tmux sessions; do not edit pinned serving configs (make a new compose file for the NVFP4 candidate).
8. Long jobs: launch detached (`setsid nohup … &`), record PIDs, wait with `tail --pid=PID -f /dev/null` inside timeouts; the harness kills background tasks when a turn ends.

## 7. Execute

1. Setup: create the working root, venv, and repo; write DECISIONS_GLM53.md item 1 (start time, this prompt's sha256, the arms and gates you will run) before any download.
2. Download the source (detached) while you build the engine and port the code: `NVFP4Linear`/`MXFP6Linear`, GLM router hook (sigmoid + bias, top-8, norm, x2.5; dispatch order derived from `modeling_glm5_next.py`), Hessian chunking, `process_layer` for the GLM expert layout (gate/up stacked share one Hessian and one NVFP4 global scale; down_proj separate), FP8 embeddings, calibration windows (G5).
3. Gates G1–G4 in order.
4. Arm G-U (deliverable): uniform NVFP4 4.5 bpw routed, rotation per G4, GPTQ (static in-group act-order, MSE scale search with 4-vs-6 range) on REAP Hessians, causal layer-by-layer re-encode (Hessians recaptured per layer on the partially quantized model, early-exit forward), confirmation re-anchors every 4 layers, FP8 embeddings, BF16 lm_head. Score: sealed 25 (FP64), clean 17, W4A16 and W4A4 emulation; paired vs stock (G3 numbers) and vs EXL3 (harness per-window teacher). Export (ModelOpt format, shared gate/up global scale, static `input_scale` from calibration amax), G6, serve, G7, harness paired run.
5. Research arms, in-process only, if the budget allows: G-R (the rotation variant not chosen in G4), then G-M (multi-tier 5.029 bpw: candidate ladder {NVFP4, MXFP6, FP8, BF16} from full-expert output residuals on the stored routed samples, exact-byte knapsack, causal).
6. Report: `REPORT_GLM53.md` (tables in the Qwen REPORT.md style: arm, bpw, panel, mean KLD, top-1, p99, paired diff with 95% CI vs stock and vs EXL3; allocation histograms; re-anchor trajectories; gate values; deviations), `DECISIONS_GLM53.md`, every arm's receipts, and a final message to Brandon of ≤ 15 lines: the deliverable's paired numbers, whether it is servable, where the checkpoint and compose file are, and what was skipped.

Expected (projection from stage zero; state it as such): G-U ≈ 0.035 (0.031–0.045) = −29% vs stock, ≈1.15x EXL3-4bpw, same 188 GB, checkpoint swap. G-M ≈ 0.027 (0.024–0.035) = 0.89x EXL3 at +20 GB, not servable today.
