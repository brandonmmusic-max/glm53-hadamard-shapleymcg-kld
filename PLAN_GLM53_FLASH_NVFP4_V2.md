# GLM-5.3-Flash BF16 to NVFP4 campaign, V2

Date: 2026-09-02. Execution host: Brandon's Pop!_OS workstation with four RTX PRO 6000 Blackwell 96 GB GPUs.

## Decision and claim boundary

Build a servable, uniform ModelOpt NVFP4 checkpoint from the immutable BF16 source and determine whether calibrated GPTQ improves paired teacher KLD over the existing stock NVFP4 checkpoint. The primary result is confirmation-level evidence. The already opened 25-window panel is a legacy benchmark, never a new final holdout. No result becomes a final claim until a newly generated, untouched panel is captured after the candidate and analysis code are frozen.

Primary candidate: identity-basis GPTQ, uniform NVFP4, group size 16, static within-group activation order, MSE scale search, BF16 lm_head, and the existing stock checkpoint as the non-expert carrier. Block-16 Hadamard and mixed precision are research arms only after the primary candidate passes.

Primary estimand: per-window `candidate KLD - stock KLD`, using full-vocabulary FP64 `KL(teacher || student)` on the predeclared confirmation windows. Success requires at least 10% lower mean KLD, a paired bootstrap 95% upper bound below zero, and no runtime/correctness gate failure.

## Immutable identities

- BF16 source: `zai-org/GLM-5.3-Flash-BF16@a6c167b62691b2bac901344b65cb651a70f53e43` (120 safetensors shards, 642,672,288,322 repository bytes).
- Teacher: `brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits@95f4fdd94bf29989db2e0d1054e4931f55edb6aa`.
- Stock carrier/control: `/home/brandonmusic/models/GLM-5.3-Flash-NVFP4`; its config and index hashes are recorded at preflight.
- Runtime image: `klc/glm53-flash-nvfp4:r19-sm120-tp4-ep4-dcp4-v79-dflash2-packed-aux-candidate@sha256:ed027a3a2ff93b9cf60c95f7adfaf676cabc8e040a28cffa7486a262c82fdfbe`.
- Calibration source: `/home/brandonmusic/klc-linux/reap_recall_build/work/calibration/reap_recall_calib.jsonl`, SHA-256 `cf247acc7c5da9f0600c7d6ab3b7c2fcfc54ec30b794e3b6047559285fa44df4`.

## Disjoint roles

The local 665-window token panel is already split into fit, conditional-fit, selection, confirmation, and legacy final roles. V2 selects deterministically by the smallest `sha256("glm53-nvfp4-v2:" + window_id + ":" + token_ids_sha256)` within each domain:

- fit: 64 windows, 16 per domain, activations/Hessians only;
- conditional-fit: 16 windows, 4 per domain, causal re-anchor diagnostics;
- selection: 16 windows, 4 per domain, recipe choice and early stopping;
- confirmation: 32 windows, 8 per domain, exactly one post-freeze paired comparison;
- final: empty in this protocol; the legacy 25 are reported separately and labeled opened.

The role-builder verifies token hashes, pairwise disjoint IDs, and pairwise disjoint token hashes before emitting manifests. Teacher logits are downloaded only for selected non-fit roles; downloading them does not authorize reading confirmation metrics before freeze.

## Architecture and storage

The 642 GB BF16 checkpoint cannot be resident in 384 GB VRAM. The quantizer therefore uses the safetensors index to load only tensors for one MoE layer at a time. The existing 188 GB NVFP4 checkpoint is copied/reflinked as a carrier; each completed layer is written to a new candidate shard and atomically indexed. A layer receipt records source shard hashes, tensor names/shapes, quantizer parameters, pack/unpack error, output hashes, elapsed time, and peak memory. BF16 source shards are immutable.

For each MoE layer (3 through 44), expert gate/up inputs share the routed input Hessian and a single ModelOpt `weight_scale_2`; down projection uses its own Hessian. The exact source tensor names and packed carrier layout must be derived from both indices and checked before GPU work. The pilot is one layer and one expert group; a full campaign is forbidden until its export reloads through the runtime.

## Gates

1. P0 preflight: revisions, role disjointness, tokenizer hash, topology, free bytes, initial service state, environment, carrier hashes, and runtime digest are recorded.
2. P1 pack: CPU synthetic E2M1 encode/decode round-trip, nibble order, scale dtype/shape, shared gate/up global scale, and candidate index integrity pass.
3. P2 mapping: every required source/carrier tensor maps exactly once; all 42 MoE layers and 288 experts have the expected shapes; no output alias targets an input.
4. P3 pilot quality: for the chosen layer, Hessian-GPTQ weighted error is below same-grid RTN; failure is preserved and blocks the full run.
5. P4 pilot runtime: the one-layer candidate loads in the pinned TP4/EP4/DCP4 runtime, backend identity is proven in logs, and a deterministic canary is coherent.
6. P5 full export: every layer receipt verifies; `hf cache verify` passes for the BF16 source; candidate inventory and config are immutable.
7. P6 freeze: candidate, analysis code, selection results, and confirmation-opening receipt are hashed before any confirmation score is read.
8. P7 confirmation: success threshold above, plus finite per-token KLD, no excluded windows, and stock/candidate evaluated on identical runtime configuration.
9. P8 performance: after quality only, run target-only decode, Estonia, and LAVD as separate regimes with raw JSON and server-start receipts. Never combine speculative and non-speculative throughput.

## Failure, stopping, and restoration

All failures remain in `evidence/`; no row is replaced. OOM first reduces expert chunk size, not data roles. Selection may stop after the pilot or a full primary failure; confirmation is opened at most once. No upload or publication is authorized. GPU sessions acquire the existing model-stack lock and restore the observed pre-run state. Preflight observed `glm53-r10-tp2-mtp3.service` inactive and port 8000 closed, so successful restoration means stopped, not forced startup.

