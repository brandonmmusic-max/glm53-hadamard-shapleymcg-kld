# Autonomous execution prompt: GLM-5.3-Flash BF16 to NVFP4, V2

Execute `PLAN_GLM53_FLASH_NVFP4_V2.md` end to end on the local Pop!_OS host. The user explicitly authorized local downloads, GPU sessions, and automatic continuation on 2026-09-02. Publication and uploads remain unauthorized.

Use `/media/brandonmusic/klcstore/bmxfp4-glm53` for all large downloads, candidates, scratch, logs, and evidence. Use this Git repository for code. Preserve all unrelated files and preserve the initially stopped model-service state.

Mandatory rules:

1. Pin the BF16 source exactly to `zai-org/GLM-5.3-Flash-BF16@a6c167b62691b2bac901344b65cb651a70f53e43` and teacher exactly to `brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits@95f4fdd94bf29989db2e0d1054e4931f55edb6aa`. Never substitute the FP8 release or the 78-layer full-GLM logits.
2. Seal the experiment record before model/logit downloads or adaptive measurements. Keep fit, conditional-fit, selection, confirmation, and final disjoint. The old 25-window result is legacy evidence, not a final holdout.
3. Download the complete BF16 source resumably to klcstore and verify it. Download only the predeclared non-final teacher assets required for selection/confirmation.
4. Never try to resident-load the full BF16 checkpoint. Stream one layer from safetensors, quantify memory, and write a new candidate; the stock NVFP4 checkpoint is a carrier/control, not a source.
5. Run CPU format tests and a one-layer swap before a full quantization. Fail closed at every gate and preserve the failing artifact.
6. Primary candidate is identity GPTQ uniform NVFP4. Rotation and multi-tier formats are optional research arms only after the primary result.
7. Compare stock and candidate using the same pinned runtime and teacher metric. Open confirmation exactly once after freezing the candidate and analysis code. Require a paired bootstrap CI, not an aggregate-only claim.
8. Benchmark quality first, then target-only decode, Estonia, and LAVD separately. Prove the selected backend from runtime logs.
9. Do not upload, publish, change power limits, edit production configs, or delete anything outside the campaign root. Restore the pre-run stopped state after every GPU session and at completion.
10. Maintain `DECISIONS_GLM53.md`, stage receipts, raw logs, hashes, and `REPORT_GLM53.md`. Conclusions must be bounded to completed gates.

Continue without questions while a safe in-scope action remains. A gate failure stops dependent stages, but does not stop evidence collection, diagnosis, reporting, or restoration.

