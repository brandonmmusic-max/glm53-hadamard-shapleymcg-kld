# GLM-5.3-Flash NVFP4 V2 decisions

## 1. 2026-09-02: start and evidence level

Before model downloads or adaptive measurements, execute the uniform identity-GPTQ NVFP4 candidate as a confirmation-level sealed experiment. Treat the existing 25-window panel as opened legacy evidence. Do not claim a new final result without a newly captured post-freeze final panel.

## 2. 2026-09-02: corrected immutable sources

Use the 120-shard BF16 model `zai-org/GLM-5.3-Flash-BF16@a6c167b62691b2bac901344b65cb651a70f53e43` and teacher dataset `brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits@95f4fdd94bf29989db2e0d1054e4931f55edb6aa`. The official FP8 checkpoint and full 78-layer GLM teacher are invalid substitutes.

## 3. 2026-09-02: implementation architecture

Use a layer-streamed BF16 loader and a copy-on-write stock NVFP4 carrier. Require a one-layer pack/load/KLD pilot before the 42-layer campaign. This avoids the impossible full-model residency assumption and makes every output layer independently receipted.

## 4. 2026-09-03: capture transport and checkpoint ABI correction

Use the teacher dataset's pinned BF16 hidden-state and router captures as a streamed calibration transport, selecting only the sealed fit-role rows. The resulting claim is capture-calibrated group-16 GPTQ, not causal propagation through a locally resident BF16 model. Store ModelOpt block scales in logical row-major checkpoint order; the 128-by-4 swizzle is performed after load for kernel ingestion. Retain the failed pre-pilot swizzled reconstruction as negative-control evidence.

## 5. 2026-09-03: post-quality benchmark implementation

Pin `local-inference-lab/llm-inference-bench` at commit `d115feee75095081bda2520aa046986a8885f449`. Only after the registered KLD confirmation passes, measure target-only decode, Estonia, and LAVD as three distinct candidate-only regimes. Do not treat their throughput or task scores as part of the primary KLD estimand.
