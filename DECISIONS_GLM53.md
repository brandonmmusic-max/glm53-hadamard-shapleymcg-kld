# GLM-5.3-Flash NVFP4 V2 decisions

## 1. 2026-09-02: start and evidence level

Before model downloads or adaptive measurements, execute the uniform identity-GPTQ NVFP4 candidate as a confirmation-level sealed experiment. Treat the existing 25-window panel as opened legacy evidence. Do not claim a new final result without a newly captured post-freeze final panel.

## 2. 2026-09-02: corrected immutable sources

Use the 120-shard BF16 model `zai-org/GLM-5.3-Flash-BF16@a6c167b62691b2bac901344b65cb651a70f53e43` and teacher dataset `brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits@95f4fdd94bf29989db2e0d1054e4931f55edb6aa`. The official FP8 checkpoint and full 78-layer GLM teacher are invalid substitutes.

## 3. 2026-09-02: implementation architecture

Use a layer-streamed BF16 loader and a copy-on-write stock NVFP4 carrier. Require a one-layer pack/load/KLD pilot before the 42-layer campaign. This avoids the impossible full-model residency assumption and makes every output layer independently receipted.

