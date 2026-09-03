# V10 pre-inference amendment: full H16 chunk-root mount

Date: 2026-09-03. The first full-model H16 server attempt failed during
InstantTensor metadata loading, before readiness, inference, or any KLD row.
The sparse overlay's 210 candidate shard symlinks resolve into
`/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-v9-large/had16-all`, but the
container mounted only the overlay directory, stock carrier, learned-candidate
root, and model-storage campaign root. The worker therefore reported
`FileNotFoundError` for valid H16 shards beginning at routed layer 4.

The KLD and candidate-canary launchers now mount the complete H16 artifact root
read-only at its identical absolute path. This does not change a tensor, model
index, quantization recipe, activation transform, runtime image, role, metric,
or decision rule. The failed pre-inference session is retained under an
attempt-specific diagnostic name. The identical candidate must start, emit the
all-layer rotation runtime proof, and complete conditional-fit before selection
wave 2 can be authorized.
