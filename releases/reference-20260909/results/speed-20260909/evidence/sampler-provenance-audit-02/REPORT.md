# Effective sampler source audit

The archived wait-first container sets PYTHONPATH=/opt/glm53-flash/vllm:/opt/glm53-flash/b12x. Its startup.log records Using V2 Model Runner. Source was therefore extracted from /opt/glm53-flash/vllm/vllm in both immutable images, correcting the prior audit's use of site-packages alone. Five inspected files match byte for byte; sources.json preserves identities and hashes. No bind overlays the source tree in the archived container.

This establishes the relevant configured source and observed runner, not a dynamic trace of every sampling branch. The extracted autoregressive speculator passes stored request temperature and seeds into sample_draft; the Gumbel helper gates noise on nonzero temperature.

Archived benchmark result records include output counts and request timings, not full SSE output or exact measured request bodies. The smoke response is retained but is a different prompt. Existing benchmark artifacts therefore cannot establish whether the C1 greedy text trajectories match.

Next diagnostic: capture identical benchmark requests and raw SSE in each candidate image without modifying payloads, fixed prefix, temperature, or benchmark source. Compare request hashes and the common generated-text prefix per matching request. Different text establishes divergence; equal text alone does not prove equal token IDs, logits, MTP proposals, or quality. Keep instrumented timing separate from ordinary llm_decode_bench results.

## Rejection utility follow-up

The exact-image rejection_sampler_utils.py also matches across both images. Lines 548 and 570-590 select the greedy branch at temperature zero and, outside synthetic mode, accept only when the draft token equals target argmax. Runtime configuration is standard rejection, not synthetic. A random draw exists in the loop but does not govern acceptance in this branch. This further rules out missing request seeds as an established explanation; numerical, batching, and trajectory differences still require the paired diagnostic.
