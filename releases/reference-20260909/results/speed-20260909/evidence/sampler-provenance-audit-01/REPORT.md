# C1 regression evidence audit

CPU/read-only inspection; no GPU workload or new throughput measurement.

The previous turn reported existing results only; this turn adds image-pinned source evidence and corrects interpretation of acceptance metadata.

## Findings

Four sampler/speculator source files extracted from the installed site-packages paths of the exact grid376 and wait-first images match byte for byte. See sources.json for image identities and SHA256 hashes. This establishes equality of these files, not proof of which runner executed or equality of all runtime code.

The extracted Gumbel sampling implementation adds random noise only when temperature is nonzero. An absent request seed alone is therefore not sufficient to explain divergence in a temperature-zero Gumbel path. The workspace source previously consulted is not a substitute for the extracted image source.

The pinned llm_decode_bench sustained-duration monitor updates srv_spec_accept_rate from successive metrics scrapes (lines 9628-9645), then writes the last state value into the result (line 10021). Thus this field is a last available interval ratio, not a whole-cell accepted/drafted ratio. Comparing 0.62162162 against 0.43378995 does not establish a whole-window acceptance regression or explain its cause.

Measured throughput remains 199.7184 t/s C1 for retained grid376 and 165.3988 t/s C1 for wait-first. No rates have been adjusted. These are single-run fixed-prefix temperature-zero diagnostic observations, not quality qualification.

## Next action

Before another kernel candidate, establish the executed runner/sampler from archived runtime source and logs, and inspect whether exact request bodies and output token trajectories were retained. If absent, use a bounded paired output diagnostic with captured request hashes and generated tokens to distinguish greedy divergence from timing variation. Do not change seeds on the assumption that temperature-zero drafting is random. Do not adopt wait-first, resume N64/P4/TP2, restore production, or publish as part of this audit.
