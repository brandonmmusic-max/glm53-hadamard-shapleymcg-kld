# V3 W6A8 runtime-marker gate

Date: 2026-09-03. Following the W6A8 config correction and before any Shapley score, the runtime canary and every mixed KLD invocation are strengthened to require the B12X log identity `source_format=mxfp6_w6a8 act_fmt=e4m3` in addition to the mixed-method and learned-rotation markers. The earlier coherent W6A6 diagnostic canary remains recorded under its original stage and cannot satisfy the new W6A8-specific stage gate.

An empty temporary guard at the not-yet-created Shapley manifest path prevents the already-running, weight-only bulk producer from advancing into candidate freeze or scoring when it finishes. The guard is removed only after the W6A8 canary passes. Packed E2M3 weight production is unaffected by the activation-format correction.
