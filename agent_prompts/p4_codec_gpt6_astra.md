# GPT-6 Astra assignment: independent P4 codec and reference closure

You own an independent audit and implementation of the P4 codec/reference
boundary in the supplied isolated Git worktree. Commit to its current branch
and do not push. Do not edit the fused kernel implementation file owned by the
kernel agent.

## Exact question

Can a K4 procedural-MCG sliding 16-bit trellis stream be decoded bit-exactly
to the native NVFP4 operand representation—packed E2M1 nibbles plus physical
E4M3 residual scales per 16 weights—at a defensible exact stored rate?

Read `README.md`, `glm53_nvfp4/trellis_nvfp4.py`, the sidecar schemas, and the
P8 runtime boundary. Then implement or strengthen a CPU reference packer,
decoder, format contract, verifier, and focused tests for:

- per-element sliding-window state reconstruction with K4 storage;
- exact E2M1 code/nibble mapping, including signed zero or reserved-code
  behavior required by the actual Blackwell format;
- E4M3/16 residual-scale fitting, serialization, addressing, and round-trip;
- exact payload and container-overhead bpw accounting;
- a deterministic fixture consumable by a later GPU P4 closure probe;
- explicit rejection of shape, scale, endianness, and layout mismatches.

Do not claim that a CPU reference proves GPU execution or speed. Do not use
LDLQ or BlockLDLQ. Do not run GPUs, access protected roles, inspect the 28
reserved confirmation logits, download models, or modify services. Keep all
tables at or below 4 KiB and preserve/clarify ExLlamaV3, KQuant, QSRT, and
vendored `w4a8_trellis` attribution.

Place an evidence-bounded receipt under
`evidence/opened/codec-v2/p4-astra/`, run relevant tests, inspect the diff,
and commit the focused result.
