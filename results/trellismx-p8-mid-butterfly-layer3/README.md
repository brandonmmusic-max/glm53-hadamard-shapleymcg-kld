# TrellisMX-P8 shared middle-butterfly layer-3 interaction

Status: build preflight v1 failed before any output; corrected v2 is pending.

Decision 21 froze one `+pi/16` interaction regardless of the standalone tune
result. The physical target remains K4 procedural-MCG into E4M3 with one
UE8M0 scale per 32 weights (`4.25` payload bpw), full-Hessian GPTQ-style
inter-group feedback, and no LDLQ.

The first four-GPU build reached the post-encode diagnostic for each first
expert and then failed identically because a BF16 intermediate was passed to
an FP32 diagnostic `F.linear`. No codec chunk, dense chunk, receipt, KLD row,
or experimental result was written. All four logs are preserved under
`failures/build-v1/`; each has SHA-256
`a38cd5cd7251bd9189e38bc52791bed992e73c9e00da20b52af6abb7a1be1fe8`.

The correction casts both diagnostic operands to FP32 and adds a regression
test. It does not change the physical trellis encoder, rotation, calibration
role, scale geometry, or acceptance rule. A fresh design and destination are
required before retrying.
