# P4 kernel implementation receipt

**Evidence level: structural. NVFP4 speed claim: blocked.**

Implemented a separate `P4NativeTPMoE` with two fused trellis projections,
an explicit launch ABI, and a synthetic CPU/device probe seam. The K4 stream
decodes to packed E2M1 with physical E4M3/16 scales in double-buffered shared
stages. The producer can prepare the next tile while the MMA warp consumes
the previous tile. There are zero lookup-table bytes and no dense floating
weight tensor or floating-weight matmul in the runtime. FP32 scratch stores
activations. P8 sources remain byte-for-byte unchanged.

Demonstrated: compiled C++ decode agrees with an independent IEEE-half CPU
oracle and the frozen repository law for all 65,536 states; cyclic sliding
windows, full projection nibble packing, scale bytes/addresses, and K64
register/scale-selector mappings pass. Five CPU reference executions have
identical hashes. PTX, cubin and the linked launch library select only
`mxf4nvf4` / `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` at all 15 static MMA
sites (compiler unrolling, not 15 issues per K64). Projection compilation
reports 72 registers, 1,728 shared bytes, no local stack or spills.

Validation: **46 passed, 1 skipped**; the skipped existing test requires a
CUDA Viterbi encoder. Exact commands, environment, source/artifact hashes
and limitations are in `kernel-receipt.json`. Raw output is retained in
`kernel-tests-final.txt`; the default CPU probe is `kernel-cpu-probe.json`.
Attempt 1 passed 27 tests and failed the full-library link because nvcc's
architecture shorthand included generic `sm_120`. Explicit
`compute_120a`/`sm_120a` code generation fixed the link; attempt 2 passed all
28 then-existing focused tests. Both attempt logs are retained.
Source/documentation diff checks pass. The full diff check flags only
pytest-emitted trailing spaces in the preserved attempt-1 traceback.

**Not tested:** any GPU execution, device byte equality, race/sanitizer
closure, real GLM sidecars/dimensions, integrated KLD, CUDA graphs, thermals,
or GPU determinism/throughput. No selection, confirmation, final, or reserved
confirmation logits were opened. No model was downloaded, no service was
launched/modified, and no LDLQ/BlockLDLQ was used.

The speed claim still requires parent device closure and profiling, then
matched TP4 prefill/decode and five-run determinism. Stable route sorting,
per-call scratch, two-warp M16/N8 tiling, and the explicit activation scale
policy are unqualified costs/choices. No real-model sidecar converter or
production loader integration is supplied. See `docs/P4_NATIVE.md` for the
ABI and the unexecuted parent probe command. ExLlamaV3, KQuant/QSRT, and
vendored B12X port notices are retained and expanded in `THIRD_PARTY_NOTICES.md`.
