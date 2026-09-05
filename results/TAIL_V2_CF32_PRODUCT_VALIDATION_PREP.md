# Tail-V2 CF32 P8 versus production-EXL3 preparation

Status: **CPU-ready and sealed; no GPU launch, service mutation, student KLD
capture, determinism result, or speed measurement was performed in this step.**

The initial EXL3 image build (`exl3-tail-v2-product-image-v1`) failed closed
because it assumed a duplicate site-packages GLM source path that the parent
image does not contain. That receipt is preserved. The corrected source-only
build is `exl3-tail-v2-product-image-v1a`: image
`sha256:af4e6a9ac0feb29ae2399ea45fad7db189594f0ff564c5bff3a691156c60b6b7`,
with KPool SHA
`494192195da43c46d99a684555fc10fd13a19e89288cb9f51da2536ccdf1f251`,
identical to P8. A separate CPU receipt binds the actual EXL3 kernel source
SHA `ae92592ea8fcd249978134357ea3cd2510fe2aa9bdb1d1a3ab02afdbaeb39f45`
and its BF16 dispatch contract.

The first sealed execution plan (`...validation-v1.json`) is preserved as a
pre-execution invalidation because implementation sources changed after its
seal. It authorized no run. The current plan is
`experiments/tail-v2-p8-exl3-cf32-product-validation-v1a.json`, SHA-256
`7917435982fe66426e3f836d96ae90842601e4b801b9b34470f55b41802da135`.
It authenticates against the final lifecycle sources, 32 teacher windows,
both image receipts, activation receipt, and fresh payload audit.

The product protocol compares the already-qualified native-P8 Tail-V2 image
with a new production-EXL3 capture image carrying the **identical** KPool
Tail-V2 source. Both must run B12X MLA sparse attention, NVFP4 DS-MLA KV,
CUDA graphs, TP4, one sequence, and no MTP. P8 remains its real
TP4/no-EP/DCP1 E4M3 native-MMA product. EXL3 remains its real
TP4/EP4/DCP4 BF16 product. These are intentionally labeled a product/system
comparison, not a matched-topology codec causal comparison.

## Execution order

1. Build the EXL3 Tail-V2 image with
   `scripts/build_exl3_tail_v2_product_image.py`. Its parent is the immutable
   production-EXL3 forced-decode image, so capture is available for KLD and
   disabled by environment for speed. The Docker build fails unless the
   unmodified KPool source hash matches the P8 parent, applies the same patch,
   and installs byte-identical source into the vLLM checkout and site-packages.
2. Run the fresh 288-file P8/EXL3 payload audit already implemented by
   `glm53_nvfp4.p8_weight_identity`.
3. Resolve and seal the plan with
   `scripts/prepare_tail_v2_product_validation.py`. The plan command verifies
   all 32 conditional-fit teacher files but never enumerates confirmation or
   final roles.
4. Run `python -m glm53_nvfp4.tail_v2_product_runner quality` for five
   independent cold server processes per arm in alternating order.
   Within a process, request the fixed balanced CF32 sequentially. After every
   request, use `scripts/tail_v2_stream_score_delete.py score-retire` before
   requesting the next window.
5. Run `verify-determinism`. All five raw hashes must match in every arm/window
   cell. KLD uses the 32 repeat-1 windows only; repeats are not pseudo-replicates.
6. Run `python -m glm53_nvfp4.tail_v2_product_runner speed`. With capture
   environment variables absent, it reuses the established benchmark
   client for five new cold processes per arm. Primary cells are 32K
   Prometheus-validated prefill and C1 32K continuous-usage decode; 64K is
   secondary. Runtime logs are re-audited under the same fail-closed contract.

## Storage bound

One full raw is 1,268,157,440 bytes. The scorer permits only one active window
and the lifecycle rejects total campaign bytes at or above 30,000,000,000
bytes. It writes and fsyncs a
score receipt binding the raw SHA, per-row score NPZ, capture metadata, and
runtime audit before unlinking only that authenticated raw. It then fsyncs the
directory and writes a second retirement receipt. Failed/partial artifacts are
never removed automatically.

## Remaining device execution

- Execute the 10 cold KLD processes and separate 10 cold speed processes under
  fresh authorization; report every failure without reroll.

The lifecycle adapter performs authenticated per-container cleanup, verifies
that no owned campaign container remains live, and restores the exact prior
timer/backend state only after that safety check. Its tests prove request N is
scored/retired before request N+1. Forty relevant CPU tests pass, and both
real recipes produced fail-closed capture-on and capture-off launch arguments
without creating a container.

The draft proposal is
`experiments/tail-v2-p8-exl3-cf32-product-validation-proposal-v1.json`.
