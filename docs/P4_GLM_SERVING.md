# P4 integration into GLM serving

The opt-in GLM adapter now reaches a CUDA P4 backend for routed gate/up and
down projections. Host tests execute the exact local vLLM
`RoutedExperts.forward_modular` method and validate the load hook and both
carrier method signatures. Offline assembly proves the selected CUDA source
uses native E2M1 K64 matrix instructions. No GPU was launched in this stage;
coherent served output, device closure, full-model KLD, CUDA-graph parity and
TP4 throughput remain untested.

## Call path and ownership

```text
sitecustomize (GLM53_P4_NATIVE=1)
  -> p4_glm_serving.install: wrap GLM's FusedMoEFactory
  -> original factory: retain router, shared experts and TP runner
  -> selected RoutedExperts method: per-instance P4 specialization
       post-load -> pinned TP4 sidecar -> B12X P4 backend -> CUDA prepare
       forward_modular -> method.apply -> P4TrellisMoEBackend.apply
          BF16 X -> packed E2M1 + E4M3/16 activation prepass
          FC1: K4 weight prologue -> E2M1/E4M3/16 -> mxf4nvf4 K64
          FP32 clamped SwiGLU
          packed E2M1 + E4M3/16 activation prepass
          FC2: K4 weight prologue -> E2M1/E4M3/16 -> mxf4nvf4 K64
          fixed-order top-k weighted sum -> BF16 routed output
  -> original GLM runner: shared-expert combination and TP all-reduce
```

The router's original expert IDs, routing weights, normalization and scaling
reach P4 unchanged. The P4 method is modular, reports no internal shared
expert overlap, and leaves the enclosing runner's shared-expert scheduling
and final reduction intact. Stock method classes and unselected layers are
unchanged. Unsupported EP, DP, sequence parallelism, EPLB, DBO/microbatching, bias, LoRA,
input route weighting, different activation/clamp, or rotated boundaries fail
closed. The current served geometry is GLM E288/H4096/I512-per-TP-rank,
top-k 8, TP4 and SwiGLU clamp 10.

The source-level integration was checked against local vLLM commit
`df684ff47dcbf088b41311494fd20347a702e56a` at
`/home/brandonmusic/KLC_SANDBOXES/glm53-exl3-k4-sm120/vllm`.
Per-file hashes, exact commands and compiler output are in
`evidence/opened/codec-v2/p4-serving/`. This is source execution with host
stand-ins for model objects, not a complete imported vLLM server or CUDA run.

## Enabling the pinned backend

Set these variables in the vLLM worker environment, with the runtime patch
directory first on `PYTHONPATH`:

```text
GLM53_P4_NATIVE=1
GLM53_P4_NATIVE_LAYERS=3
GLM53_P4_NATIVE_SIDECAR_DIR=/p4-sidecars
GLM53_P4_NATIVE_DESIGN=/p4-design/encoder-design.json
GLM53_P4_NATIVE_MANIFEST=/p4-design/manifest.json
GLM53_P4_NATIVE_MANIFEST_SHA256=<exact manifest file hash>
GLM53_P4_NATIVE_BUILD_DIR=/cache/p4-native
```

Layer lists must be ascending, unpadded decimal lists without whitespace, such
as `3,4,44`. Other spellings fail before installation. This keeps the readiness
line identical to the launcher's literal `GLM53_P4_NATIVE_LAYERS` value.

The manifest uses `glm53-p4-mcg-tp4-manifest.v1`, identifies the v2 codec and
the exact source design, and contains all four rank files per listed layer.
Each entry binds `layer`, `rank`, basename `path`, exact `bytes`, and `sha256`.
The loader rejects missing/duplicate ranks, wrong layer/design, legacy ABI,
malformed scale/layout/dimension metadata, paths outside the mounted root,
and sidecar content changes. All six physical tensors are checked on CPU
before CUDA allocation. File hashes are checked before and after loading.
Successfully loaded sidecars replace the now-unused carrier parameters.

Python normally swallows exceptions raised by `sitecustomize`. An explicitly
requested P4 installation failure instead exits with status 78, including an
import-time `SystemExit` or a failure while reporting the traceback. A bad
sidecar or import cannot quietly serve the carrier as a supposed P4 result.
The readiness and first-forward receipts name both projections and the exact
codec/MMA path. Those logs are necessary dispatch evidence; they do not prove
arithmetic correctness by themselves.

## Optimizations present and their evidence

- Each stage quantizes A once; all output tiles reuse native packed A and its
  actual scale bytes. The weight stream still decodes within the producer
  warp's prologue while the MMA warp consumes the other shared-memory stage.
- A tight, exhaustively host-checked route bound avoids launching hundreds of
  idle expert tiles for single-token decode: E288/top-k8 has Y=8 versus Y=289.
- Stable sorting writes into reusable outputs. Routing and activation scratch
  is owned by one model namespace/TP rank and keyed by shape, CUDA stream and
  capture sequence. Sequential layers within that owner can reuse it; separate
  models, streams, eager execution and independent captures cannot. Returned
  model outputs own fresh storage.
- The shared library is cached by source/build identity. `cudaFuncGetAttributes`
  loads all functions before capture without a kernel launch. Cold compile or
  preparation during graph capture is rejected. All C launches use the
  caller's current stream; buffers retain stream lifetime records.

These changes remove repeated operations and reduce launch bounds in the
source. Their throughput effect and graph execution correctness require
device measurements. The compiler reports 62 registers, 1,728 bytes shared,
zero spills for the optimized projection. The two activation kernels use
31 registers each. No state-indexed table is stored: runtime lookup bytes=0.

P4 uses `mxf4nvf4 m16n8k64`, NVFP4's native issue class. P8 uses
`mxf8f6f4 m16n8k32`, twice the MMA issues per equal K span. P8 is a quality
product; P4 is the intended speed product. Neither instruction selection nor
this structural integration establishes a speed improvement.

## Focused scratch audit and lifetime restrictions

The initial `af76911` implementation keyed a process-global scratch pool only
by stream and shape. That was insufficient to establish independence between
models or captured graphs. The follow-up replaces it with explicit model
ownership, CUDA's capture-sequence identifier, and a nonblocking host-enqueue
guard. CUDA documents the sequence identifier as unique for the process
lifetime; the implementation distinguishes eager execution from capture ID
zero without assuming IDs begin at one. An invalidated capture or a failed
query is an error, never an eager fallback. The exact CUDA 13.2 header contract
and offline compilation are checked in the receipt. [CUDA stream API](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-runtime-api/group__CUDART__STREAM.html)

Captures replay stored addresses, and custom CUDA launches are outside Torch's
operator-level input-liveness tracking. Therefore each capture retains the
runtime, physical sidecars, scratch, and external-launch input references.
There is deliberately no eviction/reset operation: these pins remain until
worker exit. The GLM adapter rejects a second construction of the same model
namespace/layer/rank; reloading requires a new worker. Distinct model namespaces
receive independent owners. DBO and `use_ubatching` are rejected using the
actual vLLM parallel configuration before the stock carrier factory runs.
[PyTorch graph memory semantics](https://docs.pytorch.org/docs/main/notes/cuda.html),
[custom-launch liveness caveat](https://docs.pytorch.org/docs/stable/generated/torch.cuda.graph.html).

Tradeoffs: each eager/stream/shape or capture-specific workspace consumes
additional activation memory, and repeated recapture retains additional
resources until worker exit. One capture-state query and a host guard are
added per eager/capture forward; replay bypasses this Python path. Multiple
host calls into one owner's P4 projection cannot overlap; separate stream
buffers prevent GPU scratch aliasing, but concurrency itself is not qualified.
Warm-up/capture memory accounting and graph replay tests are required before
production use. Do not describe static pointer ownership as device closure.

The replayable audit decision, numbered attempts, exact source hashes,
readiness-grep command and source-contract checks are under
`evidence/opened/codec-v2/p4-serving/scratch-audit-v2/`.

## Remaining device gates

After the P8 build releases the GPUs, test exact packed decode bytes,
FC1/SwiGLU/FC2/output arithmetic, eager-versus-graph outputs, then coherent
GLM generation and same-window pseudoquant/kernel KLD. Preserve five-run
determinism and exact TP4 prefill/decode benchmark conditions. Keep the
28 confirmation logits unopened. No LDLQ or BlockLDLQ was implemented.
Also exercise multiple capture IDs, graph replay on alternate streams, model
teardown, repeated capture and peak retained-memory accounting. The host
ownership audit cannot replace any of these device gates.

The procedural constants, stream and lane conventions are ports from
ExLlamaV3 through KQuant/QSRT. Producer/MMA organization follows the vendored
B12X `w4a8_trellis` design. The v2 integer RNE/signed-zero projection, CUDA
activation reuse and GLM adapter are new implementation work. Retain
`LICENSE.exllamav3`, B12X's retained Apache-2.0 license and all qualifications
in `THIRD_PARTY_NOTICES.md`; this work does not relicense those dependencies.
