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
unchanged. Unsupported EP, DP, sequence parallelism, EPLB, DBO, bias, LoRA,
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

The manifest uses `glm53-p4-mcg-tp4-manifest.v1`, identifies the v2 codec and
the exact source design, and contains all four rank files per listed layer.
Each entry binds `layer`, `rank`, basename `path`, exact `bytes`, and `sha256`.
The loader rejects missing/duplicate ranks, wrong layer/design, legacy ABI,
malformed scale/layout/dimension metadata, paths outside the mounted root,
and sidecar content changes. All six physical tensors are checked on CPU
before CUDA allocation. File hashes are checked before and after loading.
Successfully loaded sidecars replace the now-unused carrier parameters.

Python normally swallows exceptions raised by `sitecustomize`. An explicitly
requested P4 installation failure instead exits with status 78, so a bad
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
  is keyed by shape and CUDA stream and shared by sequential layers, avoiding
  one retained workspace per layer. Returned model outputs own fresh storage.
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

## Remaining device gates

After the P8 build releases the GPUs, test exact packed decode bytes,
FC1/SwiGLU/FC2/output arithmetic, eager-versus-graph outputs, then coherent
GLM generation and same-window pseudoquant/kernel KLD. Preserve five-run
determinism and exact TP4 prefill/decode benchmark conditions. Keep the
28 confirmation logits unopened. No LDLQ or BlockLDLQ was implemented.

The procedural constants, stream and lane conventions are ports from
ExLlamaV3 through KQuant/QSRT. Producer/MMA organization follows the vendored
B12X `w4a8_trellis` design. The v2 integer RNE/signed-zero projection, CUDA
activation reuse and GLM adapter are new implementation work. Retain
`LICENSE.exllamav3`, B12X's retained Apache-2.0 license and all qualifications
in `THIRD_PARTY_NOTICES.md`; this work does not relicense those dependencies.
