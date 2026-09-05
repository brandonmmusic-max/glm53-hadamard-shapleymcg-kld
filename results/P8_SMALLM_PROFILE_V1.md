# P8 small-M integrated decode attribution

Status: **diagnostic complete-prefix only; original strict trace gate failed.**
Product gate versus EXL3 remains failed; allocation remains stopped.

The measured N128x2 small-M serving path already improved unprofiled 32K C1
decode from 20.6937 to 75.5962 tokens/s in one integrated diagnostic. It still
trails the fixed EXL3 reference, 91.2645 tokens/s, by 17.168%. This trace is
for localization, not another throughput measurement or five-cold-run gate.

## What the complete replays show

The first 16 complete CUDA graphs on each of four distinct GPUs have matching
2,475-node inventories. Every replay contains 42 FC1, 42 FC2 and 42 top-k
reduction kernels. Launch geometry proves active M1. Thus the optimized
full-model GLM path is executing under CUDA-graph replay, not eager fallback.

The distributed median graph span is 12.9731 ms (range 12.9519–30.4844 ms;
the first-replay rank skew is retained). On critical worker 3097,
median summed kernel durations per replay are:

| Execution family | ms/token |
|---|---:|
| Routed FC1 gate/up | 4.968 |
| Routed FC2 down | 0.524 |
| Native dense MMA | 2.505 |
| GEMV | 0.833 |
| NCCL, including rank waiting | 0.799 |
| MHC | 0.647 |
| Explicit fills | 0.551 |

FC1 is about 90% of routed-MoE duration and 38% of graph span; FC2 is about
4% of graph span. Critical worker 3099 agrees: FC1 4.950 ms and FC2 0.526 ms.
This explains why the N256 FC2 A-reuse port yielded only a small whole-layer
improvement. It does not establish which instructions or internal barriers
dominate FC1. The next target is FC1 scheduling/occupancy and staging, with a
separate investigation of the 2.5 ms native dense-MMA family if needed.

Kernel sums can overlap; the JSON also reports interval unions. These are
16 correlated replays of one server, not 16 independent runs. Nsight overhead
prevents substituting reciprocal graph span for serving tokens/s. NCCL includes
waiting for slower ranks; symbols alone do not assign dense kernels to exact
model projections. No new KLD measurement is claimed.

### Next FC1 candidate, not yet measured

The source audit finds 32 route-by-N128 FC1 tasks in a 64-CTA grid. A narrower
N64 or N32 output tile would create 64 or 128 independent tasks while retaining
each output's full K4096 reduction order. This is a scheduling hypothesis,
not a measured gain. Gate/up already share the A fragment, and only one
grid-wide input-publication barrier remains; the old histogram/prefix path is
already bypassed for M1.

B12X phase1 is a structural donor, not a drop-in: its helper omits the active
dynamic path's gate upper clamp of 10 and up clamp of [-10, 10]. A port must
retain those clamps, K32 scale grouping, post-SwiGLU BF16 seam and FC2 ordered
accumulation. Splitting the K reduction is not the proposed optimization.
An identical-payload, bit-exact eager/graph gate must precede another serving
comparison. No FC1 candidate result is claimed by this report.

The audit pinned image `dynamic.py` to SHA-256
`46c5829009bf4be952eaf384cdd5fc0825cafa183298b270972aa6d8e735a6d9`;
relevant source locations are `dynamic.py:4702` (task ownership), `:5735`
(shared gate/up A), `:1651` (clipping), and `:5930` (BF16 activation seam).
Image `p8_small_m.py` has SHA-256
`a0c398e9d412672d1138c69a5c0c3677bb29b7215379c701062d29b8b9b9265f`.
The local donor file differs; FC2 provenance therefore follows the executed
image, not an unexecuted local source copy.

## Preserved capture failure

The frozen exact-count check failed. Worker 3097 has one negative-duration
NVTX generation event in addition to 16 valid ranges. Workers 3098 and 3099
have one extra trailing graph call each, with only four captured kernel nodes;
worker 3100 has 16 calls. The first 16 complete graphs finish before these
trailing records. The raw trace and first strict error remain immutable.

A separately selected diagnostic mode verifies the complete prefix, reports
every excluded tail record, requires unchanged node/symbol/geometry inventories
and matches graph calls to same-thread generation ranges. This mode is not a
pass of the original trace gate. No GPU rerun or protected-role opening was
used to obtain a cleaner result. See the explicit exclusions in
`evidence/opened/codec-v2/p8-smallm-profile-v1/analysis-diagnostic.json`.

## Identity, reproducibility and costs

- Plan: `experiments/p8-smallm-profile-v1.json`, SHA-256
  `e5d7c7b33a94cea2428a37712a15fc8bdea5dca031c7f9cda2ca75e19fa20495`.
- Runner commit: `8e25bb070d1ef0fdda81f8a115657886e6cb7dc7`.
- Image: `sha256:c9eddeca0d9dedf210eaaff918f1bf44d5338202902143f7d646d2325ad475ed`.
- Runtime tree matches commit `02418d30e245bfae6586cb34b46bafd7c9e6a98f`.
- TP4, no EP, DCP1, no MTP; synthetic 32K C1 input; FULL graphs.
- Same K4 procedural MCG, E4M3 reconstruction, UE8M0/32, identity boundary,
  4.25 routed bpw, no LDLQ. Decoder feeds register fragments to native
  `mxf8f6f4`; no BF16 decoded-weight buffer. Fusion does not by itself prove
  the decoder's instruction latency is hidden.
- **P8 uses twice NVFP4's MMA issue count.** It is a quality product, not the
  native NVFP4 speed-class P4 endpoint.
- This is a B12X scheduling/pipeline port retaining the ExLlamaV3 procedural
  law and KQuant/QSRT lineage; see `THIRD_PARTY_NOTICES.md` for retained terms.
- Capture ran 2026-09-05 06:28:48–06:32:18 UTC. Exit 0, maximum 68 C. Prior
  backend-active/timer-inactive states restored. No protected inputs opened.

Replay analysis (no GPU required):

```bash
python3 -m glm53_nvfp4.analyze_p8_smallm_nsys \
  --trace evidence/opened/codec-v2/p8-smallm-profile-v1/smallm-c1.sqlite \
  --client evidence/opened/codec-v2/p8-smallm-profile-v1/client/receipt.json \
  --report evidence/opened/codec-v2/p8-smallm-profile-v1/smallm-c1.nsys-rep \
  --allow-trailing-capture-artifacts --output /tmp/p8-smallm-diagnostic.json
```

The synthetic client's `profiler_source.image` records the inherited profiler
source audit image, not the deployed candidate. Execution identity is the
image above and the runner receipt. Private container environment dumps are
not published. The launch builder, frozen source-run hash and plan preserve
the authorized reproduction recipe without exposing those dumps.
