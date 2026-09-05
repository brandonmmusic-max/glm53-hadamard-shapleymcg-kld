# Native P8 C1 decode: Nsight Systems attribution

Decision: **the fused TrellisMX MoE kernel is the measured C1 bottleneck; port
the B12X small-M scheduling geometry while preserving P8 arithmetic.**

This is a synthetic diagnostic of the already-failed P8 product arm, not a new
throughput or KLD qualification. It opened no calibration or evaluation role.
The service completed successfully, restored the prior backend/timer state,
and reached a maximum logged temperature of 83 C.

## What the trace proves

- All four TP workers recorded exactly 16 generation ranges and exactly 16
  `cudaGraphLaunch` calls. Each replay contains the same inventory of 2,433
  CUDA graph nodes. FULL CUDA-graph replay was active for the measured C1 path.
- Every replay contains 42 fused `MoEDynamicKernelBackend` launches and 42
  deterministic `W4A16TopKSumKernel` launches: one pair per routed layer.
- The top-k kernel always launches with `grid.x=16`. With hidden size 4096 and
  256 threads, `grid.x = ceil(M*4096/256)` proves `M=1`; the result is not a
  padded M2 proxy.
- The procedural MCG decoder remains fused into the MoE CUDA kernel that feeds
  E4M3 fragments to `mxf8f6f4`. No decoded-weight global materialization was
  observed or introduced by this diagnostic.

## Device timing

Values are medians over 16 repeated graph executions. CUDA kernel intervals
are asynchronous; category sums must not be added across streams or ranks.

| Worker | Trace device / CUDA id | Graph span ms | Fused P8 MoE ms | MoE / graph | Explicit fills ms | NCCL ms | Top-k sum ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2992 | 0 / 0 | 47.597 | 39.520 | 83.03% | 0.833 | 1.106 | 0.0363 |
| 2993 | 1 / 2 | 47.603 | 32.637 | 68.56% | 0.694 | 8.750 | 0.0289 |
| 2994 | 2 / 3 | 47.594 | 39.313 | 82.60% | 0.824 | 1.195 | 0.0358 |
| 2995 | 3 / 1 | 47.608 | 32.760 | 68.81% | 0.703 | 8.574 | 0.0293 |

The median distributed device span, computed as latest rank completion minus
earliest rank start rather than by summing ranks, is **47.614 ms**. Workers
2992 and 2994 are the critical compute ranks. Workers 2993 and 2995 finish the
fused MoE work about 6.7 ms earlier and spend the difference in NCCL. This is
the signature of a slow-rank fused-MoE limit, not a client accounting error,
missing graph replay, top-k reduction bottleneck, or dominant scratch clear.

Explicit `FillFunctor` kernels consume only 0.69–0.83 ms per graph. Reusing
workspace is still worthwhile, but eliminating every measured fill cannot
explain the 39.3–39.5 ms fused-MoE time on the critical ranks. The first
optimization target is therefore the small-M schedule inside the fused kernel.

## B12X reuse decision

B12X already contains an M1 route/slice schedule that exposes 32 FC1 tasks and
128 FC2 tasks for the current E288/H4096/I512/top-k-8 shape. The executed P8
deterministic path instead combines all four intermediate slices into one task
per selected expert, exposing only eight useful compute tasks at C1.

The existing B12X M1 implementation cannot be enabled unchanged: its FC2 path
interprets E2M1 weights, issues the E2M1 MMA, and uses BF16 atomic scatter. The
P8 port must reuse only its route/slice work decomposition and caller-owned
workspace pattern while retaining procedural MCG decode, E4M3 fragments,
physical UE8M0/32 scales, `mxf8f6f4`, and fixed-order deterministic reduction.
Simply relaxing the constructor flags would change the arithmetic and is
rejected.

## Boundaries and next gate

Nsight Systems proves graph replay, launch geometry, and elapsed intervals; it
does not provide occupancy, instruction, or SASS counter attribution inside
the fused kernel. The next candidate therefore needs a kernel-level A/B on
identical M1 routes and payloads, bit-exact output and five-run determinism
before integrated serving. Only after that closure may a new whole-model
decode product gate be frozen. The completed 20.69 versus 91.26 tokens/s speed
failure and its Shapley-allocation stop remain unchanged.

No LDLQ or BlockLDLQ was used. Confirmation and final roles, including the 28
reserved confirmation logits, remain unopened.

## Reproduction

Run:

```bash
python3 -m pytest -q tests/test_analyze_p8_nsys.py
python3 -m glm53_nvfp4.analyze_p8_nsys \
  --sqlite evidence/opened/codec-v2/p8-decode-profile-v1/native-p8-c1.sqlite \
  --receipt evidence/opened/codec-v2/p8-decode-profile-v1/client-receipt.json \
  --report evidence/opened/codec-v2/p8-decode-profile-v1/native-p8-c1.nsys-rep
```

The committed `analysis.json` was generated with `--output` from the immutable
trace. Trace SHA-256:
`f8d50d383197eb3d4eefa596a0ded1607184299bcffc9a98792ffbb9679246e0`.
Nsight report SHA-256:
`1c63f4d9f17b207a753e3ad00810e9c247bfa9993662d736ec73ff9af9590b0d`.

Existing attribution for ExLlamaV3, KQuant, QSRT, and `w4a8_trellis` applies.
