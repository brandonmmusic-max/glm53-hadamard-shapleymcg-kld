# P4 ISA/runtime audit

**Verdict: the proposed mechanism is plausible, but neither “native Tensor
Core P4” nor “NVFP4 speed class” is established by this checkout.** The
implemented procedural runtime inspected here is P8. Its FP4-typed descriptor
carriers do not establish the arithmetic instruction it executes. There is
also a concrete rounding incompatibility to resolve before porting the MCG
prologue to E2M1.

Audit target: `74471a7cba6a5058088608d71949f36a541045b1`, branch
`astra-p4-critic`, initially clean. The assignment plan is
[p4-astra-agent-plan-v1.json](../experiments/p4-astra-agent-plan-v1.json).
Its older `repository.base_commit` is provenance, not the checkout audited.
Sibling worktrees and later kernel/codec commits are outside this verdict.
The [critic receipt](../evidence/opened/codec-v2/p4-astra/critic-receipt.json) pins inspected
sources, this report, and the replayable diagnostic script.

Only source, public ISA documentation, and synthetic CPU diagnostics were
used. No role manifests, model tensors, teacher payloads, or reserved
confirmation logits were opened. No GPU initialization, GPU compilation,
service operation, download of models, LDLQ, or BlockLDLQ was performed.
Repository narrative summaries are not accepted as current P4 measurements.
Role disjointness and holdout protection were **not verified**; honoring an
access boundary does not prove cryptographic isolation.

**Evidence matrix.** Status applies to this contract at the audited commit,
not to every historical experiment in the repository.

| Evidence | Current observation | Status and claim limit | Required gate below |
|---|---|---|---|
| Source inspection | CPU E2M1 projection/packing and P8 runtime call sites inspected; 15 Python files syntax checked | Demonstrated structural inspection; no complete procedural P4 execution path established | G0 |
| CPU bit closure | Three offline audit diagnostics pass, including two independent binary16-add models over all 65,536 MCG states | Partial analytical evidence; repository Torch code and complete stream-to-operand closure **Not tested** | G1 |
| GPU arithmetic closure | No device test or disassembly produced | **Not tested**, including FC1, FC2 and scale-fragment mapping | G2–G3 |
| Integrated serving | Existing hook and sidecar class explicitly select P8 | P4 **Not tested**; startup or carrier dtype would be insufficient | G4 |
| KLD | No role payloads or logits accessed | **Not tested**; no quality, equivalence or superiority claim | G6 |
| Determinism | P8 source offers route-slot output plus top-k reduction | P4 five-run bitwise determinism **Not tested** | G5 |
| Prefill speed | No P4 timing, utilization or transfer counters | **Not tested** | G7–G8, prefill cells |
| Decode speed | No P4 C1 or concurrent serving timings | **Not tested**; aggregate throughput cannot establish C1 | G7–G8, decode cells |

**Source findings and blockers.** Line references below refer to the audited
commit; the receipt records content hashes.

| ID | Inspected source and observation | Consequence |
|---|---|---|
| F1 | [trellis_nvfp4.py](../glm53_nvfp4/trellis_nvfp4.py), `procedural_state_values` (287), `e2m1_state_lut` (311), `_values_to_codes` (438) | MCG reconstructs and rounds a half sum, then projects by nearest **magnitude**, ties toward smaller magnitude, and canonicalizes zero. Compander value and operation order must be frozen. |
| F2 | Same file, `tensor_core_permutation` (115), packing (362), cyclic reconstruction (408), logical inverse (478), endpoint decode (1361) | The 16×16 EXL3 stream permutation is a source-layout contract. Its bijectivity alone does not prove SM120 MMA lane placement. |
| F3 | [endpoint_codec.py](../glm53_nvfp4/endpoint_codec.py), `pack_identity_k4_stream` (179); [export_native_endpoint.py](../glm53_nvfp4/export_native_endpoint.py), export metadata (53–68) | `identity-low4` is a different law from MCG. Export explicitly disables runtime trellis decode. Neither establishes the requested procedural prologue. Canonicalized zero also requires care when claiming arbitrary raw-nibble identity. |
| F4 | [p8_native_kernel.py](../runtime_patch/p8_native_kernel.py), metadata/shape checks (92–143), compile (194–299) | Required alphabet is E4M3, law is `procedural-mcg-alpha2`, scales are UE8M0/K32. K4 names branch-bit rate, not arithmetic precision. |
| F5 | [dynamic.py](../runtime_patch/b12x_h16/b12x/moe/_shared/kernels/dynamic.py), `_setup_attributes` (1280), FC1 decode/MMA (5516/5726/5772), FC2 decode/MMA (7498/7638) | A `MmaMXF4NVF4Op` layout object exists, but trellis consumers call `mxfp8_mma_m16n8k32_f32_e4m3`. An FP4 descriptor or constructor search hit is a false positive for native P4. Imported intrinsics are not vendored here, so emitted PTX/SASS remains unverified. |
| F6 | P8 class storage/scale setup (145–187), call scratch and dispatch (315–421); [sitecustomize.py](../runtime_patch/sitecustomize.py), P8 hook (33–234) | Aliased FP4 descriptor carriers coexist with uint32 stream pointers and logical/repacked scale planes. The hook verifies TP4 for P8. P4 needs its own complete binding, scale ABI, graph behavior and inventory. |
| F7 | [w4a8_mcg_decode.py](../runtime_patch/p8_mcg/w4a8_mcg_decode.py) (21–144); [patch_split_phases.py](../runtime_patch/p8_mcg/patch_split_phases.py) | Both monolithic and split P8 decoders exist. Split source is patched by hash, but the base phase1/phase2 modules and lane/staging helpers are imported from external B12X. Their installed contents were not inspected. |
| F8 | [P8 Dockerfile](../runtime_patch/p8_mcg/Dockerfile) | `FROM` uses a mutable local tag. A parent-digest label does not constrain tag resolution or attest installed code. A runnable P4 candidate must pin the actual image digest and dependency/source hashes. |
| F9 | [test_trellis_nvfp4.py](../tests/test_trellis_nvfp4.py), [test_trellis_mxf.py](../tests/test_trellis_mxf.py), [test_modelopt.py](../tests/test_modelopt.py) | Existing tests check useful component properties, but mostly reuse producer helpers. The endpoint encode/decode test is CUDA-gated, and its decoder merely clones the endpoint's scales; scale equality there does not exercise physical scale transport. Floating equality misses signed-zero bit differences; a roundtrip may hide matching forward/inverse permutation bugs. |

**Numerical contract to freeze.** For a weight stored logically as `[N,K]`,
require

`W[n,k] = gB * E4M3(scaleB[n,k//16]) * E2M1(code(state[n,k], alpha))`.

`gB` denotes the declared global decode multiplier at its actual scope
(for example expert/projection), not an implicit inverse scale. The existing
[ModelOpt codec](../glm53_nvfp4/modelopt.py) stores logical `[N,K/16]` scale
bytes plus `weight_scale_2`; its `dequantize` multiplies both. Activation
quantization needs an equally explicit `gA`, E2M1 payload and per-16 scale
contract. NVIDIA describes the same two-level decode-scale structure for
NVFP4. Its training options do not change this assignment's one-dimensional
weight groups. [NVIDIA NVFP4 format](https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html#data-format)

Apply global factors exactly once at the correct boundary. FC1 gate/up
factors must precede SiLU, multiplication and the declared clipping policy;
moving them after a nonlinear activation changes the function. FC2 factors
must precede the specified routed reduction. If a factor varies within K it
cannot be factored out of the dot product. Do not absorb a nonrepresentable
global factor into E4M3 and silently requantize it.

The source's E2M1 magnitude codes 0…7 represent
`0, 0.5, 1, 1.5, 2, 3, 4, 6`; bit 3 supplies the sign. There are 16 encodings
but 15 numerical values. The endpoint convention puts the even-K code in the
low nibble: `byte[j] = code[2*j] | code[2*j+1]<<4`. Comparing an E4M3 byte
whose *value* is legal E2M1 is not a packed-nibble comparison.

UE4M3 scale operands occupy bytes with a zero top bit; reject NaN `0x7f`
and invalid sign-bit encodings. Declare zero-block behavior explicitly.
Zero scales can represent zero blocks; nonzero-block division must not
silently underflow to zero. The physical scale is a multiplicative operand,
not a UE8M0 exponent, scale inverse, or an FP16 broadcast applied to weights
before an FP8 MMA. [PTX alternate formats](https://docs.nvidia.com/cuda/parallel-thread-execution/#alternate-floating-point-data-formats)

For K4 cyclic symbols `e[i]`, the inspected reconstruction is
`state[i] = e[i] | e[i-1]<<4 | e[i-2]<<8 | e[i-3]<<12`, with indices wrapping
within each 256-symbol tile. Packing writes MSB-first symbol/word bits and
swaps adjacent int16 words. Prove crossings at symbols 0/255, 15/16,
16-bit and 32-bit word boundaries, K16 scale boundaries, K64 MMA boundaries,
and TP shards. Treat arbitrary independent states as a decoder-unit fixture,
not necessarily as an encodable cyclic stream.

**Rounding counterexample.** The offline census implements the inspected
MCG law with `struct` binary16 arithmetic and independently checks the half
sum with integer arithmetic in units of `2^-24`. It compares the source's
tie policy with a hypothetical nearest-even E2M1 projection; both sides
canonicalize zero. It does **not** execute a GPU conversion or the repository
Torch implementation.

| Compander alpha (diagnostic, not candidate selection) | States checked | Differing nibbles | Example state 110 |
|---|---:|---:|---|
| 1 | 65,536 | 45 | MCG = −1.75; source `0xb` (−1.5), nearest-even `0xc` (−2) |
| 2 | 65,536 | 53 | Scaled MCG = −3.5; source `0xd` (−3), nearest-even `0xe` (−4) |

Thus blindly replacing P8's conversion with an E2M1 `.rn` conversion is not
bit-closure evidence. Freeze alpha and preserve the chosen codec law, or
version/re-encode the candidate and repeat all gates. Also test the actual
conversion's signed zeros, saturation, argument packing order and half-add
rounding. A 65,536-entry byte LUT is 64 KiB (32 KiB even nibble-packed),
exceeding the plan's 4 KiB table budget. CPU reference lookup storage is not
a permissible deployed runtime table. Enforce the cap over the complete
runtime lookup payload, not separately over arbitrarily split tables.

**Required MMA and lane ABI.** The intended dense warp instruction is

```text
mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3
```

Both operands must be E2M1. BF16 external activations therefore need a real
FP4 activation-quantization path for FC1 and the post-activation FC2 input.
Retaining P8's E4M3 activation producer does not satisfy this instruction.
CUTLASS documents the corresponding `MmaMXF4NVF4Op` with F32 accumulation,
scale vector size 16 and SM120/SM121 architecture-specific targets. Pin the
installed compiler/PTX/CUTLASS versions and exact supported target; “Blackwell”
alone does not specify the instruction family. This audit concerns the
SM12x register-fed warp path, not an assumed SM100 `tcgen05`/TMEM port.
[CUTLASS warp MMA API](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_warp.html#cutlass.cute.nvgpu.warp.MmaMXF4NVF4Op)

The following algebraic map restates the dense PTX fragment layout. Let
`g=lane//4`, `c=lane%4`; entries in each register run low bits to high bits.

| Fragment | Registers/lane | Logical coordinate of element i |
|---|---|---|
| A, i=0…31 | 4 × b32, eight nibbles each | `(g+8*((i//8)%2), 8*c+i%8+32*(i//16))` |
| B, i=0…15 | 2 × b32, eight nibbles each | `(8*c+i%8+32*(i//8), g)` |
| C/D, i=0…3 | 4 × f32 | `(g+8*(i//2), 2*c+i%2)` |

The audit checks coordinate coverage only. The candidate must prove its
EXL3-to-fragment transformation against these coordinates for every lane.
[PTX dense m16n8k64 fragments](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16864)

Each scale register holds four K16 bytes. For `4X`, byte selectors must be
zero; A's thread selector chooses a pair (0/1 or 2/3) in each quad, B's
chooses one lane in each quad. Prove both selection and byte order with
distinct scales per row/column and K16 group, not all-one scales.
[PTX block scaling](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-block-scaling)

For example, with thread selectors both zero, lanes `4*g` and `4*g+1`
provide A scales for rows `g` and `g+8`; lane `4*g` provides B scales for
column `g`. Byte j corresponds to K group `K_base/16+j`, j=0…3.
Changing selectors requires relocating the contributing scale words, not
changing the logical groups.

Checkpoint scale storage, shared-memory swizzle and register selection are
three separate layouts. The local `swizzle_block_scale` pads to multiples
of 128 rows and four scale columns, reshapes `(R/128,4,32,C/4,4)` and permutes
to `(R/128,C/4,32,4,4)`. This utility's roundtrip does not prove that a
particular TMA/copy atom or lane selector consumes it correctly. Export a
mapping `(expert, projection, n, k16) -> global offset -> shared offset ->
lane/register/byte`. Test gate and up separately and repeat for FC2, whose
K is the intermediate dimension. Re-derive mappings if FC1 swaps A/B.

**Staging and the critical path.** Require an actual executed chain:
K4 stream/scales in global memory → validated copies into shared memory
(or a declared direct register-load route) → cyclic window extraction →
MCG/projection → packed E2M1 register fragments → scaled MMA → epilogue.
For each allocation and pointer, record semantic dtype, storage dtype,
byte capacity, stride, alignment, ownership and lifetime. A CuTe FP4 tensor,
a TMA tensor-map descriptor and an MMA register fragment are distinct.

P8's descriptor-only tensors alias stream storage. This can be valid only
while those descriptor data are never consumed as decoded weights. A P4
rewrite that enables the ordinary FP4 TMA consumer could multiply raw trellis
bits. Trace all FC1/FC2 branches and demonstrate that descriptor carriers
are either correctly populated or never dereferenced. Guard allocations and
use data-sensitive tests; dtype assertions cannot catch this error.

Scale copies need correct bounds and semantic types in both logical and
repacked planes. P8's one-byte sentinel is not a valid scale plane for a
consumer that reads per-row scales. Verify 16-byte vector-copy alignment,
tail masks, padding initialization, shared bank conflicts, barrier phases,
copy completion before reads and completion before buffer reuse. A
`uint32` view does not by itself prove alignment. Avoid porting P8's K32 byte
selectors to P4's K64/4X operands.

The inspected P8 consumer computes decoded B registers before its dependent
MMA, inside the same K loop. Separate loader warps and double buffers can
overlap **memory staging**; they do not prove decoding is hidden under MMA.
Overlap may come from independent fragments, later K tiles or other warps,
but register lifetimes and issue bandwidth can erase the benefit. Treat
“prologue hidden under MMA” as a measured hypothesis, not a property of
fusion. G7 compares identical decoded operands across predecoded, serialized
and pipelined controls and requires timeline/instruction evidence.

**Cost ledger.** These are derived logical counts, not measured transfers or
performance predictions.

| Item | Derivation / required measurement |
|---|---|
| One 16×16 K4 tile | 128 stream bytes + 16 scale bytes = 144 bytes; 4.5 bpw before global factors, tables, alignment and metadata |
| One 64×8 B operand | 256 payload + 32 scale bytes = 288 unique logical bytes |
| One 16×64 A operand | 512 payload + 64 scale bytes = 576 unique logical bytes; activation generation traffic is additional |
| One dense m16n8k64 | 16,384 FLOPs counting multiply/add separately; same logical K span needs two m16n8k32 issues |
| General ideal issue count | `ceil(M/16)*ceil(N/8)*ceil(K/64)` per projection, before scheduling/padding duplication; count gate and up separately |
| Source MCG P8 template | 53 PTX operations per eight state labels: 8 mul, 8 lop3, 8 prmt, 8 add, 8 and, 6 shr, 4 cvt, 3 mov; excludes shared loads, address work, shuffles, scales and MMA; not a P4 or SASS count |
| Minimum single-fragment register contents | 4 A + 2 B + 4 accumulator + 2 scale b32-sized registers/lane, before selectors, decode temporaries, addresses and additional live fragments |
| P8 resource hints | `load_register_requirement=32`, `mma_register_requirement=232`, fixed staging depths and an `occupancy=1` planning input in dynamic.py; these are source settings, not ptxas allocation or achieved residency |
| P4 runtime tables | Total lookup payload ≤4,096 bytes; identify all constant/global/shared tables, per-CTA replicas, initialization traffic and any lookup spills |
| Actual resources | **UNKNOWN — needs verification:** registers/thread and warp role, spill loads/stores, local bytes, static/dynamic shared bytes/CTA, resident CTAs/warps, occupancy limits, active-lane fraction, tensor/ALU issue utilization, barrier stalls, DRAM/L2/shared bytes |

K4 plus E4M3/16 has the same leading 4.5-bpw weight rate as an ordinary
predecoded NVFP4 endpoint. P8's 4.25-bpw scale geometry cannot be reused as a
P4 rate claim. Count any retained decoded endpoint as an additional payload.
Separate checkpoint/file bytes, resident VRAM and bytes moved per request;
include globals, tables, padding, duplicate scale layouts, graph workspaces,
routing arrays, FC1 intermediates and route-slot reduction buffers.

For an actual CTA, compute residency from the limiting register allocation
(including granularity and warp redistribution), shared-memory allocation,
threads/warps, architectural block limit and persistent-grid scheduling.
Report achieved occupancy as well. Do not turn comments about “two blocks”
or a stage count into measured occupancy. Table-free decode still has ALU,
shuffle, dependency and register costs.

**Completeness conditions for product language.** A scoped “native P4
arithmetic kernel” requires G0–G3: preserved procedural stream semantics,
bit-exact independent operand closure, actual `mxf4nvf4` execution and both
projection paths. A serving claim additionally requires G4–G6, the full
claimed layer inventory and topology, and attribution resolution. Successful
quality closure need not mean better quality than stock NVFP4.

Prove FC1 gate/up pairing, global scaling, SiLU/clamp ordering, FC2 input
quantization, down projection, router-weight placement and routed sum.
Cover every dispatched shape, including split large-M paths. “Materialized”
FC1 activations or route outputs are not dense weight materialization; label
them and count their cost. Runtime traces must exclude BF16/FP16 decoded
weight tensors, CPU weight offload, and dense weight-matmul fallbacks for
the claimed projections. Scalar/half register arithmetic during MCG decode
is not itself a dense weight tile.

P8 selects its split path at `m*topk >= 36*experts`, allocates scratch on
calls, and optionally writes `[m*topk,hidden]` route outputs followed by a
top-k sum. P4 must validate its own crossover, graph capture/replay allocation
rules, scratch reset, lifetime and reduction order. Deterministic local
reduction does not prove deterministic routing or TP collectives. Verify
rank-local shape/scale slicing, all four rank identities, expert ownership,
FC1 concatenation, FC2 collectives and reduction dtype/order. TP4 must not
silently stand for a different EP/DCP regime.

“NVFP4 speed class” is an operational performance label, not an ISA
classification. This audit proposes at least **90% of matched stock NVFP4
throughput in every declared cell in each of five paired runs** (G8), after
correctness and integration gates. Report the ratio and workload with the
phrase. Native instruction selection, theoretical issue count, a large
GEMM, prefill alone or C8 throughput cannot establish decode/C1 speed class.
A separate “hidden prologue” claim also requires G7; it is not logically
necessary for a measured throughput claim if the overhead is visible but
the declared throughput gate still passes.

**Predeclared falsifiable tests.** All device/integration tests below are
**Not tested** and require a separately authorized device phase. They do not
authorize opening protected roles or operating production services. These
are proposed audit gates, not an assertion that the original plan already
specified these numerical intervals. Freeze implementation, fixtures,
analysis and this protocol before device results; amendments retain failed
attempts and start a new candidate record.

| Gate | Fixture / measurement | Acceptance and stop condition |
|---|---|---|
| G0 — Freeze and eligibility | Pin candidate/reference commits, complete dependency/image/compiler hashes, model/config/tokenizer identities, alpha/rounding/zero rules, TP/EP/DCP, activation/global-scale policy, dispatch matrix and runtime table/byte inventory. Independently approve provenance. | Stop on any unresolved execution-critical identity, table budget violation, unsupported target, missing FC1/FC2 branch, prohibited weight fallback, or unapproved data/service/GPU access. |
| G1 — Independent CPU bit closure | A separately authored verifier must read only frozen payload bytes and the public contract; no import of candidate packer/decoder/LUT/permutation helpers. Compare all 65,536 raw states, legal scale bytes, cyclic fixtures, low/high nibbles and logical-to-fragment maps. Include zeros, negative zeros, midpoints, extrema, cross-word/tile/shard boundaries and unequal neighboring scales. | Zero byte mismatches in packed E2M1 and physical scales. Raw-state census and packed-stream closure are separate outputs. Reject invalid scale/shape/metadata fixtures. Stop at first mismatch; preserve the complete mismatch artifact. CPU reference tensors are controls, never the product weight path. |
| G2 — Device bit/ISA/safety closure | Standalone bounded prologue capture to raw nibble/scale buffers for every lane; compare against G1. Basis/impulse vectors address each MMA coordinate. Negative controls deliberately swap nibbles, scales and selected lanes and must be detected. Save generated PTX, cubin and SASS for every specialization; correlate actual launches to binary hashes. Run memory/race/init/sync checks on synthetic buffers. | Zero raw mismatches and sanitizer errors; all negative controls detected. Actual weight arithmetic must use dense `mxf4nvf4` K64/E2M1/UE4M3-4X in both paths. Stop on wrong opcode, descriptor-carrier consumption, invalid selector, missing scale read, spill-mediated dense weight buffer, deadlock or nonfinite output. A string found in unused PTX is insufficient. |
| G3 — Arithmetic and routed function | First isolate FC1 gate/up and FC2, then compose quantization, activation and routing. Compare with independent scalar/FP64 arithmetic over frozen decoded operands and a separately predecoded packed-NVFP4 device control. No BF16 weight GEMM is needed. Use exact powers-of-two/basis cases and random signed/cancellation cases; M={1,7,16,17,32,33,64,65,128}, gate/up (N,K)=(512,4096), down=(4096,512), top-k=8, 288 experts with empty, one-hot, skewed and uniform routing. Force both supported branches. Unsupported shapes must reject explicitly. | Exactly representable one-term/basis outputs must match bitwise. For general raw FP32 dots require `abs(error) <= 1e-6 + 8*gamma(K)*sum(abs(a*b))`, `gamma(K)=K*2^-24/(1-K*2^-24)`, with a/b including the exact decoded scales being compared. For composed BF16 output require every element within `1e-3 + 0.02*abs(reference)` and relative L2 error ≤0.005, with zero-reference denominator floored at 1e-12. Report max errors, not only averages. Stop on any exceedance/nonfinite value; no post-result tolerance widening. |
| G4 — TP4/serving/graphs | Four rank-distinct synthetic shards with expert/scale sentinels; prove collectives and inventory. Exercise eager, capture, replay with changed input values/routes, and replay after scratch poisoning. Cover both sides of every dispatch/capture boundary, including the P8 crossover if reused: M=1295,1296,1297 for top-k=8/E=288. Then verify all claimed layers in an isolated serving instance using authorized inputs. | Eager/replay and split/monolithic outputs must satisfy G3; any stricter bit parity declared by the candidate must also pass. Loaded source/binary hashes and logs must show the actual P4 path for every claimed layer/rank. Stop on stale buffers, missing/duplicated routes, hidden fallback, mismatched shard geometry, capture failure or mixed topology. |
| G5 — Five-run determinism | Five separately initialized TP4 processes, same candidate, seeds, routing/tokens, library algorithms and exact workload order. Within each process repeat eager and graph replay. Hash raw outputs, route IDs/weights and logits at the agreed precision; compare per mode and across eager/graph. | All five corresponding output hashes, and all declared eager/graph parity hashes, identical. Keep every run. One bit mismatch fails determinism even when G3 tolerance and generated text pass. Diagnose local reduction versus TP collective versus routing; no dropping the first run. |
| G6 — Integrated KLD closure | Only already-open, explicitly authorized conditional-fit inputs; a role-safe runner pins teacher precision/revision, exact token IDs, masks, positions and all causal rows. Compare native P4 with independent pseudoquant under the **same FP4 activation** and global-scale boundaries, matched topology, KV, MTP and intervention inventory. | Proposed interval: absolute difference in mean teacher-to-student KLD ≤1e-5 nats/prediction and per-window difference ≤1e-4; report full-row and first-64/rest diagnostics. Stop on misalignment, missing rows/windows, nonfinite logits, or exceeded interval. This is engineering closure only, not untouched quality qualification or superiority. Selection, confirmation, final and the 28 reserved logits remain excluded. |
| G7 — Resource and overlap attribution | Synthetic identical operands, shapes and cache policy; controls: predecoded NVFP4, serialized decode+MMA, candidate pipeline, isolated decode. Keep tiles/occupancy matched where possible; report differences otherwise. Measure ptxas resources, SASS issues, full allocation/traffic ledger, tensor/ALU activity, waits, spills and per-tile producer/consumer timing. Five paired runs; profiler runs separate from timing runs. | “Hidden” requires pipelined time ≤1.05×predecoded time and added time ≤10% of isolated decode time per equivalent work in every cell where claimed, plus an instruction/timeline explanation of overlap. If timings are below resolution or controls change residency, stop the overlap claim pending a better controlled test. Unpredicted local-memory traffic or resource overcommit stops performance qualification for diagnosis. |
| G8 — Prefill/decode speed | Matched stock NVFP4, candidate, and pinned exact EXL3; P8 only as a separately labeled boundary comparator. Five cold **process starts**, followed by 3 fixed warmup bursts and 10 timed bursts/cell (C simultaneous requests per burst). Rotate arm order P/N/E, N/E/P, E/P/N, P/E/N, E/N/P. Prefill: input lengths 512/2048/8192 at C1/C4 with one output token, separate prefill tokens/s and TTFT. Decode: fixed 2048-token prefix, 256 forced decode steps at C1/C4/C8; report C1 tokens/s, inter-token latency and aggregate tokens/s separately. Run graph on and off; state exact DCP/EP, KV dtype, scheduler, power/clocks and GPU identities. | For each cell, each paired run's median throughput P4/stock ≥0.90; additionally TTFT and decode p95 inter-token latency ratios ≤1/0.90 for the applicable cells. Gate graph modes separately. Exact EXL3 is reported without assuming a win. Stop the speed-class claim on any failed cell, memory fault/OOM, mismatch in graph/topology/input/output lengths, thermal/clock regime drift, or failed correctness gate. Retain all failures and raw requests; no best-of-five selection. |

For G1–G4, freeze synthetic generator seeds `20260904`, `20260905` and
`20260906`, one fixture per seed per listed shape/routing case, and the
generator revision plus output hashes before device execution. Generate
weight fixtures from cyclic K4 edge streams; reserve independent raw states
for prologue-unit tests. Cycle unequal valid scales through each K16 group;
basis tests cover every legal scale byte, including zero/subnormal/extreme
values. Malformed shape, missing scale, nonzero scale-MSB and NaN fixtures
must reject before launch. Bound each standalone device test to 60 seconds
after compilation; a timeout stops promotion and preserves the attempt.

Before G8, freeze the stock kernel/backend and exact EXL3 revision, supported
context/batch matrix, concurrency semantics, graph configuration and hardware
operating envelope. If 8192-token inputs or a listed cell are unsupported,
declare the narrower claim before timing; do not omit that cell after failure.
Use identical replay tokens for isolated mechanism timings to hold routing
work constant. Separately report ordinary generation if quality-dependent
routes differ. Exclude model loading/JIT/warmup from steady-state rates but
publish startup and end-to-end costs separately. Include tokenization and
queueing consistently in TTFT.

The independent performance unit is the paired cold-process run. Its 10
timed bursts, requests, tokens, layers and experts are subsamples. Publish all five
ratios and descriptive spread; this minimum engineering gate does not imply
a population confidence claim. Performance on synthetic tiny tensors cannot
qualify TP4 serving. Holdout-based quality qualification would require a new
authorized sealed protocol; this audit neither opens nor schedules it.

**Attribution and licensing.** This is a provenance inventory, not a grant
of rights inferred from a successful test.

| Component | Observed attribution / terms | Required before distribution or a provenance claim |
|---|---|---|
| ExLlamaV3 / ExL3 | MCG constants, state interpretation and tile ordering attributed to turboderp/contributors; [LICENSE.exllamav3](../LICENSE.exllamav3) retains MIT terms | Retain copyright/license and identify exact upstream file/revision and ported ranges. An immutable upstream revision for these particular borrowed portions is not established by the inspected headers. |
| KQuant / QSRT | [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) names Luke Alonso/contributors and snapshot `104dd9233f850a3955f4991bea68b07dd34deeb8`; `sqg_xor_rank_permutation` identifies itself as a byte-for-byte port | The local notice reports no LICENSE in that snapshot. Upstream rights were not independently resolved here: **UNKNOWN — needs verification**. Record source files, hashes and permission/terms; do not label copied material MIT/Apache by inheritance. |
| B12X `w4a8_trellis`, lane/staging helpers and split phases | B12X Apache-2.0 text retained at [LICENSE.b12x](../runtime_patch/b12x_h16/LICENSE.b12x); local notices identify QSRT SQG-XOR-Cheb-T12 ancestry | A changed MCG arithmetic law does not erase inherited pipeline, lane-map or T12 provenance. Retain B12X and QSRT notices, identify modifications, and resolve copied portions' terms separately. Hash-pinned phase transformer inputs do not establish the rest of the dependency tree. |
| This repository and the audit | Repository [LICENSE](../LICENSE) is the ShapleyMCG source-available license; third-party components explicitly retain their own terms | Preserve the existing notices and distinguish new audit work from borrowed algorithms. No blanket relicensing or “from scratch” claim. |

**Checks performed and replay.** Run from this worktree:

```bash
python3 -I -B evidence/opened/codec-v2/p4-astra/audit_static_checks.py
git diff --check
```

The diagnostic run passed 3 tests, syntax-compiled 15 explicitly listed source
files without executing them, checked 65,536 half sums using two models,
checked the audit's A/B/accumulator coordinate coverage and produced the
rounding census and P8 source-instruction counts above. The receipt embeds
the deterministic JSON output and hashes the script. These are safe offline
checks, not execution of the production code or independent product
reproduction.

Dependency probes of `/usr/bin/python3` and the existing
`/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53/.venv/bin/python`, with isolated
Python and CUDA hidden, found no Torch, pytest, NumPy or safetensors modules.
Therefore the repository unit suite was **Not tested**; no packages were
installed. Running its CUDA-gated tests would also exceed this assignment.
The report records pending gates instead of converting this environment
limitation into a passing codec result.

Public references above were consulted as PTX ISA 9.3, the current CUTLASS
warp API, and Transformer Engine's NVFP4 format documentation. They establish
documented operand requirements, not the installed toolchain or a device
result. All candidate measurements, device binaries, full stream bit closure,
role verification and serving outcomes remain **Not tested**.
