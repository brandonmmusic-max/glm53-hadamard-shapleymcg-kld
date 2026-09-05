# EXL3 deterministic-output path audit

Date: 2026-09-05

Status: CPU-only, read-only source audit and diagnostic preregistration

Evidence level: exact-image static path audit; no device execution and no
performance or numerical qualification

## Decision

Do **not** use `B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1` as the proposed EXL3
determinism repair on the current Tail-V2 production image. The exact image
explicitly returns `False` for that option when `quant_mode == "w4a16"`, which
is the mode selected by EXL3. An off/on run would therefore be an inert-label
negative control, not a test of ordered EXL3 output.

Do **not** patch only `Caps(deterministic_output=True)`. The W4A16 workspace
plan hard-codes `deterministic_output=False`, and bind time independently
recomputes the request through the W4A16-disabled environment path. That
single-field patch would not establish a coherent end-to-end contract.

The exact production `trellis_t256` full-rotation path also does not use the
packed W4A16 tensor-core decode kernel's atomic fused-sum epilogue. It writes a
per-route FC2 buffer and invokes a separate top-k sum kernel whose source loops
over routes in fixed router order. This narrows and corrects the strongest
source-grounded rival stated in `EXL3_COLD_START_DETERMINISM_AUDIT.md`: an
atomic **output reduction** is not established for this production path.
Atomic route-packing operations, graph scheduling, collectives, upstream
kernels, and other arithmetic-order effects remain possible rivals. None is
identified as the cause by current evidence.

After the active baseline completes, the smallest useful diagnostic is one
predeclared window, `conditional-fit-0056`, under two independent cold starts,
with boundary hashes around the already ordered reducer. It localizes whether
the first differing bits exist before or after that reducer without changing
KV geometry. It is diagnostic only, not a determinism or KLD qualification.

No GPU, service, image, source, plan, score, checkpoint, or active quality slot
was changed for this audit.

## Exact evidence identity

All source below was read from the exact EXL3 Tail-V2 production image with a
CPU-only container invocation (`--runtime=runc`, `--network=none`,
`NVIDIA_VISIBLE_DEVICES=void`).

| Object | Immutable identity |
|---|---|
| Tail-V2 `v1b` plan | `d055d4db3563fa112326979c70da9ddc058abd5de6f67213aeee3c7308820beb` |
| Tail-V2 `v1b` amendment | `6d1cb6e9d9edad5e118fbd14da9df5e7dffb229a9c6ba4e6dd73c70c767e2a11` |
| EXL3 image | `sha256:af4e6a9ac0feb29ae2399ea45fad7db189594f0ff564c5bff3a691156c60b6b7` |
| Round-1 EXL3 launch specification | `0dd6d8fe2efe286cd3cc7b8798eb40795a8e861ba0f193b987126eb0de09f431` |
| Round-1 EXL3 container inspection | `43c61ca76abfd8e5bc7845d40b284c51930060b3dc721a956a19f4a5ff4e003d` |
| EXL3 runtime audit | `b5b1c199e7f6af55f86b3a350b29d04d0f0ddf532f098f1a10f598a6b0ac1e61` |
| Patched KPool source | `494192195da43c46d99a684555fc10fd13a19e89288cb9f51da2536ccdf1f251` |
| Conditional-fit role manifest | `b5d7e4524eb98ddfbd230a5d9a44de0dc5dbeb796c03e859898b5838e4463d14` |
| `b12x/moe/fused_moe/_impl.py` | `bc08e69ba855307000608d57eb64dc066c73330e797b4083195b2f4241da1309` |
| `b12x/moe/_shared/execution.py` | `e9d395871bd2ec4eae76b629c75ed38f10cc833a83d417bfc85e41af2431e63c` |
| `b12x/moe/_shared/kernels/dynamic.py` | `01a87c3e0e19ba3c7e4482a7dab318300493d3ee3dcd14bf87460e3ef149b46f` |
| `b12x/moe/_shared/kernels/w4a16/kernel.py` | `d7d2302ce507e4117b8d43d6359fad6c4b6ec1470b257d9ccba88fa8c3b40b80` |
| `vllm/model_executor/layers/quantization/exl3.py` | `ae92592ea8fcd249978134357ea3cd2510fe2aa9bdb1d1a3ab02afdbaeb39f45` |

The launch does not set `B12X_DYNAMIC_DETERMINISTIC_OUTPUT` or
`VLLM_EXL3_MCG_A8MX`. Its measured regime is attention
`B12X_MLA_SPARSE`; KV `nvfp4_ds_mla`; MoE `b12x-exl3-full-expert`; BF16
activations; EXL3 4.0 bpw; TP4/EP4/DCP4; CUDA graphs enabled; MTP off.

## Exact flag and call path

1. `exl3.py:3441-3477` plans rank-sliced EXL3 as `quant_mode="w4a16"`,
   `source_format="btx"`, and prepares the projected trellis layout.
2. `exl3.py:4690-4702` constructs `api.Caps` with
   `quant_mode="w4a16"`. It does not request `deterministic_output`.
3. `exl3.py:4863-4874` binds and runs that plan for the trellis serving path.
4. `_impl.py:1263-1270` implements the only matching environment switch.
   `_dynamic_deterministic_output_enabled()` returns `False` before reading
   `B12X_DYNAMIC_DETERMINISTIC_OUTPUT` whenever normalized mode is `w4a16`.
5. `_impl.py:2900-2925` constructs the W4A16 `_TPCoreWorkspacePlan` with
   `deterministic_output=False` hard-coded.
6. `_impl.py:7669-7737` recomputes the requested value at binding, passes it
   into planning, then forwards the resulting value to the common binding.

Therefore the existing environment option has no call path to an ordered EXL3
W4A16 reducer in this image. The public `Caps` field is not, by itself, a
complete alternative call path for W4A16.

## Which reducer actually executes

The exact-layout predicates matter:

- `w4a16/kernel.py:12480-12489` permits the atomic tensor-core decode shortcut
  only when `weight_layout == "packed"`.
- EXL3's prepared source is `btx`/projected trellis; the runtime W4A16 path
  resolves it as `trellis_t256`, so that packed predicate is false.
- `w4a16/kernel.py:12494-12518` separately permits the mapped, full-rotation
  `trellis_t256` layout on the direct top-k route path.
- The atomic fused-sum store at `w4a16/kernel.py:12531-12539` and
  `12822-12832` is guarded by `use_tc_decode`. With the trellis layout it is
  false, so FC2 writes `intermediate_cache13`, a per-route buffer.
- `w4a16/kernel.py:13127-13145` then invokes
  `_w4a16_topk_sum_launch_flat` for full rotation.
- `W4A16TopKSumKernel` loops over `range_constexpr(self.topk)` at
  `w4a16/kernel.py:8354-8365` (and at `8504-8510` in its coupled branch),
  accumulating route values in fixed route order.

This is a static call-path result, not device proof that every launch and
intermediate is bitwise deterministic. In particular, it does not establish
whether the first cold-start difference originates in FC1, activation, FC2,
route metadata, the reducer, a later collective/layer, or outside the MoE.

## Existing observations retained without reinterpretation

The completed cold starts remain:

| Cold repetition | True-decode KLD | Raw hashes equal repetition 1 |
|---|---:|---:|
| 1 | `0.031611840268931456` | self-reference |
| 2 | `0.031244056862183633` | `0/32` |
| 3 | `0.03104528212736431` | `0/32` |

The prior `v1a` versus `v1b` comparison also showed exact one-token-prefill
KLD and first per-row score divergence at decode row 19, 20, or 21. P8 repeat
1 versus repeat 2 was 32/32 raw-hash identical. Those facts establish
cold-start variability on this EXL3 serving path, not its cause.

## Risk assessment

| Candidate action | Expected effect in exact image | Performance risk | Semantic risk | Status |
|---|---|---|---|---|
| Set `B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1` | None for W4A16; the guard returns first | Expected none, but not measured | False-confidence risk: label says deterministic while path remains unchanged | Statically inert; do not use as repair |
| Set only `Caps(deterministic_output=True)` | Incoherent with W4A16 plan and bind-time recomputation | Unmeasured | May change metadata without changing the executing reducer | Reject as incomplete |
| Add a coherent W4A16 deterministic contract | Exact production trellis path already has per-route output and ordered top-k combine; incremental reducer cost may be small | **Unmeasured**; could change graph keys, scratch, compilation, or route selection | **Unmeasured**; requires closure against current logits | Future implementation only after localization |
| Instrument existing boundaries | Adds diagnostic stores/hashes and changes timing | **Unmeasured and not representative of production speed** | Intended to preserve values, but closure must be checked | One-window diagnostic only |

No throughput conclusion follows from source inspection. Even if the final
top-k loop is fixed-order, a modified build could change scheduling or graph
capture elsewhere. Report any future speed and output effects as unmeasured
until directly tested.

## Preregistered smallest diagnostic after the baseline

### Fixed arm and window

- Wait until the current baseline/campaign finishes; do not consume or replace
  any of its ten quality slots.
- Use `conditional-fit-0056`, the first entry in immutable role-manifest order.
  It was not selected by its observed KLD or cold-start delta.
- Keep image/model weights, request bytes, forced token IDs, seed, TP4/EP4/DCP4,
  attention backend, BF16 activation, CUDA-graph setting, and
  `nvfp4_ds_mla` KV geometry fixed.
- Run two independently started processes. Two is the minimum cold-start pair;
  failure to reproduce is inconclusive and does not pass determinism.

### Diagnostic-only observations

For decode rows 18 through 22, record value-preserving hashes at each MoE
layer for:

1. the MoE input;
2. the per-route FC2 buffer immediately before `W4A16TopKSumKernel`;
3. the reducer output immediately after that kernel; and
4. the final layer/MoE output boundary already available without changing KV.

Also record the resolved runtime predicates and launch identity:
`weight_layout=trellis_t256`, `full_rotation=true`, `use_tc_decode=false`,
`use_direct_topk_routes=true`, and the top-k-sum kernel/cache identity. Retain
the one window's diagnostic raw logits until the comparison receipt is written,
then follow the existing retirement policy.

### Decision rule written before execution

- Pre-reducer hashes equal but post-reducer hashes differ: the reducer or its
  immediate launch boundary is implicated; this still does not imply an atomic
  reducer.
- Pre-reducer hashes already differ: divergence originates upstream of the
  final top-k reducer; the reducer is exonerated as the origin for that run.
- Every MoE boundary matches but final logits differ: localize outside the EXL3
  MoE path.
- No cold-start difference: inconclusive diagnostic, because two starts have
  no power guarantee.

Only if the two-start diagnostic reproduces the difference should a second,
separately declared single-variable arm set `--enforce-eager` while retaining
the same KV geometry. That follow-up tests graph involvement; it must not be
reported as a production-speed comparison.

## Claim boundary

This audit establishes source reachability and excludes one specific proposed
control: the current deterministic-output environment flag cannot affect EXL3
W4A16. It also shows that the exact production trellis path's final top-k
combine is source-ordered rather than the packed-path atomic fused sum. It does
not establish bitwise device behavior, explain the observed KLD spread, prove
or disprove atomic activity elsewhere, qualify a patch, or measure cost.
