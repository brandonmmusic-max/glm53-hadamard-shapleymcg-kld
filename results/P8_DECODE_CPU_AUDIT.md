# Native P8 decode: source audit and post-series profile plan

Status: structural, exploratory diagnosis; no performance attribution or
optimization qualified. Written after the first completed v2a3 speed pair,
before any diagnostic GPU profile. The frozen five-pair series continues
unchanged. This document does not authorize an overlapping GPU workload.

The first pair measured P8/EXL3 32K server prefill at 6,971/6,418 tokens/s
and C1 decode at 20.64304193362273/91.19875913641265 tokens/s. Both passed
four-rank FULL graph checks. These are observations, not five-run medians.
The benchmark hashes are respectively
`97bdf0c83cc823231d1f1a598eba367769bfa961221ec7e318808cc04a7cad20`
and `540a06d234aa11be955e4dc42c9bda4453ec3d9e100e2f1ef5b848ad13567112`.
P8 uses TP4/no-EP/DCP1; EXL3 uses TP4/EP4/DCP4. Attention, communication,
host work and MoE are rival contributors to this system-level difference.

P8 decodes MCG K4 into E4M3 and uses native `mxf8f6f4` m16n8k32: twice
NVFP4's MMA issue count for equal K. It is the quality product, not P4 or
proof that codec decode is hidden beneath useful MMA work.

## Audited identities

- Runtime commit `efd250b002e8da5a0e603253e0cd07ba6249c7b4`,
  `runtime_patch/p8_native_kernel.py` (the executed immutable runtime tree).
- Image `sha256:5da4ef3e814a71c6bcc47a7eb409a02fe4e3d5d867261f0b8e2e9b2d6ebb8ef8`.
- Installed `/opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/dynamic.py`,
  SHA-256 `2428e8a6abaec75348181e6afee37deb2edf7697c820dd762029a453c8b071e4`.
  Line references below refer to this installed source, not the unselected
  vendored runtime tree.
- Read-only CPU audit by GPT-6 Astra agent `p8_decode_cpu_audit`; no GPU
  launches, runtime edits, service changes or protected-data reads.

## Ranked hypotheses, not measured bottlenecks

| Priority | Source fact | Discriminating measurement |
|---|---|---|
| 1 | Deterministic scheduling groups all four intermediate slices into one expert-M task (`dynamic.py:3044-3056,4060-4090,4951-4952`). At actual M=1/topk8, at most eight useful compute CTAs execute the complete slice sequence. | Actual graph-padded M, active work, occupancy and kernel duration; do not assume C1 implies M=1 inside every captured graph. |
| 2 | Wrapper lines 324-357 create zeroed scratch on every invocation; only compiled code is cached. At M1 the three largest arrays total 23,814,400 bytes/layer, or 953.87 MiB/rank/token over 42 layers. | Device fill/memset nodes and elapsed time in graph replay. Graph capture may eliminate Python allocations, but not necessarily their device initialization. |
| 3 | Monolithic MCG helper computes both E4M3 halves then selects one (`dynamic.py:255-294,5516,5538,7498`). Adjacent N8 halves belong to different warps. | Actual SASS and executed instruction mix: compiler dead-code elimination may already remove some source-level redundant work. |
| 4 | General routing uses a resident cooperative grid, expert histogram/prefix work and five grid barriers before compute (`dynamic.py:2617-2626,3186,3197-3233,4025,4094`). | Routing/barrier cost versus useful compute. Existing direct-routing flag explicitly rejects trellis; it cannot safely be enabled as a shortcut. |
| 5 | Deterministic top-k reduction and possible ID dtype conversion add work per layer (wrapper line 321; installed `fused_moe/_impl.py:9910-9943`). | Kernel/cast counts and time, relative to attention and communication. |

The wrapper selects split materialization only when `M*8 >= 36*288`,
i.e. M >= 1296. Smaller shapes use the monolithic path. Forcing materialized
selects M64, not a proven M1 specialization. Static scale repacks are
constructor-only (wrapper lines 152-173), not per-token overhead.

## Decision before diagnostic measurements

1. Finish all five frozen speed pairs and analyze/publish them, including
   any failure. If either primary median does not beat EXL3, stop the
   allocation game under the existing rule. Do not substitute a faster
   exploratory kernel into that series.
2. In a fresh immutable diagnostic attempt, profile 16 warmed C1 decode
   steps on the unchanged P8 runtime using a dedicated synthetic or fit-only
   input, not qualification windows. Record source/environment identities,
   actual M, graph replay, fills/casts, MoE, attention and communication.
   Nsight Systems 2025.6.3 is installed; `ncu` is not currently on host PATH.
   Verify profiler availability and capture permissions before launch.
3. If available, profile one representative M1 monolithic kernel for
   occupancy, stalls, memory throughput, instruction mix, registers/spills
   and SASS. Profiler runs are diagnostic, never product throughput samples.
4. Select the first implementation change from measured dominant cost.
   If no candidate is dominant, report that uncertainty rather than claim
   a codec bottleneck. Do not change the encoder, checkpoint or arithmetic
   simply to make the performance trace more favorable. No LDLQ.

## Smallest candidate changes and correctness conditions

- Scratch: consider graph/stream-safe reuse and justified capacity reduction.
  The plain-scale ABI describes 591,872 bytes at M1, versus 2,433,024
  allocated; all specialized descriptor accesses must be audited first.
  A candidate route-tile bound is `min(E,P)+ceil(P/tile_m)`, P=M*topk;
  expert metadata remains E-sized. Neither change is yet proven safe.
- Decoder: select the required four-state half before procedural decoding,
  only if SASS confirms redundant work remains. Preserve MMA lane layout.
- Scheduling: independent output-column work or per-slice temporary outputs
  could expose more CTAs. Preserve sequential BF16 rounding after each slice
  (`dynamic.py:7928-7935`), not just deterministic final top-k order.

Before adoption: exhaustive 16-bit decoder-state byte equivalence, sliding
window/lane boundary tests, poisoned scratch and inactive-route coverage,
graph replay isolation, and bitwise outputs against the frozen runtime on
synthetic/fit-only data. A changed runtime needs its own device/full-model
closure and fresh preregistered speed comparison; the existing full-model
KLD result does not automatically transfer.

Existing ExLlamaV3, KQuant, QSRT and w4a8_trellis attribution remains in
force. This audit introduces no new codec port and no measured speedup.
