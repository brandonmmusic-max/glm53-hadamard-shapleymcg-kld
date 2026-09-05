# P8 small-M captured-graph closure

Decision: **advance to opt-in integrated TP4 GLM serving.**

The corrected comparison used the deployed monolithic M1 path as the baseline
and the route/slice small-M path as the only changed arm. Both arms received
identical input, route, route-weight and dense-reference bytes on each GPU.

| Physical GPU | Monolithic graph median | Small-M graph median | Reduction | Speedup |
|---:|---:|---:|---:|---:|
| 0, Max-Q | 0.956304 ms | 0.144832 ms | 84.8550% | 6.6029x |
| 1, Workstation | 0.771760 ms | 0.116480 ms | 84.9072% | 6.6257x |
| 2, Max-Q | 0.961824 ms | 0.145120 ms | 84.9120% | 6.6278x |
| 3, Workstation | 0.775776 ms | 0.118288 ms | 84.7523% | 6.5584x |

Every arm captured successfully, replayed five times with one output hash,
matched its own eager output, and matched the opposite arm bit for bit. Each
timing median contains 100 CUDA-event graph replays after five warmups. The
minimum 84.7523% reduction clears the preregistered 50% advancement threshold.

## Preserved invalid diagnostic

The first graph command accidentally used the probe's old default
`materialized` mode for scheduler-off while scheduler-on selected small-M.
That compared different arithmetic graphs: materialized FC2 accumulates full K
in FP32 and rounds once, while deployed monolithic M1 and small-M preserve an
ordered BF16 boundary after each K128 slice. Their output hashes correctly
differed. Those files remain under `raw/` and are explicitly invalidated by
`INVALIDATED_FORCED_MATERIALIZED.json`; they are not used in the table.

The probe now defaults to monolithic, records requested and resolved modes,
uses a dedicated post-construction CUDA RNG for payload identity, records all
payload hashes, and rejects `--small-m-scheduler --mode materialized`.

## Interpretation boundary

This is a synthetic, single-layer M1 CUDA-graph closure. It proves the
optimized fused codec path can be captured and retains its device-time gain on
all four GPUs. It is not full-model tokens/s or KLD. The existing 20.69 versus
91.26 tokens/s P8 product failure and stopped Shapley allocation remain
unchanged until a newly frozen integrated comparison passes.

No protected role or teacher logits were opened. No LDLQ or BlockLDLQ was
used. P8 remains K4 procedural MCG to fully scaled E4M3 with physical
UE8M0/32, using `mxf8f6f4`; that ISA issues twice the MMA instructions of
NVFP4 for equal K.

Machine-readable plan, addendum, invalidation and raw cells are in
`evidence/opened/codec-v2/p8-smallm-graph-v1/`.
