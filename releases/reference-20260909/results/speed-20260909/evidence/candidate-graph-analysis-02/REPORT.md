# Candidate FP8 serving graph profile

All four planned cells completed on retained image sha256:9bb99e0b47f00c4ccf77f2d4be77bb7c3ce8d6b5636169aa1f1fddda040ab45e. Runtime/source receipts match TP4/DCP4, MTP3 probabilistic, NVFP4 KV, graphs, maxseq24, batch4096, four300W GPUs. Original production and clients stayed stopped. These are instrumented diagnostic traces, not new throughput claims.

| Profile | Graph kernel events mapped | Non-graph kernel events |
|---|---:|---:|
| 8K C1 decode | 1020936 | 31624 |
| 8K C4 decode | 429400 | 16400 |
| 32K cold prefill plus setup | 214348 | 238940 |
| 64K cold prefill plus setup | 219208 | 350852 |

All1,883,892graph-associated events map via explicit process-scoped clone/original node IDs to captured topology. All6,368graph receipts succeeded. Both decode clients report zero errors and no underfill/capacity/warmup-timeout flags. Raw capture timings include start/stop RPC latency; nominal1.5seconds is the controller sleep between profiling RPCs, not an exact measured trace duration.

Target P8 direct FC1/FC2 kernels are present in both decode traces. P8 grouped FC1/FC2 kernels are additionally present in both prefill traces. The serving adapter directly dispatches P8NativeTPMoE with use_a16=False; missing target runtime raises an error. Marlin kernels are recorded separately and must retain their layer-attribution boundary. Four representative compiled target regimes were independently checked for QMMA.SF.16832.F32.E4M3.E4M3.E8; no W4A16 target replacement was made.

Do not remove waits from this evidence. Many graph edges have non-default dependency/port types, which the analyzer preserves without interpreting them as full-completion barriers. Some default-edge kernel timestamps overlap by approximately1.0-1.44microseconds in C1; keep these anomalies instead of treating them as dependency violations or deleting edges. Cross-rank, eager-prefill and external-stream dependencies remain incomplete. Summed durations are not wall time or attainable speedup fractions.

Next bounded candidate comes from actual grouped FC2 launch geometry:256CTAs,128threads, registers199(K4)/216(K5), reported dynamic shared34816/38912bytes and executed partition102400bytes where available. These resources admit2blocks per188-SM device, or376CTAs, but current grid exposes only256. Test376CTAs with unchanged task stride/ownership and unchanged FP8/Hadamard/orderedK512 math. More CTAs might reduce imbalance or hurt locality; CPU task-coverage/source-scope proof passed and incremental component test is running. This does not assume the300C1/20000prefill targets are reachable from this change.

All raw Nsight reports and graph records are indexed by graph-profile-preservation-02/INDEX.json. Exported SQLite and these analyses are additional artifacts, hashed separately here. Earlier failed profiling attempts remain preserved.
