# Retained FC1 source-counter profile

Two targeted M4096 grouped-FC1 profiles completed on real rank0 K4/K5 checkpoint layers with synthetic routes. Image:sha256:0fb9c471ccb43bca9f0ca7b0041c26a95424f4aab43d83ac3ecb0fd7399a0906. No production baseline or new serving throughput measurement. Four GPUs retain300W limits; only rank0 was used for this component profile. Both probes produced finite outputs; this is not quality qualification.

Unlike the earlier profile, SourceCounters collected nonzero PC samples. COUNTER-SUMMARY.json records values and original units; SOURCE-HOTSPOTS.json preserves leading instruction addresses and counter values. Raw reports were exported with the same image's Nsight version.

In both K4 and K5, the largest single excess-shared-wavefront instruction is a4-byte LDGSTS to shared offset0x2000:9,358,776excess wavefronts and7,162,596excess global sectors at that instruction. The4-byte async copy and8192byte offset are consistent with the source's grouped M64 input-scale gather (a_payload_bytes=64*128, sfa_offset=a_payload_bytes). This source attribution is inferred from instruction shape/offset and source, not lineinfo correlation. Vectorized A-payload copies also appear among excess-wavefront sites.

Math-pipeline stall samples are concentrated at FP8 QMMA instructions among the leading sites. Wait and short-scoreboard samples appear at several instructions rather than identifying one removable barrier. Do not interpret stall samples, excess sectors, or Nsight heuristic estimated speedups as achieved end-to-end speed or a speed bound.

Next source investigation: input-scale gather traffic and its duplicated retrieval for each K64 half and FC1 output tile. Determine whether route-ordered scale materialization or reuse can remove repeated scattered4-byte accesses without adding more packing/synchronization than it saves. Preserve FP8 scales byte for byte, K32 boundaries, and original arithmetic. No new candidate is implemented or authorized to bypass component/serving checks by this profile.

Both profiling containers are stopped. Full plans, weight receipts, logs, source/SASS counter exports and reports remain local. No publication or checkpoint modification.
