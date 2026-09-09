# FC2 decode instruction profile

Both K4/K5 real-weight M4 rank0 probes completed, finite warmup and profiled outputs. Unchanged row-alias376 image FC2; synthetic routes, not serving throughput. Historical C1 trace motivated selecting FC2.

NCU reports384CTAs x128threads on188SMs. K4/K5 registers143/165; dynamic shared22528/26624B plus1024B driver reservation. Both use65536B shared configuration and2block shared-memory limit versus3block register limit. Achieved occupancy14.64/14.81%. K4 duration29.06us; see raw K5 report. NCU replay durations cannot predict served speed.

All-sample stall distribution K4/K5: long scoreboard29.35/37.54%, dependency wait14.74/12.63%, barrier7.54/9.68%, selected30.17/26.24%. Dependency waits often occur at LOP3/PRMT/SHF; the stalled opcode need not be the latency-producing instruction. summary.json retains PCs and counts.

Action: investigate explicit function carveout to raise residency without changing arithmetic. Full-component correctness and timing must precede serving viability. No production restart or candidate adoption.
