# Grouped FC2 grid376 serving screen

Only grouped FC2 grid changes from 256 to 376; target FP8 MMA remains intact. One fresh server per candidate, temporal comparison, fixed prefix tmxrepeatabc and measured temperature 0. MTP3 probabilistic remains active; this protocol differs from earlier randomized-prefix/model-default runs.

| Metric | Prior grid256 | Grid376 | Change |
|---|---:|---:|---:|
| 32768 prefill | 8179.000 | 8325.000 | +1.79% |
| 65536 prefill | 8115.000 | 8240.000 | +1.54% |
| 8192 C1 decode aggregate | 193.027 | 199.718 | +3.47% |
| 8192 C4 decode aggregate | 321.903 | 334.304 | +3.85% |

Prefill direction agrees with the 3.4–4.4% component elapsed-time reduction, but the served increase is smaller and one server per image does not establish repeatability. Decode kernels are unchanged. C1 MTP acceptance changed from 0.52968 to 0.62162 and C4 from 0.59896 to 0.56771; do not attribute decode differences to the grid change. Temperature 0 does not remove all speculative execution variability.

TP4/DCP4, graphs on four ranks, MTP3 probabilistic, NVFP4 KV, maxseq24 and batch4096 receipts verified. Startup KV capacity: 23,418,300 tokens (prior 23,411,764). All eight candidate image source files matched recorded serving hashes. Decode has zero errors and no underfill, capacity or warmup-timeout flags. Component exactness evidence is separate from full-model quality, which remains unqualified.

Raw screen, source, launch configuration and retained image identity are in publication-handoff-grouped-fc2-grid376-supplement-01.tar.gz; checksum verified. Nothing published or adopted. Production remains stopped under the current user instruction.

Next: retain grid376 provisionally and investigate N64 direct-FC1 producer occupancy while preserving ordered FP8 MMA, the existing FP16 boundary and full H128 coupling. An additional epilogue launch and scratch traffic could erase its benefit. No split-K or W4A16 substitution.
