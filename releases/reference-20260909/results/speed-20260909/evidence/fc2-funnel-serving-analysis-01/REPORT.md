# Direct FC2 K5 funnel serving screen

Runtime and changed-source hashes matched the pinned image sha256:9bb99e0b47f00c4ccf77f2d4be77bb7c3ce8d6b5636169aa1f1fddda040ab45e. TP4/DCP4, MTP3 probabilistic, graphs on all four ranks, NVFP4 KV, maxseq24, batch4096, 4x300W. Fixed prefix tmxrepeatabc and temperature0 measured decode; keep separate from earlier model-default rates. One fresh server, one pass.

| Metric | FC2 funnel | Prior B: FC1 funnel | Prior A: no funnel |
|---|---:|---:|---:|
| 32K prefill t/s |8179|8197|8058|
| 64K prefill t/s |8115|8136|8022|
| 8K C1 decode t/s |193.027|184.537|193.229|
| 8K C4 decode aggregate t/s |321.903|332.309|332.192|

C1 is +4.60% versus B but effectively tied with A (-0.10%); C4 is -3.13% versus B. Prefill is within0.3% of B, whose prefill code is unchanged. This sequence does not establish a reliable overall served improvement. MTP acceptance differs (new C1 .52968, C4 .59896), order is temporal, and one server per image cannot separate route/thermal/numerical variation. Retain the component improvement as component evidence only; no adoption or full-suite qualification is justified.

KV startup capacity is23,411,764 tokens. Both profile receipts succeed, raw hashes match, fixed-prefix logs are present, and decode reports zero errors, no underfill/capacity flags and no warmup timeout. Raw metrics, runtime, source and launch files are retained. Supplemental archive completed and checksum verified. Production/clients remain stopped. Next work is a disposable graph-dependency recorder validation, followed by profiling if native IDs join correctly to Nsight; no waits or synchronization removed.
