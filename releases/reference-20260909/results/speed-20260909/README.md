# September 9 TrellisMX FP8 speed experiments

Selected runtime: **token-map-hoist reference**, local image identity `sha256:ca6b80188dce154b91f49108b7d87792d2ba6328935afc71b44d1c0e6f6a1adf`. TP4/DCP4, MTP3 probabilistic, CUDA graphs, NVFP4 MLA KV, 24 sequences, batch4096, four RTX PRO6000 GPUs at300W. Expert math remains E4M3 FP8 with UE8M0 scales. Checkpoint unchanged.

All rates are tokens/sec; C4 is aggregate. Single exploratory server runs; MTP acceptance/output trajectories and sustained clocks differ. Later cooled screens require90seconds minimum idle and all GPUs<=55C continuously30seconds before each cell. Earlier runs retain their original protocols; do not silently treat them as cooled replications. No independent replication or superiority claim. No measurement reached300t/s C1 or20000t/s prefill.

| Measurement | Selected reference | Task-count | Selective grid-floor |
|---|---:|---:|---:|
| 32K prefill | 8407.000 | 8439.000 | 8423.000 |
| 64K prefill | 8407.000 | 8451.000 | 8448.000 |
| 8K C1 decode | 222.100 | 212.098 | 214.588 |
| 8K C4 decode aggregate | 318.560 | 325.083 | 326.620 |
| 16K C1 decode | 216.636 | 208.708 | 217.293 |
| 16K C4 decode aggregate | 319.157 | 353.532 | 343.859 |
| 0K C1 decode | 204.611 | 195.372 | 185.422 |
| 0K C4 decode aggregate | 327.670 | 333.729 | 325.106 |

## Selected reference extended context

| Measurement | Tokens/sec |
|---|---:|
| 32K prefill | 8457.000 |
| 64K prefill | 8443.000 |
| 128K prefill | 8323.000 |
| 32K C1 decode | 204.930 |
| 32K C4 decode aggregate | 329.343 |
| 64K C1 decode | 199.446 |
| 64K C4 decode aggregate | 330.096 |
| 128K C1 decode | 204.445 |
| 128K C4 decode aggregate | 334.086 |
| 8K C8 decode aggregate | 594.639 |
| 16K C8 decode aggregate | 602.965 |
| 0K C1 decode | 204.611 |
| 0K C4 decode aggregate | 327.670 |

Task-count and grid-floor were not measured at C1 contexts32K/64K/128K. Their shorter-context results cannot rank long-context C1. Reference KV capacity is23,562,091 tokens in the expanded run; separate zero-context reference sessions reported23,568,627. Use the explicit engine KV metric, not the benchmark's generic block-times-DCP estimate.

Reference is retained for balanced use: best observed C1 at0K/8K and effectively tied at16K. Task-count is a concurrent-throughput alternative, especially16K C4. Neither alternative has a meaningful prefill improvement established. Both alternatives passed172 exact representative K4/K5 component comparisons and FP8 compiled-opcode checks; these are not full-model quality evaluations.

`benchmark-index.json` indexes every discovered llm_decode_bench result in today's campaign, including earlier/negative/profile diagnostics. Results under profiling paths are diagnostic, not serving throughput qualification. `evidence/` includes raw benchmark JSON and associated logs/receipts plus candidate summaries/failures. Local host/path prefixes are redacted; `public-file-inventory.json` records both original and public hashes. Historical raw receipt hashes refer to the originals, not the redacted copies. Raw local archives are retained separately. KLD is published separately with cache dtype and capture-image provenance.
