| Measurement | Reference t/s | Candidate t/s |
|---|---:|---:|
| 32K prefill | 8407.000 | 8439.000 |
| 64K prefill | 8407.000 | 8451.000 |
| 8K C1 decode | 222.100 | 212.098 |
| 8K C4 decode aggregate | 318.560 | 325.083 |
| 16K C1 decode | 216.636 | 208.708 |
| 16K C4 decode aggregate | 319.157 | 353.532 |
| 0K C1 decode | 204.611 | 195.372 |
| 0K C4 decode aggregate | 327.670 | 333.729 |

FP8 FC1 task-count grid candidate; TP4/DCP4, MTP3, CUDA graphs, NVFP4 KV, four GPUs at 300W. Each cell has a verified cooldown. Single exploratory runs; output trajectories, MTP acceptance and sustained clocks can differ. No adoption or publication.
