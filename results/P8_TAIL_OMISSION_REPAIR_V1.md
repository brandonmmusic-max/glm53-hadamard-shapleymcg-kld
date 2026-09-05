# P8 fixed-order KPool tail repair

Historical V1 result: **inconclusive because the intended intervention did not
execute on the tail rows.** This file is retained as failed-method evidence; it
must not be cited as a negative tail result. The corrected, device-closed V2
result is in `results/P8_TAIL_OMISSION_REPAIR_V2.md`.

The preregistered single-variable candidate kept the fixed compressed-pool
order and placed each request's 1-3 incomplete current-pool tokens inside the
2,048 columns consumed by B12X, displacing the same number of final history
entries. The P8 codec bytes, 4.25-bpw rate, all-42-layer native Tensor Core
path, index scores, topology, NVFP4 MLA KV, capture protocol, windows, and KLD
implementation were unchanged.

| Window/domain | Baseline KLD | Repaired KLD | Delta |
|---|---:|---:|---:|
| conditional-fit-0032 / general | 0.0974280 | 0.1229099 | +0.0254819 |
| conditional-fit-0021 / legal | 0.2015925 | 0.2015925 | -0.000000054 |
| conditional-fit-0090 / code | 0.0981727 | 0.0981701 | -0.000002631 |
| conditional-fit-0003 / reasoning | 0.0727031 | 0.0727031 | -0.000000002 |
| Equal-window mean | 0.1174741 | 0.1238439 | +0.0063698 |

The recorded V1 mean regressed by 5.4223%, but post-run row-diff inspection
showed that three of four windows changed only one final row and the fourth
also changed zero-tail rows. The patch inserted tail entries near column 2048
rather than directly after valid history. Therefore the preregistered verdict
`material-tail-cause-rejected` is mechanically invalidated, not rerolled.

The successful run used 5,072,629,760 raw bytes on NVMe and none on klcstore.
Two earlier host-only failures are retained: a recursive adapter before
container creation, then a missing timeout field before any request. Both
wrote zero raw logits; owned containers were removed and production restored.

Receipts:

- preregistration: `experiments/p8-tail-omission-repair-v1-prereg.json`
- successful plan: `experiments/p8-tail-omission-repair-v1b.json`
- evidence: `evidence/opened/codec-v2/p8-tail-omission-repair-v1/`
