# P8 fixed-order KPool tail repair

Decision: **reject incomplete-tail omission as a material cause of the shared
approximately 0.11 forced-decode endpoint.**

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

The mean regressed by 5.4223%. Although three windows improved numerically,
their changes were effectively zero; the general window caused a material
regression. The paired four-window percentile interval for candidate-minus-
baseline was `[-0.00000197,+0.01911143]`. The preregistered verdict is
`material-tail-cause-rejected`; no reroll or alternative ordering was tried.

The successful run used 5,072,629,760 raw bytes on NVMe and none on klcstore.
Two earlier host-only failures are retained: a recursive adapter before
container creation, then a missing timeout field before any request. Both
wrote zero raw logits; owned containers were removed and production restored.

Receipts:

- preregistration: `experiments/p8-tail-omission-repair-v1-prereg.json`
- successful plan: `experiments/p8-tail-omission-repair-v1b.json`
- evidence: `evidence/opened/codec-v2/p8-tail-omission-repair-v1/`
