# Historical evidence classification

The `historical/` tree is preserved for auditability, not pooled as one result
set. Its files include failed, invalidated, superseded, and valid attempts.

| Family | Classification | Reason |
|---|---|---|
| `run-v4-*`, `run-exact-v5-*`, `run-exact-v6-*`, `run-wave1-*` | invalid for rotation efficacy | Sparse-overlay weights could be overwritten by later carrier shards under the ordinary safetensors loader. |
| `run-exact-v7-*` | diagnostic only | Sign/permutation invariance tests isolated the loader defect and signed-zero handling; they were not candidate-selection evidence. |
| `run-exact-v8-stock-humming-indexed` | valid control | Stock checkpoint loaded through the same InstantTensor/Humming path as candidates. |
| `run-exact-v8-l3-identity-humming-indexed` | valid diagnostic | Identity-basis GPTQ comparison at layer 3. |
| `run-exact-v8-l3-had16-all-humming-inner-indexed` | valid conditional-fit evidence | Fixed H16 on gate/up and down at layer 3; statistically qualified versus same-loader stock. |
| `run-exact-v8-l3-learned-all-humming-inner-indexed` | valid conditional-fit evidence | Learned rotation at layer 3; failed versus fixed H16 and did not qualify versus stock. |
| `run-exact-v10-l3-had16-qwen256-indexed` | valid conditional-fit evidence | Exact Qwen 256-sample recipe; mean improved versus stock but its interval crossed zero. |
| `exact-v8-stock-humming-w4a4*` session logs | failed runtime attempts | Native Humming FP4 activation path was not qualified; no W4A4 efficacy claim is made. |

The authoritative compact summary is
`results/current-kld-summary.json`. Raw files should only be interpreted with
this classification and the contemporaneous decisions in
`DECISIONS_GLM53.md`.
