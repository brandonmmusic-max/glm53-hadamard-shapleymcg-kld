# TrellisMX P8: latest measured serving results

Checkpoint: coupled 17-K5 / 25-K4, 42 routed layers, 4.6587417643 routed
stored bits/weight including metadata. Native P8 reconstructs E4M3 weights
for `mxf8f6f4` MMA: twice the NVFP4 MMA issue count for equal dimensions,
not native FP4-rate arithmetic. These are development/exploratory results,
not a universal model/runtime qualification.

## Reasoning profiles, September 7

| Profile | Official credited score | Exact breakdown | Output-token p99 |
| --- | ---: | --- | ---: |
| LAVD | 28/30 | 22 exact, 6 near, 2 truncated, 0 FAIL | 100,000 |
| Estonia | 30/30 | 30 PASS, 0 truncated | 8,766.38 |
| Hotel Lights | 28/30 | 28 exact, 2 FAIL, 0 truncated | 97,012.24 |

LAVD's two incomplete answers reached the 100,000-token output cap.
All 28 finished LAVD answers received credit under the official scorer
(which includes six NEAR answers). This is **28/28 credited among finished
answers**, not proof of 30/30 exact accuracy. The official 28/30 is retained.
Hotel's two failures did **not** hit the output cap and are not relabeled.
All three selected runs had zero request errors. Earlier Estonia attempts
were rejected by the server's context budget; they remain separate failed
setup receipts, not model-answer failures.

Each profile: 30 requests, concurrency 10, `reasoning_effort=max`, output cap
100,000; seeds 20260907 (LAVD), 20261007 (Estonia), 20261107 (Hotel).
Four RTX PRO 6000 96GB SM120 GPUs, PCIe, TP4/DCP1, MTP3 greedy,
B12X_MLA_SPARSE attention, NVFP4 MLA KV (`nvfp4_ds_mla`), native P8 MoE,
E4M3 activations, CUDA graphs, prefix caching enabled, batch 8192.
These are requests within profile runs, not 30 independent server launches.
See the receipt manifest for exact launch commands and differing context limits.

Runtime image: `sha256:609a5fc1cd7d994ba32d9c03626c414d315947eb9f13fab474a15bc8dfbe0129`.
The accompanying JSON/log pairs preserve the benchmark's scoring.

## Full-model KLD

**0.03418114591027796 mean true-decode KLD**, window BCa 95% interval
**[0.0291483518, 0.0409784257]** on 32 already-opened conditional-fit windows.
Uniform coupled K4 measured 0.0369674524: upgrades-only is 7.54% lower at a
larger bit budget. BF16 teacher-to-student KL, CPU FP64; 2,047 true-decode
rows/window, first prefill row excluded. This is a historical checkpoint
measurement on the earlier pinned runtime, **not a fresh RC5-runtime KLD**.

[Protocol and raw per-window receipts](P8_LATEST_KLD.md).

## Speed: separate sustained-decode benchmark

| Concurrency | Aggregate sustained decode tokens/s |
| --- | ---: |
| 1 | 178.336464 |
| 2 | 235.605573 |
| 4 | 300.874345 |

Client-measured prefill: 8k **8,063** tokens/s; 32k **7,998** tokens/s
(rounded). Source is the final `llm_decode_bench` JSON and matching log,
not vLLM rolling throughput. MTP3, TP4/DCP1, B12X attention/PCIe collectives,
NVFP4 MLA KV, P8 E4M3, 4.65874 routed bpw, graphs, 300W/GPU and +6000 memory
offset; no clock/power changes are made by the serving recipe. Prefix cache
enabled for prefill; separate cache-disabled decode launch and zero measured
prefix-hit delta. One configuration run: no repeated-run confidence claim.
The 200 tokens/s C1 target was not reached. This is not a matched comparison
against EXL3 or stock NVFP4. Reasoning-profile generation statistics above
are not substituted for this sustained-decode benchmark.

## Context and release boundary

The base model declares 1,048,576 maximum positions. The requested production
recipe limit is 1,000,000, without new RoPE scaling. That configuration is
not itself proof of retrieval accuracy at one million tokens. Latest needle
validation is tracked separately; no 500k/1M pass is claimed here.

Raw receipt index: [manifest](../evidence/rc5-release-20260907/manifest.json).
Older tests and versions are retained. License remains the repository's
ShapleyMCG source-available license; upstream component licenses still apply.
