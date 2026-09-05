# P8 N64 / fused-scratch versus EXL3: five cold starts per arm

Qualification: the sealed five-cold-run **speed-only system comparison passed** on 2026-09-05. P8 median sustained C1 decode was 100.5501 tok/s versus EXL3 91.2200 (+10.23%); median server-validated prefill was 6,959 versus 6,239 tok/s (+11.54%). All ten independent server runs completed without benchmark errors or underfill. Allocation remains stopped; no full-model M1 numerical or KLD closure follows from this result.

## Question

Does the fixed N64 FC1 plus fused-scratch P8 serving configuration exceed the authenticated EXL3 serving configuration on both predeclared median 32K C1 decode and standalone prefill metrics over five new starts per arm? The preceding single-run pilot motivated this comparison but is not included in these results.

## Identities

- Frozen launch commit: `524beeff6c53b643277af464da65d006295c0bc5`.
- Plan: [p8-fc1-cold-comparison-v1.json](/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-smallm-v1/experiments/p8-fc1-cold-comparison-v1.json), SHA-256 `77fd98c7af0c16a44101beb006a2fc26f0036e227ef1a89316e613ca26da1ec8`.
- P8 image: `sha256:6c08dffb4184c2704173a12909f4bbfaaa866351e55cbf03a2182741baf81141`.
- EXL3 image: `sha256:d1b6c021df11056cebde469cadc05f55cb21ec4ffc8b54ae6f08161df0c493bf`.
- Benchmark: `local-inference-lab/llm-inference-bench`, version 0.4.29, clean checkout `42c38fdd0476cf86940268f4e098581e5b61542e`; executed script SHA-256 `c1d9b66dbf70df23545a71451b5bc2ecaaa746ed2441312e4fb77a8bf476504a`. The script hash is the sealed execution identity.
- Model: GLM-5.3, uniform P8 routed sidecars versus the authenticated EXL3 K4 serving checkpoint. Exact model manifests, image/source hashes and historical serving-recipe receipts are bound by the plan; no new model download or conversion occurred.
- Fresh byte audit: 168 P8 routed sidecars and 120 EXL3 shards, 337,357,865,080 bytes. Receipt SHA-256 `8ce9c07dd89e307ad50aa7c4af93ffbdac51e0637a143ceb64add47f8651eefb`. P8 carrier provenance remains the prior receipt, not a fresh carrier rehash.
- Community wiki/runbook: N/A as an execution authority for this already-sealed local experiment. The frozen plan and authenticated prior serving recipes governed it; no current wiki configuration was substituted after collection.

## Fixed conditions

- Four NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition GPUs on the same host. Power limits: 300/300/300/275 W, checked unchanged. GPU UUID/PCI identities are preserved in the hardware inventory.
- Driver 610.57.04; NVIDIA-reported CUDA compatibility 13.3. This is not a claim about the container toolkit version. Exact PyTorch/NCCL version strings: UNKNOWN — needs verification; images are immutable. Clocks were not newly locked; before/after NVIDIA XML and per-run hardware telemetry are preserved.
- Target-only vLLM serving, speculation disabled, NVFP4 MLA KV (`nvfp4_ds_mla`), FULL CUDA-graph capture, C1, nominal 32K prompts, 20-second sustained-decode and 20-second standalone-cold prefill collection, 3-second decode warmup, max_tokens 4096. Exact launch/scheduler arguments remain authenticated in private recipes with public hash references.
- Actual decode inputs: 32,318–32,320 tokens; prefill: 32,320–32,322. These satisfy the sealed ±5% prompt-length gate. Random synthetic prompt bytes are not claimed identical across starts.
- Compiled caches were warm and reused in each arm's existing cache directory. “Cold” means ten independently created server processes, not ten empty JIT caches or uncached weight reads.
- Thermal policy: start only at or below 75°C; abort at or above 90°C. Highest sampled runtime temperature was 84°C, below the abort threshold, not the earlier pilot's 73°C.

## Changed variables

This is a **system comparison**, not codec or isolated kernel causality. P8 is TP4/noEP/DCP1 with N64 FC1 and fused scratch; EXL3 is TP4/EP4/DCP4. Images, weight representations, serving backends, topology and compiled caches differ. The narrower FC1 and arena changes are not isolated against N128 by this experiment.

P8 remains K4 procedural MCG to E4M3, UE8M0/32, identity, 4.25 routed bpw, without LDLQ. Its native E4M3 `mxf8f6f4` m16n8k32 path issues **twice as many MMA instructions as NVFP4 for equal K**. This does not turn P8 into the native P4/NVFP4 speed-class product. B12X/ExLlamaV3/KQuant attribution is retained in the repository notices.

## Commands

The executed runner was `python3 -m glm53_nvfp4.p8_fc1_cold_compare run --plan <the sealed absolute plan path>`. This report is not authorization to rerun it. Candidate and control shared this benchmark command, changing only the served model name and per-slot result path:

```text
python3 /home/brandonmusic/KLC_SANDBOXES/glm53-exl3-k4-r10-rebase/tooling/llm-inference-bench/llm_decode_bench.py
  --host 127.0.0.1 --port 8022 --model <glm53-p8-fc1-cold-v1-p8 OR glm53-p8-fc1-cold-v1-exl3>
  --concurrency 1 --contexts 32k --duration 20 --max-tokens 4096
  --decode-warmup-seconds 3 --standalone-prefill --prefill-contexts 32k
  --prefill-duration 20 --prefill-metric auto --token-targeting estimate
  --display-mode plain --no-resume --output <fresh-slot/result.json>
```

Launch/environment/container contents remain private; the snapshot retains their hashes, not arbitrary environment values or unsanitized logs.

## Method

- Experimental unit: one independently started and warmed server process; five per arm. Each contains one sustained C1 streaming request and three standalone prefill samples. Requests, tokens and thermal observations are subsamples, not independent server repetitions.
- Predeclared order: EXL3/P8, P8/EXL3, EXL3/P8, P8/EXL3, EXL3/P8, serialized in five round blocks.
- Primary decode: `openai_continuous_usage`, measurement intervals 19.982940–19.999978 seconds. The duration-limited stream need not finish its natural completion; this is not a finite-request throughput test.
- Primary prefill: `server_validation`, Prometheus `kv_computed`, zero cached tokens, three valid samples per run. The client TTFT headline is retained in raw JSON but not substituted for the primary metric.
- Aggregation: median of five per-process outcomes, separately for each arm and metric. Full per-run distributions below describe uncertainty; no confidence interval or p-value was predeclared. Repeated EXL3 control ranges were 91.1788–91.2761 decode and 6,225–6,555 server-prefill tok/s.
- Stopping/exclusions: exactly ten new starts, no retries, replacements or exclusions; all completed. A failed protocol would block qualification. Pass required both candidate medians strictly above comparator medians.
- Interpretation: confirmatory for this fixed speed gate only. Different topology/backend and thermal or cache history remain rival explanations for component-level attribution. Prefill declined after the first block in both arms; all rows are retained. Tail latency, reasoning suites, qualitative coding, accuracy and KLD: Not tested in this comparison.

## Results

### Target-only sustained decode

| Round | EXL3 tok/s | P8 N64/fused tok/s |
|---|---:|---:|
| 1 | 91.178761 | 100.527209 |
| 2 | 91.276088 | 100.550111 |
| 3 | 91.210887 | 100.535756 |
| 4 | 91.240914 | 100.571522 |
| 5 | 91.219974 | 100.609534 |
| Median | 91.219974 | 100.550111 |

Median delta: +9.330137 tok/s, **+10.2282%**. Candidate range: 100.5272–100.6095 tok/s. Every run had zero errors, effective concurrency 1, `underfilled=false`, and the declared continuous-usage source.

### Standalone server-validated prefill

| Round | EXL3 tok/s | P8 N64/fused tok/s |
|---|---:|---:|
| 1 | 6,555 | 7,067 |
| 2 | 6,279 | 6,986 |
| 3 | 6,239 | 6,959 |
| 4 | 6,225 | 6,924 |
| 5 | 6,229 | 6,941 |
| Median | 6,239 | 6,959 |

Median delta: +720 tok/s, **+11.5403%**. Candidate range: 6,924–7,067 tok/s. Both arms used the same server-validation basis; client and server measurements are not mixed.

### Runtime, thermals and cleanup

Every P8 run produced exactly 168 native-forward and 168 actual M1-dispatch pairs: all 42 routed layers on four ranks selected N64/fused scratch. Both arms completed FULL graph capture on all four ranks. These logs prove capture/configuration and dispatch, not a new Nsight trace of measured-request graph replay.

Maximum sampled GPU temperatures by round were EXL3 67/82/83/82/83°C and P8 81/83/84/84/84°C. The warmer later blocks are preserved, not excluded. All ten owned containers stopped and were removed. The root receipt records exit 0, ten completed slots, independently verified empty owned-container inventory, backend restored active and timer inactive as found, with no restoration errors.

Collection ran 2026-09-05 07:23:49.667102–07:52:48.193036 UTC. Snapshotting began only after the same unit reported `inactive`, `Result=success`, `MainPID=0`; no new serving run was launched while preserving evidence.

## Evidence statement

Under the pinned four-GPU regimes and 32K C1 protocol, the P8 serving system exceeded EXL3 on both five-start medians, passing the declared speed gate. This supports only the tested serving-system speed claim. It does not establish cross-topology codec superiority, a full-model M1 exact-logit/KLD result, untouched-final quality, or authorization to restart allocation. Prefill quality evaluation alone cannot qualify the decode-only M1 change.

## Supporting artifacts

- [Terminal snapshot and hash manifest](/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-smallm-v1/evidence/opened/codec-v2/p8-fc1-cold-comparison-v1/snapshot.json): SHA-256 `05d24901ed24a27838eb2120848135f932aaac7d5464be3e7345ebd105ff5b30`; retains every raw result, summary, thermal series, safe graph/dispatch log, runtime audit and execution receipt, with private-file hash references.
- [Replayed analysis](/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-smallm-v1/evidence/opened/codec-v2/p8-fc1-cold-comparison-v1/analysis.json): SHA-256 `f18a0dffde98f5a32dffda365d06176a7b416ca9bfee8833c5f07c104d83d226`.
- [Structured report](/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-smallm-v1/evidence/opened/codec-v2/p8-fc1-cold-comparison-v1/report.json): SHA-256 `89966011a74b77842ab382f7d93fc28caffa251084ffa4e04c5b31592e57cae6`.
- [Root execution and restoration](/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-smallm-v1/evidence/opened/codec-v2/p8-fc1-cold-comparison-v1/execution.json): SHA-256 `01c6b00bd9471bb81ca07a04094abcd3e7bb57a41421ff2185e1f7bbb5953824`.
- Raw root: `/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-smallm-scheduler-v1/fc1-cold-comparison-v1`. No raw source was mutated. Full raw and sanitized runtime replay passed during terminal snapshot creation.
- CPU-only verification route: `python3 -m glm53_nvfp4.p8_fc1_cold_compare analyze --plan <sealed-plan>`. The snapshot helper is terminal-only, fresh-destination-only and cannot overwrite this snapshot.
