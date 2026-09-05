# P8 N64 / fused-scratch integrated TP4 pilot

Status: **successful single-run diagnostic**, 2026-09-05. Allocation remains
stopped pending the separate five-cold-run product comparison. No new KLD or
full-model decode-path numerical closure is claimed.

## Measured result

| Serving version | 32K C1 decode tok/s | 32K prefill tok/s | Evidence |
|---|---:|---:|---|
| Original uniform P8 | 20.6937 | 6940 (server) | Historical five-cold-run result |
| Previous P8 small-M N128 | 75.5962 | 7166 (client) | Historical single-run pilot |
| N64 FC1 + fused scratch | **100.5667** | **7176 (client)** | This single-run pilot |
| EXL3 serving | 91.2645 | 6252 (server) | Historical five-cold-run result |

The new decode point estimate is 33.03% above the previous P8 pilot and
10.19% above the historical EXL3 median. These are descriptive comparisons,
not confidence intervals or a five-run product pass. P8 is TP4/noEP/DCP1;
EXL3 is TP4/EP4/DCP4. The latter comparison is between serving systems, not
an isolated codec effect. Random synthetic padding also differs between runs.
Historical five-run prefill used server-side validation; pilot headlines use
client TTFT. Do not treat those mixed-basis prefill rows as a matched comparison.

The benchmark used the same authenticated 20-second sustained-decode protocol,
3-second warmup, max_tokens 4096, C1 and standalone-cold prefill. Actual decode
input was 32,319 tokens; prefill was 32,321, with four samples and no cached
tokens. Decode measurement lasted 19.996676 seconds, used continuous OpenAI
usage, had zero errors and was not underfilled. Server-side prefill validation
was 7,244 tok/s; 7,176 is the client measurement.

## Runtime and safety receipts

- Exactly 168 native-forward and 168 actual M1-dispatch pairs: all 42 routed
  layers on all four ranks selected N64 with fused scratch.
- FULL CUDA-graph capture completed on all four ranks. Earlier profiling
  established graph replay and fused procedural decoding; this pilot verifies
  the new dispatch under that serving configuration, not a fresh Nsight trace.
- Maximum benchmark GPU temperature: 73 C. Four-GPU peak used VRAM was
  370,306 MiB versus 370,250 MiB in the prior pilot; no serving memory failure.
  This is an observed total, not isolated attribution to arena lifetime.
- Exit code 0, owned container removed, backend restored active, timer left
  inactive as found. No protected data was opened.

The bottleneck investigation found insufficient useful small-M work despite
fusion and graphs already being active. The prior integrated trace then
localized approximately 5 ms/token to FC1 on the slow ranks and 0.55 ms to
fills. The B12X-derived narrower FC1 plus single scratch-clear arena reduced
device routed-block medians by 44.11–45.42% before this serving pilot. Failed
earlier device variants remain in `P8_FC1_TILES_DEVICE.md`; duplicated-copy
removal alone did not clear their frozen speed gate.

## Reproduction and limits

- Frozen launch commit: `3b29de8`; plan
  `experiments/p8-fc1-integrated-v1.json`, SHA-256
  `298e2eca8d5ba71c7834a016302eaf3e47e84553cf76fd1737ce9bd9e3fe99d6`.
- Image: `sha256:6c08dffb4184c2704173a12909f4bbfaaa866351e55cbf03a2182741baf81141`.
- Raw root: `/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-smallm-scheduler-v1/fc1-integrated-v1`.
- Portable receipts: `evidence/opened/codec-v2/p8-fc1-integrated-v1/`.
  The snapshot authenticates raw hashes and replays summary/dispatch checks;
  private environment/container dumps are omitted and logs are field-sanitized.
- Runner: `python3 -m glm53_nvfp4.p8_fc1_integration --plan <sealed-plan> --output <fresh-campaign-output>`.
  Reproduction requires the pinned local model/sidecar/image inputs; it is not
  independent reproduction merely because receipts replay.

Next: freeze a five-cold-run comparison and validate full-model M1 numerical
closure. Prompt-prefill KLD alone would exercise the unchanged M>1 fallback,
so cannot qualify this decode-only optimization.

P8 is K4 procedural MCG to E4M3, UE8M0/32, identity, 4.25 routed bpw, without
LDLQ. **Its native mxf8f6f4 path has twice NVFP4's MMA issue count. It is not
the P4/NVFP4 speed-class product.** See `THIRD_PARTY_NOTICES.md` for the
ExLlamaV3, KQuant/QSRT and B12X port attribution.
