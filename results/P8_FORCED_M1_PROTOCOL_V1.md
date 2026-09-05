# Complete causal-row validation of the optimized P8 decode path

Status: **CPU-tested and independently reviewed; not GPU-executed yet**,
2026-09-05. The fresh five-cold-run comparison remains active. No new KLD,
numerical-closure pass, allocation result or speed qualification is claimed here.

## Frozen design before execution

The dedicated image is
`sha256:071da1f9e9b24709e59a0959b97a8d524e82fca2fbf4de9a5dc2113a938c6b60`.
Its source-verified build receipts are described in
[the capture-image report](P8_DECODE_CAPTURE_V2_BUILD.md).
Both arms use that same image, the all-42-layer P8 checkpoint, TP4/noEP/DCP1,
NVFP4 MLA KV, Model Runner V2, FULL graphs and max_num_seqs=1. The factors are
FC1 N128 versus N64 and separate versus consolidated scratch clearing.

| Start | Arm | Scope | Advance condition |
|---|---|---|---|
| 1 | N128, separate scratch | First fixed conditional-fit window | Complete capture |
| 2 | N64, consolidated scratch | Same fixed window | All 2,047 raw logit rows bitwise equal |
| 3 | N128, separate scratch | All 32 conditional-fit windows | Complete capture |
| 4 | N64, consolidated scratch | Same complete panel | Preserve full exact result and score KLD |

Each request supplies one prompt token and forces all remaining 2,047 original
token IDs. Captured teacher rows 0–2046 are one single-token prefill row plus
2,046 true decode rows, without shifting or dropping a tail. The full panel is
65,504 scored rows, eight windows per domain. The canary is a smoke test, not
domain-general evidence. Its fixed first-window choice cannot be rerolled.
All 66 window captures, including both canaries, require 83,698,391,040 raw
bytes; the launcher additionally requires 20 GiB free margin.

The exact gate requires all values over the real 154,880-token vocabulary to
match, including signed-zero bit patterns. Original head dtype and width must
match between arms. Storage uses FP32, which does not imply a native FP32 head.
Every complete request must return exactly the forced token IDs, and runtime
logs must prove all 168 native layer/rank dispatches plus V2/FULL operation.

## KLD and evidence gates

`p8_decode_analysis.py` uses the pinned existing CPU FP64
`KL(teacher || student)` metric, with eight Torch CPU threads. It reports all
window means, per-position arrays, row-zero and true-decode diagnostics, and
the equal-window aggregate. BCa intervals use 20,000 paired window resamples,
seed 20260905. These are window intervals, not repeated model preparations.

Raw equality is rechecked immediately before score reuse. When current logits
are bitwise equal, the reference score can be reused with an explicit receipt;
otherwise both arms are scored. Identical paired deltas have no inferred BCa
interval and are reported with a null interval plus explanation. A favorable
KLD change never replaces a failed exact-logit gate.

The analyzer replays exact-comparison receipts, requests, responses, native
runtime proofs, hashed artifact inventories, temperatures, container ownership,
fresh-process chronology and restoration. Reloaded capture identities are
bound to the authenticated stage receipts and raw hashes are rechecked after
scoring. The final analysis binds plan, execution and both exact-comparison
receipts, preserving mismatch diagnostics as well as successful outcomes.

## Execution and safety

The launcher is `python3 -m glm53_nvfp4.p8_decode_launcher`, with `plan` and
`run` commands. Planning requires the completed ten-run cold-comparison
receipt and writes a fresh plan plus SHA-256 sidecar. A failed speed threshold
does not prevent correctness measurement; a partial or failed execution
protocol requires an amendment. Planning cannot silently substitute partial
speed evidence. See `--help` for the required absolute input/output paths.

Execution requires a clean checkout, exact image/package/source identities,
the terminal cold-comparison unit, available storage and the model-stack lock.
It checks sources and mounted payloads between stages. SIGTERM, SIGHUP and
SIGINT cannot interrupt critical cleanup/restoration. Production restoration is
withheld while an owned container is live or its identity/state is uncertain.
Root-written capture files are normalized only through explicit regular-file
ownership changes inside the authenticated capture container; no recursive
model/cache permission changes are used.

After a complete capture protocol, run:

```bash
python3 -m glm53_nvfp4.p8_decode_analysis \
  --plan /absolute/path/to/sealed-plan.json \
  --output /absolute/fresh/analysis-directory
```

Independent reviewers cleared the launcher and analyzer after negative tests
for interrupted cleanup, source/payload drift, incorrect storage requirements,
failed canary advancement, stale exact flags, role drift, missing response
artifacts and nonfinite cooldown data. Actual V2 device capture is still the
unmeasured next gate; CPU tests cannot establish it.

Capture synchronization invalidates timing. No protected role is opened and
allocation remains stopped. **P8 uses E4M3 mxf8f6f4 at twice NVFP4's MMA issue
count; this is not the native P4/NVFP4 speed-class endpoint.**
