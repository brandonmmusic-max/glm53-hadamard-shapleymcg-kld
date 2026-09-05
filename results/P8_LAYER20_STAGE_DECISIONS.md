# Layer20 preparation decisions

1. Preserve the original layer20 encoding stage: chunk 0:72 completed, then
   session 43625 exited 1 before the next launch because the idle GPU was at
   77 C, above the unchanged 75 C startup guard (2 MiB used). This was a
   scheduling preflight failure, not an encoder result. No 72:144 output was
   produced by that attempt. Original plan and chunk receipts remain intact.
2. Before resumption, declare `p8-coupled-layer20-encoding-resume-v2.json`:
   validate the completed 0:72 chunk and execute only 72:144, 144:216, 216:288.
   Add up to 180 seconds of bounded cooldown before each launch, retain the
   75 C startup guard and record observations. Encoder, data, transform,
   scales and acceptance rule remain unchanged. No retry of a completed chunk.
3. Before packing, declare `p8-coupled-layer20-pack-v1.json`: all four valid
   source receipts required; CPU packing and reopened source-exact tensor
   comparison only. Reserve 4 GB additional output, capped at 16.11 GB for
   the complete three-layer artifact directory and 25.753 GB conservative
   campaign aggregate. No source retirement or layer22 launch is authorized.
4. Integrate agent commit 429b53c96cba96488605592f136276ee73e6bc58 as
   1cfc3ee: explicit layer selection in the real-sidecar loader closure.
   Layer/rank identity must agree in receipt, sidecar and actual runtime.
   This is a loader test, not MMA execution or KLD qualification.

Validation: the initial new packing tests caught an incorrect helper-module
import (5 failed, 6 existing tests passed). Corrected the import before any
packing launch; packing/remainder tests then 11 passed. New packing plus
generalized loader tests: 14 passed. No model quality result is inferred.

Conditions: attention and KV N/A for encoding/CPU packing; MoE target P8
coupled K4 MCG E4M3/UE8M0, target activation E4M3, approximately 4.25398 bpw
including transform metadata. P8 uses twice the NVFP4 MMA issue count.
Production remains off; protected roles remain unopened. No new KLD result.
