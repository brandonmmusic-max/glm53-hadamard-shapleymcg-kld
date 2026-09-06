# Fable takeover prompt — full coupled K4 P8 with same-size Shapley allocation

## Latest user directive — governing objective

Build the full GLM-5.3-Flash coupled Hadamard + input/output-scale P8 model
across **all 42 relevant routed layers (3–44)**, perform genuine Shapley-based
global allocation **without increasing the existing K4-equivalent size**, and
measure full-model end-to-end teacher KLD with exactly:

- B12X_MLA_SPARSE attention.
- nvfp4_ds_mla KV cache.
- Tail V2.
- Native P8 MoE with fused trellis decode to the E4M3 MMA path.
- E4M3 activations.
- 4.25 payload bpw target / existing K4-equivalent global storage budget.
- TP4, DCP1, no EP, CUDA graphs on, MTP off.

**Small-scale exploratory block testing is finished for this direction.**
Do not send this work back through another three-layer pass/fail pilot.
The user explicitly superseded the previous pilot-pass requirement for an
all-layer build. Preserve that pilot's failed result, but do not use it to
block the full-model experiment. Full-model success remains unproven.
Do not substitute H16, a global K5/5.25-bpw model, native FP6, or the old
6-bpw allocation objective. Do not make another BlockLDLQ pilot a prerequisite.

This is the updated target for Fable to implement, not a claim that it has
already been built or that coupling/Shapley is guaranteed to improve KLD.
This prompt edit does not launch a new job in Codex.

## Meaning of “same size” and “Shapley allocation”

Pin the existing full uncoupled K4 P8 manifest and actual packed-byte budget
before allocating. Keep the same tensor coverage and unchanged carrier
tensors. Report both nominal codec payload bpw and actual packed/model bytes.
Count scales, suh/svh vectors, signs, padding, allocation maps, codebooks and
headers explicitly; do not hide their cost behind a rounded “4.25 bpw.”

The prior full routed-weight payload was 161,715,585,024 bytes for
304,405,807,104 weights (4.25 payload bpw); revalidate those values against
the actual immutable checkpoint. This is not its total disk size.
The three-layer coupled encoding cost about 4.2539798595 tensor bpw including
its extra metadata. Resolve that overhead within the fixed global budget;
do not silently enlarge the requested model or call a larger model equal-size.

Shapley must actually determine and install an allocation, not merely be
mentioned in the model name. Explicitly define the players, admissible
choices, payoff, calibration roles and exact byte constraint; preserve
attribution values and the selected allocation manifest. Use existing
evidence and interaction structure to size the attribution work, not an
arbitrary 84-endpoint campaign.

Keep the full coupled boundary on all relevant layers. If K4 is fixed at every
weight, Shapley must allocate meaningful equal-cost encoding choices or a
defined finite encoding resource; it cannot redistribute bits when no rate
choices exist. If using mixed rates to achieve an aggregate K4-equivalent
budget, first state the exact rate menu, demonstrate native P8 ABI support,
and clarify that this is a mixed-rate model, not uniform K4. Do not silently
introduce a K3 arm or global K5 product against the user's earlier constraints.
If a rate-menu choice requires new user direction, ask that narrow question
without restarting small-scale research. Do not claim allocation is implemented
until the selected choices are present in the packed checkpoint and runtime.

## Execution priorities

1. Read the existing artifacts and recover completed work; do not start over.
   Verify no other agent or job owns the GPUs. Predeclare the full-model
   evaluation and exact allocation budget before inspecting new results.
2. Refresh the storage ledger and produce a bounded build schedule. Reuse
   valid layers 3,20,22 where their design matches; encode the remaining routed
   layers from pinned BF16 with the matching coupled transforms/scales.
   Retain fit-only domain-balanced calibration and existing GPTQ-style
   inter-group feedback. No new error-feedback scheme is required by this directive.
3. Complete and install the same-size Shapley allocation across the full
   relevant model. A uniform full coupled build is a useful intermediate or
   allocation comparator, **not the final requested deliverable**.
4. Prove all 42 layers × 4 ranks load and execute the intended coupled native
   P8 path, including correct activation transforms and input/output scales.
   Necessary ABI, bit-exactness and serving-correctness checks remain; they
   are not authorization for another exploratory block/layer search.
5. Measure the final full-model KLD on the same 32 conditional-fit windows.
   Compare against the existing **full-model uncoupled K4 P8** endpoint when
   identities/runtime/masks match. Never compare a full coupled model against
   a three-layer identity intervention as if it were a matched full-model control.
   Reuse valid saved controls; do not run fresh stock or cold speed repetitions
   simply for housekeeping. Disclose any mismatch that prevents valid reuse.
6. Report absolute KLD, paired delta, relative improvement, paired BCa, window
   wins, domain breakdown, exact achieved bytes/bpw, and full runtime labels.
   To attribute an improvement specifically to Shapley, retain a full coupled
   uniform/no-allocation comparator at a reconciled byte budget; otherwise
   label the measured effect as the combined coupling-plus-allocation change.
7. Push reproducible code, receipts, decisions, allocation and final results
   to the existing PRIVATE repo. Do not launch unrelated work before the
   requested full-model measurement.

Full-model improvement is the objective, not a guaranteed outcome. Keep the
user's nonblocking-CI development policy: report uncertainty honestly, but
do not reintroduce the discarded three-layer screening gate. No linear
extrapolation from the pilot, and no claim of untouched final qualification
from already-opened CF32.

## Completed evidence — preserve, do not rerun

The matched three-layer pilot completed both32-window arms successfully:

| Intervention on layers3/20/22; all other layers stock | True-decode KLD |
|---|---:|
| Identity K4 P8 | 0.052847380391592064 |
| Coupled H512/H128 + suh/svh K4 P8 | 0.05499088068580576 |

Coupled minus identity +0.0021435002942136946, mean4.0560% worse;
17/32 coupled wins; paired95% BCa
[-0.0010680525204973161,+0.007625670904706085]. The old development rule failed.
This remains valid historical evidence, not evidence that the full coupled
model necessarily loses. It does not isolate rotation, scales or each layer.

Pilot conditions: B12X_MLA_SPARSE, nvfp4_ds_mla KV, TailV2, native P8
mxf8f6f4 N128 MoE, E4M3 UE8M0_K32 activations, TP4/DCP1/noEP/graphs/MTPoff;
identity4.25 versus coupled4.2539798595 selected-projection tensor bpw.

Read results/P8_COUPLED_CF32_FINAL_V3.md and
evidence/cf32-coupled-comparison-v3/analysis.json.
Analysis SHA256:
019fe64ed14a859b852cf9f6910ba888f49e4ed9e83c271ef52a6720d8a22a52.
Score/retirement/runtime receipts are under that evidence directory and
evidence/cf32-coupled-candidate-v3/.

The older **full42-layer uncoupled/identity K4 P8**, without Shapley allocation,
measured approximately0.03730955956731441 true-decode KLD on CF32 with the
requested B12X/NVFP4-KV/TailV2/TP4-DCP1/graphs/MTPoff path. Authenticate the
saved per-window receipts and exact conditions before paired reuse.
A different earlier full-model run measured0.0400213948 with FP8 KV,
FlashInfer attention and eager mode; do NOT interchange these references.
The old0.0533314176 decoded-GPTQ reference is also not the requested matched
NVFP4-KV/native-P8 endpoint.

## Local environment and reusable artifacts

Execution remains on Brandon's Pop!_OS machine.

Repository:
 /home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-coupled-integration-v1
Branch: p8-coupled-integration-v1
Git remote: github
Private repo:
 https://github.com/brandonmmusic-max/glm53-hadamard-shapleymcg-kld.git

Campaign root:
 /media/brandonmusic/nvme1n1p3/glm53-trellismx-native6

- Coupled pilot chunks, sidecars, all864 experts and receipts:
  <campaign-root>/p8-coupled-three-layer-v1
- Full identity checkpoint:
  <campaign-root>/uniform-p8-all42-v1/manifest.json and sidecars/
- Earlier full identity CF32 TailV2 result:
  <campaign-root>/tail-v2-p8-exl3-cf32-product-validation-v1b/quality/
- Final pilot output:
  <campaign-root>/p8-coupled-three-layer-cf32-v3-captures/
- Pilot runtime manifest / execution seal / storage ledger:
  <campaign-root>/p8-coupled-three-layer-cf32-v3.json
  <campaign-root>/p8-coupled-three-layer-cf32-execution-v3.json
  <campaign-root>/p8-coupled-cf32-storage-v3.json
- BF16 source:
  /media/brandonmusic/klcstore/bmxfp4-glm53/downloads/GLM-5.3-Flash-BF16
- Exact Flash EXL3 suh/svh scale source:
  /home/brandonmusic/models/GLM-5.3-Flash-EXL3-4bpw
- Stock unchanged carrier:
  /home/brandonmusic/models/GLM-5.3-Flash-NVFP4
  (44 indexed shards plus3 extras have pinned hashes).
- Design: results/P8_COUPLED_SCALE_FLASH_EVIDENCE_PREPARATION_V3.json
- Transform: experiments/p8-coupled-transform-draw0-silu10-v1.json
- Identity design:
  /home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53/experiments/p8-kld-shapley-native6-v2.json
- Encoder: glm53_nvfp4/quantize_p8_coupled_scale_layer.py
- Runner: glm53_nvfp4/p8_coupled_cf32_executor.py
  This is a THREE-LAYER runner, not already a full-model/allocation runner.
  Extend its checks explicitly; never launch it and call that all42 coverage.
- Runtime preparer: scripts/prepare_p8_coupled_three_layer_cf32_runtime.py
- Pilot analysis: glm53_nvfp4/analyze_p8_coupled_cf32.py

Immutable coupled runtime image:
 sha256:ad6b26bf6d1f265d99b09383485ddef82a4acfaea43e28af46341ebb41da24e3

Reuse fixed coupled H512/H128, draw0, exact Flash suh/svh, capped SiLU
(gate max10, up -10..10). Existing fit-only calibration is domain-balanced;
layer20/22 verified capture is <campaign-root>/fit-capture-l20-l22-v1.
Acquire/verify appropriate fit coverage for remaining layers rather than
assuming those two layers' captures cover the full model.

Pilot source implementation86ceac2 passed33 focused tests and independent
review. Synthetic M1/M2/M64/M65, five eager repetitions, CUDA-graph logical
tensor checks and all3 real four-rank loaders passed. The completed pilot
also passed real serving dispatch and final cross-arm reauthentication.

## Evaluation and hard operational limits

CF32 roles:
 /media/brandonmusic/klcstore/bmxfp4-glm53/roles/roles-codec-conditional-fit32-v1.json
Role SHA256:
 b5d7e4524eb98ddfbd230a5d9a44de0dc5dbeb796c03e859898b5838e4463d14
Teacher root:
 /media/brandonmusic/klcstore/bmxfp4-glm53/teacher
Correct teacher lineage:
 brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits

Use32 windows,8/domain;2047 prediction rows/window, exclude only row0
one-token prefill from true decode. Equal-window mean KL(teacher||student),
FP64 CPU. Paired95% BCa over windows, B20000, seed20260902. Preserve exact
tokens/teacher hashes and causal mask. No selective exclusions or rerolls.
Define Shapley fit/conditional-fit/selection use explicitly and disclose
adaptive reuse; keep the28 protected confirmation logits unopened.

- Production stays OFF: user service klc-backend.service and SYSTEM timer
  klc-model-stack.timer; never restore automatically. Port8000 stays unbound.
- The pilot executor terminated exit0; all owned containers were cleaned up.
  Old PID2766432 and session26404 are TERMINAL, not jobs to resume/relaunch.
  Recheck live state in case Fable or another agent has since started work.
- Respect /run/lock/klc/model-stack.lock and avoid competing GPU jobs.
- Maximum aggregate new NVMe use remains30,000,000,000B without asking.
  The last projection was about29.748GB, so a full build needs a fresh,
  realistic storage plan and likely explicit additional storage approval.
  The new full-model objective does NOT silently waive that limit.
  Preserve original cutoff1788632160 and prior charges; no per-run reset.
  No large new writes to klcstore. Do not delete reusable chunks/checkpoints
  or replace the existing full identity checkpoint without permission.
- Stream one window at a time; raw1,268,157,440B. Persist and hash scores/NPZ
  before retiring that exact transient raw. No dense32/81GB capture batches.
- Thermal abort90C; user accepts89C. Keep existing safety gates.
- Native P8 E4M3 mxf8f6f4 costs twice NVFP4's MMA issue count. It is not P4
  speed class. Never claim one native FP4 matmul or guaranteed speedup.
- Keep the repo private and preserve ExLlamaV3, KQuant, QSRT and
  w4a8_trellis port attribution and existing licensing.

Historical failed launches remain preserved: v1 duplicate exec before any
window; v2 canceled prelaunch to remove stock-first scheduling; v3 completed.
The latest directive supersedes only future experimental scope and the
small-pilot gate, not those receipts, the recorded result, or safety limits.

**Start by preparing the full coupled, same-size Shapley-allocated P8 build
from existing artifacts. Do not return to exploratory small-block testing.**
