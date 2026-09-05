# Fable takeover prompt — GLM-5.3-Flash coupled P8 experiment

You are taking over Brandon Music's local experiment. Continue the actual
measurement already running; do not restart planning, encoding, or baselines.
This handoff was prepared September 5, 2026, while the candidate container was
initializing. Revalidate live state: this is a snapshot, not a completion claim.

## Immediate priority and latest user direction

Get the end-to-end teacher KLD measurement for the newly encoded **coupled
Hadamard + input/output-scale P8 layers 3, 20, 22** done ASAP. Candidate first.
Do not run a fresh stock baseline, cold repetitions, speed tests, new rotation
searches, or other side projects before this measurement.

The earlier full 42-layer P8 model is **identity/unrotated P8**, not this new
coupled encoding. Only layers 3,20,22 have the new encoding. All are completely
encoded, packed, and real-TP4-loader checked. The complete candidate's
32-window true-decode KLD is now **0.05499088068580576**; matched identity is
running. Do not describe packing/loader or synthetic-kernel tests
as end-to-end quality proof.

## Active run — attach, do not duplicate

Execution host: Brandon's Pop!_OS machine. All work stays local.

Repository:
`/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-coupled-integration-v1`

Branch: `p8-coupled-integration-v1`. Runtime source HEAD when launched:
`86ceac2f46a813aec51b6efaf6c5b290ef287dc5`.

Campaign root:
`/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6`

The currently running command is:

```bash
PYTHONPATH=. python3 -m glm53_nvfp4.p8_coupled_cf32_executor execute --seal /media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-three-layer-cf32-execution-v3.json
```

Snapshot host PID: **2766432**. Codex PTY session: **26404** (Fable may not
have this handle; inspect the exact PID/command and Docker state instead).
Current container: `glm53-p8-three-layer-cf32-identity_p8`, port **8032**.
The coupled container completed and was cleaned up by the owned executor.
The container is past the shell-launch error. At23:27:19 UTC its log reported
application startup complete, followed by a successful GET /v1/models.
All32 candidate windows completed: true-decode mean **0.05499088068580576**,
including-prefill mean **0.05601149974746436**. Root replayed all32 score NPZ
hashes/means and retirement hashes; arm execution exit0, cleanup PASS, final
runtime audit PASS with all32 IDs in order. This is not yet a matched
improvement claim. Conditions: B12X/nvfp4_ds_mla/native coupled P8/E4M3/
~4.25398bpw, TP4/DCP1/noEP/graphs/MTPoff, layers3/20/22 coupled, reststock.
Read `results/P8_COUPLED_CF32_CANDIDATE_RESULT_V3.json`. Let the active
matched identity arm finish, then obtain paired delta and BCa from analysis.

Seal SHA256:
`41af897c4f52237a74af8eca7f1d6c8ef1f274edf10c1697f942a4a7d0f000d6`

Related files under campaign root:

- `p8-coupled-three-layer-cf32-v3.json`: authenticated runtime manifest.
- `p8-coupled-cf32-storage-v3.json`: current sealed storage ledger.
- `p8-coupled-three-layer-cf32-execution-v3.json`: execution seal.
- `p8-coupled-three-layer-cf32-v3-captures/`: live output root.
- Within each arm: `scores/<arm>/*.score.json`, `*.scores.npz`, retirement
  receipts, `runtime-audit.json`, final runtime audit, and `execution.json`.
  Inspect actual filenames; score receipts specify their NPZ paths.
- Root `analysis.json` exists only after both arms complete and final
  authentication succeeds. Root `execution.json` is written at termination.

Safe initial commands:

```bash
ps -p 2766432 -o pid,etime,stat,args
docker ps --format '{{.Names}} {{.Status}}'
docker logs --tail 50 glm53-p8-three-layer-cf32-identity_p8
```

A missing final receipt does not mean a live process stopped. A polling timeout
does not authorize restart. Never launch a second executor while this one lives.
If a failure is terminal, inspect `error.private.txt`, the arm execution receipt,
and `server-final.private.log`; preserve the attempt before any amended retry.
Do not publish raw private launch/environment/request files without checking
for secrets and private corpus content.

## Fixed experiment

Order: **coupled_p8 → identity_p8**, exactly 32 conditional-fit windows per arm.
Both replace only layers 3,20,22; all other layers use the same authenticated
stock NVFP4 carrier. This is a full-model output KLD measurement of a three-layer
intervention, not layer-local weight NMSE.

Candidate: fixed coupled H512/H128 sign-Hadamard, draw0, exact Flash EXL3
`suh`/`svh` scales, target Flash capped SiLU (gate max10, up -10..10).
K4 procedural MCG trellis → E4M3 weights, UE8M0 scales/32.
Nominal payload 4.25 bpw; coupled tensor payload including scale/draw metadata
approximately **4.2539798595 bpw**, not a verified whole-model disk rate.
Existing full-Hessian GPTQ-style inter-group feedback remains; no new BlockLDLQ.

Runtime: **TP4, DCP1, no EP, B12X_MLA_SPARSE attention, nvfp4_ds_mla KV,
Tail V2, E4M3 P8 activations, CUDA graphs, MTP off**. Other stock layers retain
their stock paths. The immutable image is:
`sha256:ad6b26bf6d1f265d99b09383485ddef82a4acfaea43e28af46341ebb41da24e3`.

The P8 runtime decodes to the native E4M3 MMA path; `mxf8f6f4` has **twice
NVFP4's MMA issue count**. Do not call it native P4 speed class or promise a
speed gain. Actual coupled dispatch for all 3 layers × 4 ranks must pass the
runtime audit. A generic startup banner currently contains the word `identity`;
do not use that banner alone either to prove or disprove coupled installation.
Use the sidecar metadata and exact weight/forward hooks checked by the runner.

The identity comparison is the existing immutable encoder output, not a freshly
re-encoded identical pipeline. Therefore this tests the **whole coupled
candidate**, not isolated causal attribution to Hadamard or scales individually.

## Metric and decision — do not change after reading results

Role manifest:
`/media/brandonmusic/klcstore/bmxfp4-glm53/roles/roles-codec-conditional-fit32-v1.json`
SHA `b5d7e4524eb98ddfbd230a5d9a44de0dc5dbeb796c03e859898b5838e4463d14`.
Teacher root: `/media/brandonmusic/klcstore/bmxfp4-glm53/teacher`.
Correct HF teacher lineage is `brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits`;
use pinned local teacher bytes, no new download.

32 windows, 8 per domain. Exactly 2047 captured prediction rows/window.
Row0 is the one-token prefill row: report separately. **Rows1..2046** enter
true-decode KLD. Equal-window mean; no selective exclusions/replacements.
Paired 95% BCa over windows, B=20000, seed=20260902.

Development pass = coupled mean KLD below identity mean. **CI is reported but
nonblocking**, per explicit user revision. No fixed minimum percentage hurdle.
Stock metric is **not tested**, not zero or inferred from unrelated old runs.
Already-opened CF32 is development evidence, not untouched final qualification.

Preregistered amendment:
`experiments/p8-coupled-three-layer-cf32-candidate-first-v3.json`.
Analysis: `glm53_nvfp4/analyze_p8_coupled_cf32.py`.
Runner: `glm53_nvfp4/p8_coupled_cf32_executor.py`.
Preparer: `scripts/prepare_p8_coupled_three_layer_cf32_runtime.py`.

Report the candidate mean as soon as all32 candidate windows are valid, clearly
labeling it candidate-only pending the matched identity comparison. Do not wait
for unrelated side work to tell Brandon the result. Then let the scheduled
identity arm finish and report absolute delta, relative improvement, BCa,
per-window wins, per-domain results and any failures. Every numeric report must
carry attention backend, KV dtype, MoE backend, activation precision and bpw.

## Artifact locations — no re-encoding needed for the pilot

Coupled chunks/sidecars/receipts:
`<campaign-root>/p8-coupled-three-layer-v1`
Sidecars: `sidecars/layer-003`, `layer-020`, `layer-022`, four ranks each.
864 experts total, all packed and real-loader PASS. Both original chunks and
packed sidecars remain; do not delete them casually.

Identity sidecars:
`<campaign-root>/uniform-p8-all42-v1/sidecars`
Identity manifest: same parent `manifest.json`.
Identity design:
`/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53/experiments/p8-kld-shapley-native6-v2.json`

Stock common carrier: `/home/brandonmusic/models/GLM-5.3-Flash-NVFP4`.
All44 indexed shards plus3 extras are hash-pinned. Stock authentication is
required for the shared unchanged layers; it is NOT a stock-only serving run.
BF16 source:
`/media/brandonmusic/klcstore/bmxfp4-glm53/downloads/GLM-5.3-Flash-BF16`.
Scale source: `/home/brandonmusic/models/GLM-5.3-Flash-EXL3-4bpw`.
Design: `results/P8_COUPLED_SCALE_FLASH_EVIDENCE_PREPARATION_V3.json`.
Transform: `experiments/p8-coupled-transform-draw0-silu10-v1.json`.
Layer20/22 calibration:
`<campaign-root>/fit-capture-l20-l22-v1` (old klcstore references are incomplete;
use the verified relocation recorded in encoder receipts).

## Hard operational limits

- **Production stays OFF. Never restore/start it automatically**, including on
  failure or at completion. User service `klc-backend.service` and SYSTEM timer
  `klc-model-stack.timer` were inactive; port8000 unbound.
- Root executor owns `/run/lock/klc/model-stack.lock`. Agents must not launch
  competing GPU jobs. Do not restart the running container to inspect it.
- User permits at most **30,000,000,000 bytes aggregate new NVMe use** without
  asking. Latest v3 projection approximately29.748GB, including reserves.
  Physical disk free space does NOT increase that authorization. No large new
  writes to klcstore. Preserve fixed ledger cutoff **1788632160** and prior
  attempts; do not reset the budget per run.
- Capture one window at a time: raw is1,268,157,440B. Runner saves and hashes
  durable scores/NPZ before unlinking only that exact transient raw. No dense32
  capture, no81GB capture batch. Never delete source chunks for convenience.
- No protected28 confirmation logits, no final-role opening. Fit/CF/selection
  remain distinct. No new downloads or re-encoding before this pilot result.
- Thermal abort is90C; user explicitly accepts89C. Do not impose a new lower
  sustained-run limit. Do not disable existing safety gates to force a result.

## History needed to avoid repeating mistakes

v1 failed before any window: launch command emitted `exec exec ...`.
Fixed by normalizing the real source's leading exec; actual-scoped tests added.
v2 was canceled by user direction during prelaunch authentication, exit143,
before any container/window, to remove stock-first scheduling.
v3 implements the reviewed candidate-first order. All failures/cancellations
are preserved. No matched coupled improvement or regression result exists yet.

Synthetic M1/M2/M64/M65 numerical checks, five eager repetitions, CUDA-graph
checks and all3 real four-rank sidecar loader checks passed. These are narrow
correctness checks, not full-model serving/KLD qualification. v1 graph receipt
failure was preserved; v2 distinguishes physical route permutation metadata
from bit-exact logical tensors.

Older full-model identity P8 true-decode CF32 mean was approximately0.03730956;
EXL3 repeats were approximately0.03161/0.03124/0.03105. Conditions differ:
P8 TP4/noEP/DCP1/E4M3/4.25bpw vs EXL3 TP4/EP4/DCP4/BF16/4bpw, both
B12X_MLA_SPARSE, nvfp4_ds_mla KV, TailV2, graphs, MTP off. Historical only;
do not subtract these from the current three-layer intervention as a matched
codec effect, and do not claim a new result using those old numbers.

## After the pilot

If coupled mean beats matched identity, plan one all42-layer coupled build,
then full-model KLD. First obtain storage permission if the aggregate30GB limit
cannot be respected. Do not extrapolate a three-layer percentage linearly.
Then evaluate properly matched serving speed/prologue cost; no native P4 claim
for P8. Later K5 and allocation work are not the immediate priority.

If it fails, preserve result and discuss/use the authorized encoder-only
three-layer calibrated inter-block BlockLDLQ fallback preserving P8 ABI;
do not infer it is ruled out by the August29 intra-tile-only LDLQ ablation.
Do not restart an archive audit or broad rotation search before reporting.
The old automatic goal text about H16/learned/Shapley6bpw is stale scheduling
context: the user's later codec/coupled-P8 directions control current work.

## Private reproducibility repository

Git remote `github` (not local `origin`):
`https://github.com/brandonmmusic-max/glm53-hadamard-shapleymcg-kld.git`
Keep it **private**. Push code, decisions, sanitized receipts, analysis and
attribution (ExLlamaV3, KQuant, QSRT/w4a8_trellis ports) as work completes.
Do not fabricate or silently replace license text; retain existing licensing.

Read `results/P8_COUPLED_CF32_ROOT_LAUNCH_DECISIONS.md` and the critical audit
for chronology. Source change27805aa (integrated86ceac2) passed33 focused tests
and independent review. Audit agent's latest report commit643957c exists in
the shared Git object store; it may not yet be integrated at snapshot time.

**First action: inspect the live executor and identity container, then obtain
the matched comparison. The32-window coupled KLD is complete. Do not stop or
duplicate a healthy measurement.**
