# Fable takeover plan: full coupled K4 P8 + same-size Shapley allocation

Recorded 2026-09-05 20:55 EDT. Status: **plan and predeclaration only**. No GPU
job, container, encode, or KLD run has been launched by this plan. Production
(`klc-backend.service`, `klc-model-stack.timer`) is inactive; port 8000 unbound;
all four GPUs idle at takeover; no owned containers.

Governing directive: `HANDOFF_FOR_FABLE.md` (full coupled Hadamard +
input/output-scale K4 P8 on layers 3-44, Shapley allocation at the existing
K4-equivalent size, measured with B12X_MLA_SPARSE / nvfp4_ds_mla / Tail V2 /
native P8 mxf8f6f4 / E4M3 activations / TP4 DCP1 noEP / CUDA graphs / MTP off).

## 1. Recovered state (verified from receipts, not from the handoff text)

| Item | Value | Source |
|---|---|---|
| Full identity K4 P8, requested path, true-decode KLD | 0.03730955956731441 (mean incl. row 0: 0.038271610648036664) | `evidence/baselines/tail-v2-cf32/repeat-03-p8.json`; v1b quality root |
| EXL3 4.0 bpw on the same path (TP4/EP4/DCP4) | 0.031611840268931456 true decode | `evidence/baselines/tail-v2-cf32/repeat-03-exl3.json` |
| P8 minus EXL3, true decode | +0.005697719298382958, +18.02 %, BCa [+0.00138, +0.01110] | `results/TAIL_V2_CF32_REPEAT1_PROVISIONAL.md` |
| Full identity, FP8 KV / FlashInfer / eager | 0.04002139481676188 | `uniform-p8-all42-v1/fullmodel-completion-audit.json` (do not interchange) |
| Three-layer pilot identity / coupled | 0.052847380391592064 / 0.05499088068580576 | `results/P8_COUPLED_CF32_FINAL_V3.md` |
| Run-to-run determinism of the P8 path | repeat-2 and repeat-3 bitwise equal to repeat-1 (raw SHA match) | v1b execution receipts |

The identity full-model control was measured on image `klc/glm53-p8-tail-repair:v2`
(`sha256:0336113e...`) with `GLM53_P8_FC1_TILE_N=64`, `GLM53_P8_FUSED_SCRATCH=1`.
The coupled pilot ran on `klc/glm53-p8-coupled:v9` (`sha256:ad6b26bf...`) with
`FC1_TILE_N=128`, no fused scratch (the scale component requires the M1 N128
owner path). **These runtimes differ**, so the 0.0373 control cannot be reused
as the matched pair for a full coupled model without disclosure. A one-arm
identity re-measurement on the v9-lineage image (about 18 min) removes the mismatch.

Existing Shapley evidence: 8 of 84 coalitions of `p8-kld-shapley-pilot-v1`
were run (means 0.0376-0.0397) on config
`nvfp4-v5-identity-gate-up-humming-bf16-instanttensor-tp4-ep4-dcp4-eager-nomtp-kvfp8`,
i.e. a decoded/humming overlay path with FP8 KV, not the requested native path.
They inform sizing only; they are not payoffs for this game.

## 2. Exact byte accounting

Routed weights: 42 layers x 288 experts x 3 x 4096 x 2048 = 304,405,807,104.

| Quantity | Bytes | Note |
|---|---:|---|
| Identity K4 payload (trellis + UE8M0/32) | 161,715,585,024 | exactly 4.25 bpw; 168 files, 962,592,768 payload + 728 header each |
| Identity sidecar files on disk | 161,715,707,328 | measured `du -sb` of `uniform-p8-all42-v1/sidecars` |
| Coupled K4 per rank file | 963,497,456 | tensors 963,494,176 + 3,272 header |
| Coupled metadata per rank | 901,408 | gate_up_suh 8,192 + down_svh 8,192 + intermediate_scales 884,736 + sign draw 288 |
| Coupled metadata, 42 layers | 151,436,544 | 0.0039798595 bpw, +0.0936 % over identity payload |
| Coupled files, 42 layers | 161,867,572,608 | 4 x 963,497,456 x 42 |
| Coupled files still to encode (39 layers) | 150,305,603,136 | layers 3, 20, 22 exist (11,561,969,472) |

EXL3 suh/svh vectors are real signed scales (gate/up suh: 71 unique values,
mean magnitude 0.0150; gate/up svh: 382-475 unique values, mean 1.08-1.10), so
they cannot be sign-packed. The 151,436,544 B must be paid for explicitly.
Per-expert rate steps at 25,165,824 weights per expert: K4->K3 saves
3,145,728 B, K4->K5 costs 3,145,728 B. **49 net K3 experts (0.41 % of 12,096)
pay for all coupled metadata.**

## 3. Gate A: storage (blocks any encoding)

`/media/brandonmusic/nvme1n1p3` is the NTFS volume (`fuseblk` on
`/dev/nvme2n1p3`), 1,999,473,995,776 B total, **127,677,571,072 B free**.
Campaign ledger v3: ceiling 30,000,000,000 B, charged 29,747,877,120 B,
remaining 252,122,880 B, cutoff 1788632160 (2026-09-05 14:16 EDT).

Full coupled K4 build needs, on the same volume as the runtime sidecar dir:

| Component | Bytes | Lifetime |
|---|---:|---|
| 39 coupled layers | 150,305,603,136 | permanent |
| Per-layer chunk set (4 x 72 experts) | 3,853,989,824 | transient, deleted after pack + verify (identity build policy) |
| Per-layer fit-only capture (64 windows, sparse) | 1,080,033,280 | transient, deleted after encode |
| Eval raw logits per window | 1,268,157,440 | transient, scored then retired |
| Runtime image v10 layer | about 130,000,000 | permanent |

Shortfall before transients: 22,628,032,064 B. With one layer of transients and
a 10 GB NTFS margin the volume needs **about 40 GB freed**, and the ledger
ceiling must be raised to at least **200,000,000,000 B** for this phase.
Nothing is deleted without approval. Reclaim menu (all under the campaign root
unless noted, all with results already recorded elsewhere):

| Candidate | Bytes | What it is |
|---|---:|---|
| `rotated-p8-mid-butterfly-l{3,20,22}-v1/dense` | 43.5 GB | decoded BF16 copies from the finished rotation sweeps |
| `decode-path-matched-control-v2` + `decode-path-fp8-ds-mla-p8-exl3-v2` raw | 20.3 GB | raw logits already scored (`*-kld` dirs retained) |
| `p8-coupled-three-layer-v1/chunks` | 11.6 GB | reproducible intermediates; hashes live in the sidecar receipts |
| `p8-smallm-scheduler-v1/index-order-full-v1` | 81.2 GB | index-order bug captures; result recorded |
| `fit-capture-l20-l22-v1` | 21.6 GB | only needed to re-encode layers 20/22, which are reused |
| `/media/brandonmusic/nvme1n1p3/glm53-rotation-v6,v7,v8` (outside campaign) | 146.7 GB | older rotation campaigns |

## 4. Gate B: rate menu and what the code supports today

Read from source, not inferred:

- `runtime_patch/p8_native_kernel.py`: sidecar `bits` must be `"4"` or `"5"`;
  `small_m_scheduler` requires K4 (`P8 K5 does not use the K4-only small-M
  specialization`); the scale component / full coupled boundary requires the
  small-M N128 owner path. **Therefore coupled layers are K4-only in the
  current runtime; K5 runs only uncoupled and without small-M.**
- `runtime_patch/b12x_h16/.../p8_small_m.py`: `self.trellis_bits = 4` is a
  constructor constant. `dynamic.py` accepts trellis_bits in {2,3,4,5} but the
  small-M gate requires `trellis_bits == 4`.
- One rate per layer per rank in the sidecar ABI (`w13_trellis` shape
  `[2, E, H/16, I/16, 16*bits]`). No per-expert or per-projection rate map exists
  in P8. SQG (`glm52_sqg_w4a8_runtime_release_20260812/.../glm_sqg_w4a8.py`)
  implemented per-projection K3/K4 as six compact pools with three
  expert-to-pool partitions and one route-packed launch per pool, on a
  different kernel family (`glm_trellis_w4a8`, codebook `sqg_xor_cheb_t12`).
- Encoder: `trellis_mxf.quantize_trellis_mxf_gptq` accepts bits 2-6;
  `quantize_hessian_trellis_layer.py` exposes `--bits {4,5}`; the coupled
  encoder hard-codes bits=4 and `SUPPORTED_LAYERS = (3, 20, 22)` in
  `p8_coupled_scale.py`, `quantize_p8_coupled_scale_layer.py`,
  `build_p8_coupled_scale_tp4_sidecars.py`, `prepare_p8_coupled_scale_v1.py`,
  `verify_p8_coupled_scale_tp4_sidecars.py`, and
  `scripts/run_p8_coupled_real_sidecar_loader_closure.py`.
- Runtime pins one `GLM53_P8_NATIVE_DESIGN` hash per process; the existing
  coupled layers carry the V3 design hash and identity layers carry the
  native6-v2 hash. A mixed model needs an explicit allowlist of design pins.

Consequence: with K4 fixed everywhere and coupling on every layer there is no
byte to redistribute. Shapley can then only choose equal-cost encodings. A
same-size allocation with real bit movement requires the mixed-rate menu
**K3 / K4 / K5 -> E4M3**, which needs the small-M coupled kernel generalized to
K3 and K5 stream words, a per-expert rate map in the sidecar ABI, and matching
encoder/packer/loader changes. This is the SQG precedent ported in concept, not
a code drop-in.

## 5. Predeclared Phase 0: uniform full coupled K4 (comparator, not deliverable)

Required by every allocation variant (it is the K4 tier and the
no-allocation comparator). Decision before result:

- Encode layers 4-19, 21, 23-44 with `quantize_p8_coupled_scale_layer` under
  the fixed transform `experiments/p8-coupled-transform-draw0-silu10-v1.json`
  (sha `093d219b...`), exact Flash EXL3 suh/svh per layer, draw 0, SiLU cap 10,
  `--samples 256`, fit role only (`roles-v5.json`; its 64 fit windows equal
  `roles-v3.json`'s), GPTQ-style inter-group feedback, no LDLQ. Four chunks per
  layer in parallel on the four GPUs, exactly as `scripts/build_uniform_p8_all42.sh`.
- Reuse layers 3, 20, 22 unchanged (same transform, scales, encoder, bits); the
  runtime accepts their V3 design hash alongside the 42-layer design hash.
- Prefetch fit-only captures per layer from
  `brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits` revision
  `95f4fdd94bf29989db2e0d1054e4931f55edb6aa` into the campaign root (not
  klcstore), delete after encode.
- Thermal: pause at 90 C, resume at 85 C (stricter than the identity build's 94/88).
- Measurement: 32 CF windows (roles sha `b5d7e452...`), 2,047 rows per window,
  row 0 excluded from true decode, FP64 CPU KL(teacher||student), paired BCa
  B=20000 seed 20260902, manifest window order. Arms: (a) full coupled K4,
  (b) full identity K4 re-measured on the same v10 image and env. Primary
  comparison (a) minus (b). Both arms must report 168/168 weights-ready and
  forward dispatch with the intended boundary in the runtime log.
- Report absolute KLD, paired delta, relative, BCa, window wins, domain
  breakdown, exact bytes (161,867,572,608 files / 4.2539798595 tensor bpw) and
  runtime labels. Claim boundary: already-opened CF32 development evidence.

## 6. Phase 1: same-size Shapley allocation (mixed rate)

Definition offered for approval; nothing here is implemented.

- **Players**: the 12,096 routed experts (layer, expert), each with two
  projection groups: fused gate/up (`w13`) and down (`w2`). Gate and up share a
  rate because the P8 w13 stream interleaves them atom-wise.
- **Admissible choices**: rate pairs (w13, w2) in {3,4,5}^2, coupling kept on
  every expert. Nine candidates per expert.
- **Byte constraint**: total codec bytes (trellis + UE8M0/32 + coupled
  metadata + headers) <= 161,715,585,024 + 168 x 728 header bytes, i.e. the
  identity checkpoint. Each K3 projection group funds bits elsewhere; the
  coupled metadata is paid by at least 49 expert-equivalents of K3.
- **Payoff for attribution**: ShapleyMCG quadratic attribution
  (`shapleymcg-fable-integration/.../scoring/attribution.py`,
  `quadratic_expert_attribution`): Fisher/Jacobian-projected residuals per
  candidate on the fit role, cross-expert terms shared symmetrically. This is
  the exact Shapley value of the quadratic routed-output damage game and the
  same estimator family as the GLM-5.2 SQG work. It is a local proxy; the
  end-to-end KLD is the validation payoff, never fitted.
- **Solver**: exact multiple-choice knapsack over bytes
  (`allocation/global_dp.py`), one global budget, no per-layer quota.
- **Validation**: one full-model CF32 KLD of the installed allocation against
  Phase 0 arm (a) at a reconciled byte budget; the measured effect is
  labelled "coupling plus allocation" unless arm (a) is retained as the
  no-allocation comparator, which it is.
- **Calibration roles**: fit only for encodes and attribution; CF32 for the
  single validation endpoint; selection/confirmation/final untouched.

Engineering deltas before Phase 1 can run:

1. Encoder: bits parameter in the coupled path (codec already supports 2-6).
2. Sidecar ABI v3: per-expert `w13_bits` and `w2_bits` tables, per-rate pools
   or per-expert stream offsets, exact byte receipt.
3. Runtime wrapper + small-M kernel: K3 and K5 stream words in the coupled
   M1 N128 owner path and the grouped M64/N128 path; dispatch by rate class.
4. Packer/verifier/loader-closure generalization; design-pin allowlist.
5. Candidate scoring pass: transient K3 and K5 encodes per layer, scored then
   deleted; selected experts re-encoded once (deterministic encoder).

Cost guide (from measured 327 s per 72-expert K4 chunk): K3 about 0.6x, K5
about 2x. Scoring encodes for all 42 layers: roughly 10-12 GPU-hours on four
GPUs; selected re-encode about 1-2 h. Kernel generalization is the long pole
and is unestimated until the small-M kernel is read in full.

## 7. Bounded schedule after Gate A approval

| Step | Wall clock | GPUs |
|---|---:|---|
| Tooling generalization to 42 layers, 42-layer design, executor arms, image v10 | 2-3 h (CPU) | 0 |
| 39-layer coupled K4 encode (7.5 min/layer measured) | about 5 h | 4 |
| Phase 0 KLD: coupled full + identity matched control | about 40 min | 4 |
| Phase 1 engineering | days | 1 for tests |
| Phase 1 scoring + allocation + validation | about 1 day | 4 |

## 8. Disclosures

- The three-layer pilot remains a failed development gate; it is not used to
  block Phase 0 and does not predict Phase 0.
- The 0.0373 identity control is on a different image and FC1 configuration
  than any coupled runtime; the matched control is re-measured.
- P8 is E4M3 mxf8f6f4 at twice NVFP4's MMA issue count; no speed claim.
- No CF32 window is excluded or rerolled; no protected role is opened.

## 9. Implementation status (2026-09-05 22:05 EDT, branch fable-p8-full-coupled-v1)

Done without GPUs, all tests passing:

- Coupled encoder tooling generalized to layers 3..44 (`SUPPORTED_LAYERS`,
  design generator with an explicit `--max-new-bytes` ceiling and
  `--already-encoded-layer` reuse, packer, verifier).
- `scripts/build_full_coupled_p8_all42.sh` plus
  `glm53_nvfp4/full_coupled_build_support.py`: fail-closed storage/production/GPU
  guard, hard-link reuse of layers 3/20/22 with provenance receipts, per-layer
  hash closure against packer and postwrite receipts, exact byte manifest.
- v10 runtime image lineage `runtime_patch/p8_full_coupled_image/`:
  identical to v9 except `GLM53_P8_NATIVE_DESIGN` accepts a colon-separated
  design allowlist and each sidecar's own design hash is verified and logged.
- `glm53_nvfp4/p8_full_coupled_runtime.py`, `scripts/prepare_p8_full_coupled_cf32_runtime.py`,
  `glm53_nvfp4/p8_full_coupled_cf32_executor.py`, `glm53_nvfp4/analyze_p8_full_coupled_cf32.py`
  and `experiments/p8-full-coupled-k4-cf32-v1.json`: sealed two-arm full-model
  CF32 protocol (coupled_full first, identity_full second) with a 168-pair
  runtime log gate that checks the per-layer design hash.

Still gated on the owner: the storage ceiling (needed to generate the 42-layer
design and start encoding) and the Phase 1 rate menu.

## 10. Autonomous campaign (owner directive 2026-09-05 ~21:20 EDT)

The owner delegated storage (protected: GLM-5.3-Flash NVFP4 stock, EXL3 4bpw,
EXL3 3bpw, BF16 until encodes finish), dropped the identity comparison, and asked
for the full coupled model with Shapley allocation and its end-to-end KLD, done
autonomously. Consequences recorded here as decisions before results:

- Stage-1 reclaim (`results/P8_STORAGE_RECLAIM_STAGE1_20260905.json`): identity
  K4 sidecars, rotation dense copies, scored decode-path raw logits and the
  pilot's chunk intermediates; 222.56 GB; campaign ceiling raised to 400 GB.
- Phase 0 runs `scripts/build_full_coupled_p8_all42.sh` (started 21:32 EDT).
- Phase 1 menu is **layer-granular K3/K4/K5 with coupling kept**. The sidecar ABI
  and the compiled small-M kernel carry one rate per layer; per-expert rates
  would need pool dispatch and are deferred. The parent b12x phase kernels in the
  image allow only bits 2-4, so K5 is served by the bit-parameterized P8
  subclasses in the v11 lineage (`runtime_patch/p8_mixed_rate_image`,
  `runtime_patch/b12x_mixed_rate`) with a K5-capable MCG decoder. Grouped M64
  prefill stays K4-only; non-K4 layers serve M>1 row by row through the M1 kernel.
- Payoff: fit-role routed-output damage with exact per-token Shapley shares
  (`glm53_nvfp4/p8_layer_rate_damage.py`); validated on layer 4 where the TP4
  rank inverse reproduced all 864 encoder NMSE receipts exactly. Smoke encodes:
  K3 weight NMSE 3.76x K4, K5 0.285x K4.
- Allocation: exact DP over #K5 - #K3 at the identity file budget
  (`glm53_nvfp4/p8_layer_rate_allocation.py`); at least one more K3 layer than K5.
- Orchestrator `scripts/run_phase1_campaign.sh`: uniform coupled KLD (v10) ->
  layer-3 candidates -> K3/K5 device closures (v11) -> all-layer candidates ->
  allocation -> assembly -> allocated-model KLD (v11). Logs under
  `<campaign>/phase1-campaign-v1/`.
