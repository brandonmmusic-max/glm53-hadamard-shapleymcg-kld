# P8 coupled encoder/runtime compatibility audit

Date: 2026-09-05  
Scope: CPU/static only; no GPU, build, service, encoding, or protected-role access.  
Encoder base: `60d49055248ebf103a56bb4ae911dc389ed141c5` plus the changes committed with this report.  
Runtime: `fe1df7a2695a975bce632a613a12756405ed2c84` (`dcfc5753` implementation plus immutable cast-boundary tests/report).  
Runtime CPU-contract SHA-256: `5e9220740c30bfbcb47ab0ea0857f820562b95e6a0797ff0fb0868eb6f84a0f2`.

## Verdict

The corrected encoder and the frozen runtime are compatible at the CPU/schema
boundary. The cross-tree audit passes the runtime's real fail-closed validator,
generates identical draw-0 rank-local sign bytes, and is bit-exact for the
complete pre-E4M3 coupled topology. There is no remaining CPU-contract blocker
before device build.

This is not device closure or product qualification. The runtime supports only
K4, M1, N128, TP4 on this full-coupled path. M2/M3 and prefill are fail-closed,
so M1-only support blocks production and prefill qualification regardless of
the CPU pass. No layer has been encoded with this corrected V2 design yet.

## Pass/fail matrix

| Contract item | Result | Exact evidence |
|---|---|---|
| Target activation | PASS | Target config SHA `4f5341e0...f265` declares `hidden_act=silu` and `swiglu_limit=10` at `config.json:33,267`; published target runtime SHA `11617266...5e67` clamps gate/up and applies `F.silu(gate)*up` at `glm53_tp2_exl3.py:632-636`. |
| Candidate draw | PASS, preregistered not tuned | V2 fixes all 288 experts in layers 3/20/22 to draw 0. The archived draw 6/SiTU occurrence is a synthetic fixture at `test_fused_moe_trellis.py:1441-1467`, not a target-Flash selected-draw receipt. |
| Cast order into FC1 A8 | PASS | Encoder keeps `H512*suh -> H128` in FP32 for `quantize=True` at `p8_coupled_scale.py:585-594`. Archived quantized reference does the same at `test_fused_moe_trellis.py:953-960`. Runtime full input arm states and implements this order. |
| Cast order into FC2 A8 | PASS | Encoder keeps `activated -> sign -> H128 -> down_suh -> H128` in FP32 through E4M3 at `p8_coupled_scale.py:620-630`. Archived quantized reference does this at `test_fused_moe_trellis.py:1038-1046`. Runtime full FC1 arm stores the post-H128 values as FP32 before quantization. |
| Weight coordinate transform | PASS, CPU | Encoder inverses the H512/suh/H128 input basis, coupled FC1 atom basis, sign basis, and down/output bases at `p8_coupled_scale.py:526-576`. Full cross-tree reference output is bit-exact before E4M3. |
| Weight tensor names/shapes | PASS, static | Encoder emits `w13_trellis`, `w2_trellis`, `w13_scale_ue8m0`, `w2_scale_ue8m0`; runtime validates `[2,E,H/16,Ilocal/16,64]`, `[E,Ilocal/16,H/16,64]`, `[E,2Ilocal,H/32]`, `[E,H,Ilocal/32]` at `p8_native_kernel.py:173-196`. |
| FC1 projection/scale order | PASS, static | Encoder stores trellis projection 0=gate, 1=up and physical weight scale planes `up|gate` at `build_p8_coupled_scale_tp4_sidecars.py:211-215`. Runtime stages projection 0 as gate and 1 as up, while its inherited packed scale geometry indexes up first and gate second (`p8_h128_fc1.py:314-379`). |
| Coupled scale names/order | PASS | `gate_up_suh_fp16[H]`; `intermediate_scales_fp16[E,gate_svh|up_svh|down_suh]`; `down_svh_fp16[H]`, assembled at encoder builder lines 216-220 and split in that order by runtime. |
| TP ownership | PASS, static | Encoder slices gate/up output rows, down input columns, and all private I scales by one contiguous I/4 interval (`build...py:153-196`), replicating the two H-wide shared scales. Runtime requires `global_intermediate=Ilocal*4`, `local_atom_begin=rank*(Ilocal/32)`, and contiguous atom32 slicing. |
| Sign semantics | PASS | Schema fixes draw 0, axes 1/2 and packed `pre[2Ilocal]|post[Ilocal]`. Encoder/runtime generated FP16 bytes match. For I=512 every rank hash is `95693b3d933cef1caca1d4aed176789faff9746303ec5f8bcac4395c68e25ff0`. |
| Sidecar schema and hashes | PASS, synthetic | Sidecar schema is `glm53-p8-coupled-h512-h128-tp4-rank.v1`; it includes hashes for all stored tensors, the regenerated sign vector, source design, exact scale source, and transform receipt. Runtime's actual `validate_coupled_component` accepted the synthetic encoder header. |
| Exact Flash scale source | PASS, preparation | Sealed source receipt SHA `092be1ff...2643`; exact E288/H4096/I2048 loader validation passed for layers 3/20/22. The per-layer aggregate tensor hashes are preserved in the V2 preparation receipt. |
| Weight payload rate | PASS | K4 trellis plus UE8M0/32 is exactly 4.25 bpw. The codec stream ABI was not changed. |
| Device bit-exact E4M3 closure | PENDING/BLOCKING | No SM120 build or GPU run was allowed. Must compare decoded trellis words, UE8M0 bytes, both A8 carriers, and final route output against the pinned CPU oracle. |
| Prefill/product support | FAIL/BLOCKING | Full-coupled runtime is M1/N128 only. No prefill path exists, so it cannot yet be called a production P8 product or benchmarked as such. |

## Immutable transform and design receipts

- Transform receipt:
  `experiments/p8-coupled-transform-draw0-silu10-v1.json`
  SHA-256 `093d219b18ba32471adcee746442b1481c7ba5659bbea62b94f1a665d4343a12`.
- V2 preparation:
  `results/P8_COUPLED_SCALE_FLASH_EVIDENCE_PREPARATION_V2.json`
  SHA-256 `2505ffca4d74c4439c7ad89babcd9e625ff1b3e0f11bf1dec7f1ca2e70ce01ee`.
- Exact serving pins must be:
  `GLM53_P8_NATIVE_TRANSFORM=<transform receipt>` and
  `GLM53_P8_NATIVE_DESIGN=<V2 preparation receipt>`. The runtime hashes both
  files and rejects a sidecar whose metadata differs.

## Metadata accounting

Per TP rank per layer, there are 1,811,939,328 logical routed weights.

| Item | Bytes | bpw contribution |
|---|---:|---:|
| FP16 scales | 901,120 | 0.003978587962963 |
| Stored one-byte draw IDs | 288 | 0.000001271565755 |
| Stored tensor metadata total | 901,408 | 0.003979859528718 |
| Runtime-regenerated FP16 signs | 3,072 | 0.000013563368056 |
| Full coupled accounted total | 904,480 | 0.003993422896774 |

Thus the current physical tensor sidecar is 4.253979859528719 bpw, while a
conservative accounting that also charges the regenerated resident signs is
4.253993422896774 bpw. Safetensors headers and receipt JSON are separate file
overhead. The earlier 0.00397986 figure was scales plus draw IDs, not scales
alone. The runtime report's 904,192-byte scale-plus-sign number omits the 288
stored draw IDs; both numbers are now explicitly reconciled.

The three-layer storage preflight is in
`results/P8_COUPLED_PILOT_STORAGE_PREFLIGHT.md`. Dense output is prohibited:
three-layer BF16 dense copies alone require 43,486,543,872 bytes, exceeding the
30,000,000,000-byte campaign ceiling. The nominal chunk-plus-TP4 forecast is
23,123,890,176 bytes, leaving 6,876,109,824 bytes for headers, temporary files,
caches, receipts, and other campaign growth. Actual aggregate growth/free-space
checks remain mandatory immediately before launch.

## Reproduction and remaining gates

CPU compatibility:

```bash
PYTHONPATH=. python3 scripts/audit_p8_coupled_encoder_runtime.py \
  --runtime-module /home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-coupled-scale-kernel-v1/runtime_patch/p8_coupled_scales.py
```

Expected result is `status=pass`, runtime commit `fe1df7a...`, transform SHA
`093d219b...a12`, validator pass, and bit-exact pre-E4M3 topology.

Before any quality claim: encode only layers 3/20/22 from the V2 design;
construct real TP4 sidecars; pass runtime validation for every rank/layer;
build on SM120; close both E4M3/UE8M0 carriers and final output against the CPU
oracle; then run the preregistered layer/KLD comparison. Before speed or product
claims, implement and close M2/M3 and prefill, then run the matched cold TP4
campaign. The CPU/static pass alone does not support KLD or speed claims.
