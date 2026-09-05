# Coupled P8 three-layer CF32 executor critical audit

Date: 2026-09-05  
Audit mode: CPU/read-only; no GPU, build, model service, download, or deletion  
Executor tree: `/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-coupled-input-order-v1`  
Audited executor commit: `2c6bb0ecd2aa9c209639a8e828a638038c3d995f`  
Audited ledger-access delta: `6bbbdf9422441c98e726eea49499cb5e3f7c95e0`

Post-audit launch repair: `ebf9591be99f7110d8366045a115da47abce6385`

## Verdict

**PASS to create a fresh storage ledger and execution seal, then run the fixed
CF32 development comparison.** I found no remaining code or protocol blocker in
the audited source. This is not a device, KLD, speed, or qualification pass.
Production must remain off.

## Blocking-gate matrix

| Gate | Verdict | Exact evidence |
|---|---|---|
| Fixed three-arm lifecycle | PASS | `p8_coupled_cf32_executor.py:576-662` runs `stock`, `identity_p8`, then `coupled_p8`, stops on arm failure, authenticates the owned container, and requires cleanup success. |
| No production restore | PASS | `p8_coupled_cf32_executor.py:731-771` observes production before/after and contains no restore call; the sealed policy is checked at `394-397`. |
| Actual v9 image/environment/launch ABI | PASS | Runtime authentication regenerates every arm's exact argv and environment at `271-344`; image and installed-source identities are pinned in `p8_coupled_three_layer_runtime.py:15-38,155-181`. |
| Model and sidecar identity | PASS | The stock validator rehashes all 44 index-referenced shards and three otherwise unreferenced safetensors (`p8_coupled_three_layer_runtime.py:185-257`). Identity and coupled layer/rank sidecars are rehashed and metadata checked at `68-152`; coupled postwrite and real-loader closure are mandatory. |
| Teacher and input provenance | PASS | `p8_decode_protocol.py:36-113` admits only the frozen 32 conditional-fit windows, pins the Hub revision and manifest, rehashes each token file and each teacher file, and validates safetensors metadata. Token byte and decoded-value hashes are repeated before request and score (`p8_coupled_cf32_executor.py:504-523,609-639`). |
| First-row mask | PASS | Window receipts store 2047 rows and compute the primary true-decode mean from `kld[1:]` (`p8_coupled_cf32_executor.py:522-540`). Final audit requires `exclude_rows:[0], include:[1,2047]` and replay recomputes the stored mean (`665-728`). |
| Score before raw retirement | PASS | Score NPZ and durable JSON are written before the raw hash is rechecked and unlinked; a durable retirement receipt follows (`p8_coupled_cf32_executor.py:525-558`). Final analysis reauthenticates execution, cleanup, runtime audits, score NPZ, capture metadata, retirement, raw absence, and recomputed means (`665-728`). |
| Mandatory final log/order | PASS | Cleanup must preserve `server-final.private.log`; the final runtime audit must contain all 32 completed IDs in fixed order (`p8_coupled_cf32_executor.py:643-660,665-685`). |
| Campaign-wide 30 GB accounting | PASS in source; fresh value required | Seal and every request use the original-cutoff component ledger and account one exact raw peak (`p8_coupled_cf32_executor.py:214-265,350-452`). Delta `6bbbdf9` replaces the failed permission-sensitive workspace walk with a fail-closed, read-only `sudo -n find`, retaining the fixed workspace, cutoff, exclusions, regular-file filter, and conservative `*coupled*` inclusion. The actual regenerated ledger remains a prerequisite, not an audited result here. |
| Failure preservation and ownership cleanup | PASS | Arm and root execution records begin failed, record exception type, always attempt owned cleanup, and only advance on success (`p8_coupled_cf32_executor.py:585-662,731-771`). |

## Identity-arm comparability and claim boundary

The identity arm is the immutable prior all-42-layer P8 artifact, not literally a
newly encoded copy. This is **not a blocker for the preregistered whole-candidate
comparison**, because the layer-3 encoder receipts establish:

- identical BF16 source-index hash `e6007bd58fb7e07f9fe69544257ee2713f252ef5855bbf685b48c991d524ef0f`;
- identical activation-capture manifest hash `f1a6fe7b8828b3461e81ee533d417dd1524355ef6a205145850116560197f81a`;
- the same 64 fit IDs, domains, input hashes, and token paths in roles v3 and v5
  (the role-file hashes differ because v5 adds provenance fields and changes
  non-fit role inventory);
- the same K4 procedural-MCG, E4M3, UE8M0/32, static in-group activation order,
  GPTQ-style full-Hessian inter-group feedback, and two scale-refit iterations.

The coupled arm necessarily constructs Hessians from the transformed E4M3
carriers and adds the fixed EXL3-derived scale sandwich. Therefore a measured
difference supports only **coupled H512/H128 plus suh/svh plus its matched
transformed re-encoding versus the existing identity P8 artifact**. It cannot
causally isolate Hadamard rotation, scale metadata, or Hessian-carrier change.
The proposal already labels these as changing together; reports must preserve
that wording.

## Independent checks

- Focused plus adjacent executor/runtime tests: `69 passed in 1.34s`.
- Broad `tests/test_p8*.py`: `824 passed in 24.67s` at `2c6bb0e`.
- Ledger-access delta focused executor tests: `12 passed in 0.10s`.
- `py_compile` and `git diff --check`: pass.

No result from this audit authorizes production restoration or supports a KLD,
full-model, prefill, decode-throughput, or CUDA-graph claim. Those remain
device-execution outcomes.

## Post-audit v1 launch-failure repair

The first sealed attempt preserved a zero-window failure: its actual terminal
command began `exec exec /opt/venv/bin/python`, the owned stock container was
cleaned successfully, and production remained off. Commit `ebf9591` strips one
authenticated source-leading `exec`, rejects any residual `exec` or noncanonical
serve prefix, and emits exactly one leading `exec`; it does not alter serving
options or experimental conditions. It also adds the retained v1 attempt root
as a fixed `tree-apparent` ledger component, disjoint from the fresh v2 capture
root. The measured tree-apparent charge at audit time was 57,056 bytes including
directory inode sizes. Combined runtime/executor tests passed 24/24. The repair
requires a regenerated runtime manifest, ledger, seal, and fresh output root;
the v1 evidence must not be reused or overwritten.

## Candidate-first two-arm amendment

At the user's direction, protocol amendment
`experiments/p8-coupled-three-layer-cf32-candidate-first-v3.json`
(`fc896a8300ecd40c5b7fe1c2bc49c95634e17cf43810af0dd35d826ef7597ae1`)
was preregistered before any candidate window completed. Executor commit
`27805aae28f8fab7cbdc80b2e3d729e1c734d577` implements exactly
`coupled_p8 -> identity_p8`; there is no stock serving, scoring, comparison, or
inferred stock metric. Full stock-file authentication remains necessary because
the stock checkpoint is the common carrier beneath both P8 overlays.

The 32-window role/order, rows 1..2046 primary mask, paired BCa method
(20,000 replicates, seed 20260902), image/runtime topology, teacher hashes,
score-before-retirement lifecycle, and production-off rule are unchanged. The
amendment is validated, embedded in the runtime manifest, and included in the
execution seal's source hash set. The retained v1 output plus six actual v2
metadata files are fixed historical ledger inputs; their tree-apparent total at
audit time was 175,419 bytes. The nonexistent v2 capture directory is not
invented or used to reset accounting.

Independent focused analysis/executor/runtime tests passed 33/33, with
`py_compile` and `git diff --check` also passing. Verdict: **PASS for a fresh v3
manifest, ledger, seal, and candidate-first device run**, subject to all normal
runtime gates. The result can compare the whole coupled candidate only against
the existing identity P8 artifact; stock remains historical context.
