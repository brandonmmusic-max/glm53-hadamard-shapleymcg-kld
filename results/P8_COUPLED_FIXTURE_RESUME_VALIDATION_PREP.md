# P8 coupled synthetic fixture: immutable resume validation

Date: 2026-09-05  
Role: CPU-only preparation; no device or quality claim

## Outcome

The direct-script package import defect is fixed and an explicit
`--resume-validate-existing` action is prepared. It does not call the writer.
It accepts only the already-present rank-0 sidecar and design at their frozen
names, rejects partial files or an existing success receipt, and writes only a
new `receipt.json` after every validation passes.

The original failure record is preserved at
`evidence/preparation/p8-coupled-fixture-v1/generation-failure.json` with SHA256
`cfa9469ba816ae1db41db082f0fca36c96100ee79b511fe467d0da65829a3901`.
It pins:

- sidecar: 963,497,568 bytes,
  `c37ecf60ce9d5689292c92067494ed2df6d00875c727624dcaaaa4980efb2433`;
- design: 714 bytes,
  `deb72317a1f860e769bfc681190a851cf0d261a460761612fc0044c5295ce74a`;
- original external-budget receipt:
  `fb122344c3c92d88fbe482ef6f86977509ff6e6e42d7f858d0b05b1cca3ddf38`.

## Validation contract

The resume action fails closed unless all of the following hold:

1. The output directory, budget root, transform, failure receipt and historical
   external-budget receipt have the required absolute identities.
2. The sidecar and design exist under the frozen names; no payload partial,
   receipt, or receipt partial exists.
3. The design is byte-for-byte the canonical seeded design and the sidecar has
   the exact frozen size.
4. The two preserved artifact mtimes fall inside the original budget receipt's
   authorization interval. The expired receipt is historical evidence only and
   is explicitly not renewed into permission for a payload write. Its mutable
   nested worktree evidence path is not reinterpreted as current evidence.
5. The current budget-root inventory plus the archived 4,000,000,000-byte
   external upper bound plus the 1 MiB receipt allowance remains at or below
   the 30 GB campaign ceiling; current filesystem free space also covers the
   allowance.
6. Every expected tensor SHA256 is independently regenerated from seed
   `20260905129` in at-most-1-MiB chunks. The existing sidecar is then read in
   bounded chunks and must have the exact header, offsets, tensor hashes and
   whole-file hash.
7. The existing CPU runtime validator must accept schema, shapes, scale roles,
   draw0 transform, signed non-unit suh/svh, nonidentity UE8M0 codes, and finite
   exact K4 MCG E4M3-times-UE8M0 decodes for gate, up and down.
8. The actual whole-file and design hashes must equal the immutable failure
   receipt. Only then is the success receipt published with create-if-absent
   hard-link semantics and a directory fsync.

No BF16 plane, model, calibration artifact, GPU, service, or quality evidence is
used. Expected-byte regeneration and actual-file validation are streaming; no
duplicate full tensor plane is retained.

## Prepared command

Run from any directory; no `PYTHONPATH` is required:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python3 /home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-coupled-integration-v1/scripts/generate_p8_coupled_m1_synthetic_fixture.py \
  --resume-validate-existing \
  --budget-root /media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-fixture-v1 \
  --output-dir /media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-fixture-v1/rank0-v1 \
  --external-budget-receipt /home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-coupled-integration-v1/experiments/p8-coupled-fixture-external-budget-v1.json \
  --external-budget-sha256 fb122344c3c92d88fbe482ef6f86977509ff6e6e42d7f858d0b05b1cca3ddf38 \
  --failure-receipt /home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-coupled-integration-v1/evidence/preparation/p8-coupled-fixture-v1/generation-failure.json \
  --failure-receipt-sha256 cfa9469ba816ae1db41db082f0fca36c96100ee79b511fe467d0da65829a3901
```

This exact command is a resume-validation operation, not generation. Do not add
`--generate`, remove existing files, or rerun it after `receipt.json` exists.

## CPU tests

`pytest -q tests/test_generate_p8_coupled_m1_synthetic_fixture.py` passes 11/11.
Coverage includes direct absolute-script package imports, unchanged payload
hashes and mtimes after a small-shape recovery, one-shot receipt publication,
and fail-closed seeded corruption with no receipt creation.

