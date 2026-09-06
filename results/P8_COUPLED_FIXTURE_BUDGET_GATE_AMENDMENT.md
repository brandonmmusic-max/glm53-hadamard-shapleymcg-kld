# Fixture generation: campaign-wide budget gate amendment

Prepared 2026-09-05 before full fixture generation or candidate device results.
This changes only generation authorization/accounting. Geometry, seed, code
streams, scale vectors, synthetic design and expected sidecar byte count are
unchanged. The original generator hash recorded in the prefill preregistration
identifies its preparation snapshot, not this amended budget-check source.

The former directory-only check could pass in an empty fixture directory while
omitting campaign worktrees, shared Git objects and Docker/cache growth. Full
generation now also requires `--external-budget-receipt` and
`--external-budget-sha256`. The receipt must bind the exact fixture budget root,
30,000,000,000-byte ceiling, a nonnegative external upper bound, an unexpired
measurement interval and hash-verified audit evidence. Its identity and charge
are preserved in the generation receipt.

The enforced inequality is:

```text
external campaign upper bound + current fixture-root bytes
  + exact sidecar/design forecast + receipt allowance <= 30,000,000,000
```

The external bound must include concurrent campaign growth and a conservative
reserve until its expiration. A hash proves identity, not the adequacy of that
bound; the operator must still review the audit. An expired or mismatched
receipt, changed evidence, missing external charge, or over-budget forecast
fails before output directory creation. No full fixture has been generated.

Updated command shape (supersedes the earlier two-argument generation example):

```bash
python3 scripts/generate_p8_coupled_m1_synthetic_fixture.py --generate \
  --budget-root /absolute/fixture-root \
  --output-dir /absolute/fixture-root/rank0-v1 \
  --external-budget-receipt /absolute/reviewed-budget.json \
  --external-budget-sha256 <SHA256>
```

Validation: 8 fixture tests pass, including empty-root over-budget rejection,
external charge arithmetic, receipt/evidence drift, root mismatch, expiry and
missing-receipt rejection before writes. No GPU, model encoding, or quality
measurement was performed by this amendment.
