# Corrected-index P8 full32 numerical and KLD campaign

Status: design before target collection. No full32 student capture or KLD
result exists for this corrected decode path at the time of sealing.

## Decision and prerequisite

The three-process canary under plan
`5514b3f29db3aad7e88327f1af336013b35f951aa49c96d3846bd0d0a7833e0f`
passed. Two corrected N128 processes and corrected N64 produced identical
2047x154880 raw logits, with root execution SHA256
`4836ed7d6d94880c6bef90433adddaf55ff7fe5977a36b7dafa711ba6d1f30a6`.
This protocol authenticates and replays those three captures but does not
rerun, copy, or synthesize canary stages.

The decision question is whether two fresh full-panel serving processes—N128
with separate scratch, then optimized N64 with consolidated scratch—retain
bitwise equality over the complete existing conditional-fit32 role, and what
absolute teacher KLD the corrected P8 checkpoint has on those windows.

## Frozen collection and analysis

- One fresh TP4/noEP/DCP1 process per arm, in fixed N128 then N64 order.
- All 32 conditional-fit windows, eight per domain, 2,047 forced causal rows
  each: 65,504 rows and all 154,880 real-vocabulary logits per arm.
- Same all-42-layer K4 procedural MCG to E4M3 P8 checkpoint, NVFP4 MLA KV,
  Model Runner V2 and FULL CUDA graphs. Corrected logical short-pool placement
  and one source-qualified receipt per rank; heavy trace observer disabled.
- Bitwise full-logit comparison is reported independently of quality. A
  mismatch remains a closure failure even if N64 KLD is numerically favorable.
- CPU FP64 `KL(teacher || student)`, equal-window mean, plus paired 20,000-
  replicate window BCa interval with seed 20260905. The pinned BF16 teacher,
  role manifest, tokens, logits and metric identity are checked before use.
- No retry/resume or row exclusion. Partial/failing artifacts are preserved.
  Capture, cleanup and restoration failure prevents scoring. A numerical
  N128/N64 mismatch does not erase valid complete captures or their KLD.

The full captures require exactly 81,162,076,160 raw bytes; execution requires
that amount plus 20 GiB free. Start temperature must be at most75C and any
sample at or above90C aborts. The systemd user unit supplies a separate
six-hour `RuntimeMaxSec` operational limit; it is not a scientific estimand.

## Interpretation boundary

This can establish corrected-P8 absolute development KLD and N128/N64
equivalence under the tested forced-M1 path. It cannot establish improvement
versus matched stock NVFP4 because no NVFP4 arm is in this protocol, cannot
qualify protected selection/confirmation/final data, and is not a speed run.

The logical-order correction covers retained compressed pools of length at
most512. The backend still consumes2048 of2051 expanded columns, so the
incomplete-KPool-tail omission remains unchanged and its KLD impact is not
isolated. Passing this campaign therefore does not complete final quality
qualification. P8 `mxf8f6f4` still issues twice the MMA count of NVFP4 for
equal K; it is the quality product, not P4's speed class. No LDLQ is used.

Existing attribution for ExLlamaV3, KQuant, QSRT, `w4a8_trellis` and B12X is
retained in `THIRD_PARTY_NOTICES.md`.

## Review and reproduction

Three GPT-5.6 SOL high-reasoning agents independently reviewed the runner,
analysis and test surface. One unsupported in-plan wall-time assertion was
removed before sealing. Final verdicts were GO with no remaining code-level
fatal issue. The focused runner and analyzer suite passes134 CPU tests; the
larger relevant suite will be recorded before launch.

Prepare with:

```bash
python3 -m glm53_nvfp4.p8_index_order_full plan \
  --plan <fresh-absolute-plan> --output <declared-absolute-output> \
  --roles <pinned-conditional-fit32-role> --teacher-root <pinned-teacher-root>
```

Execute once from a clean sealed checkout using
`python3 -m glm53_nvfp4.p8_index_order_full run-and-analyze --plan <plan>`.
The exact plan SHA256 and resulting receipts are appended only after they
exist; target results never rewrite the frozen decision rule.
