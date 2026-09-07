# Jovian integration: inspected, not yet implemented

Target inspected September 7: `local-inference-lab/vllm`,
`dev/jovian-judgement`, commit
`9a6b4fb3a6f5598fd2fb68cf0de92bfe145294c1`.
The published working RC5 image instead contains vLLM base commit
`6dc2f516688fe6f84c6994dcd20fddf296853a6c` plus its image patches, and B12X
`36bce2c1552ba2d47dc09f20a6f64fbfc8ec4ff8` plus the exported P8 kernels.

## Existing work to compose, not duplicate

- [PR 562](https://github.com/local-inference-lab/vllm/pull/562): EXL3 loaders,
  mixed Trellis, GLM MTP, B12X warmup; head inspected
  `97ba04f40bb48273762491749b3f0834ec32ad93`. This is not a demonstrated native
  E4M3 P8 coupled checkpoint path.
- [PR 566](https://github.com/local-inference-lab/vllm/pull/566): opt-in W4A8
  coupled QSRT prefill. It overlaps the coupled-boundary runtime design and
  needs review before introducing a separate general codec integration.
- [PR 563](https://github.com/local-inference-lab/vllm/pull/563): W4A16
  trellis prefill planning. Distinguish its compute/reconstruction format.

## Required port contract

The exported RC5 `sitecustomize.py` hooks `ModelOptNvFp4FusedMoE`,
`UnquantizedFusedMoEMethod`, and `FusedMoEMethodBase`, loads explicit sidecars,
and installs `P8NativeTPMoE`. Current Jovian exposes a custom quantization
registration API in `quantization/__init__.py` and ModelOpt methods for
`RoutedExperts`, including `apply` and `apply_monolithic`. These source
observations are integration hooks, not compatibility proof.

A focused port should register an explicit versioned TrellisMX checkpoint
method and fail closed on incompatible rates, alphabets, TP slices, coupled
boundaries, activation precision, sidecar identity, or architecture. Preserve
the existing native E4M3 kernels and validate target/MTP warmup plus graph
execution using Jovian's actual MoE interfaces. Keep generic codec metadata
separate from the GLM-specific architecture adapter. Do not silently fall back
to stock ModelOpt weights or claim EXL3 decoding establishes P8 correctness.

Required evidence before a support claim: CPU metadata/loader tests, all-rate
device closure, coherent target/MTP serving, matched JSON/log speed tests,
and KLD closure on the new runtime. The published RC5 results do not qualify
the new Jovian port. No PR has been opened for unimplemented compatibility.

The target's `AGENTS.md` additionally requires a human submitter to review
every changed line and run relevant tests, and disallows pure code-agent PRs.
Submission must wait for that human review; no review or sign-off is invented.

Existing release results and raw receipts are ready to attach to the future
PR as clearly labeled RC5 evidence, not as new-Jovian test results.
