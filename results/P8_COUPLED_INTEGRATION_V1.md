# Coupled P8 integration V1

2026-09-05. CPU/static preparation only; no GPU build, encoding or candidate
KLD capture. Branch: p8-coupled-integration-v1.

Integration starts at baseline commit 8cf3d3c, preserving its tail-V2 runner.
Two non-fast-forward merges retain contributor history:

- Encoder 23a32edaf8aa98333528fe02af3067b9868baa56: target Flash scales,
  fixed draw0/capped-SiLU transform, FP32 activation carriers and TP4 sidecars.
- Runtime fe1df7a2695a975bce632a613a12756405ed2c84: legal N128 ownership,
  full coupled H512/H128 transforms and native K4 P8 MMA path for M1.

Combined merge tree: eded4c349f0bec82634ccf0fa0749be833665670. Git reported
no textual conflicts. The independent compatibility audit supplied the
shared activation/cast/scale/sign contract before integration. No contributor
behavior was discarded. The active baseline stays in its original clean
worktree and uses its original sealed source hashes.

Validation in this combined worktree:

- scripts/audit_p8_coupled_encoder_runtime.py: actual runtime validator PASS,
  transform SHA 093d219b18ba32471adcee746442b1481c7ba5659bbea62b94f1a665d4343a12,
  pre-E4M3 topology bit-exact.
- tests/test_p8*.py, test_build_p8_coupled_scale_tp4_sidecars.py and
  test_tail_v2_product_validation.py: 717 passed, one failure.
- The failing lifecycle-order test touched the live campaign's occupied
  port despite mocking its container/network workflow. Its temporary execution
  receipt recorded OSError before any window. Mocked only that test's port
  availability check; retained the separate actual-socket listener-rejection
  test. Re-ran all eight tail lifecycle tests: 8 passed.
- git diff --check: PASS.

Remaining device gates: encode V2 pilot layers 3/20/22, verify actual TP4
sidecars, compile SM120 image, close codec/scale/A8 carrier bytes and outputs,
then run matched KLD. Full coupled currently rejects M>1. Prefill kernels are
under development in a separate branch and must pass their own closure.

P8 is E4M3/UE8M0-32 mxf8f6f4 with 4.25 bpw weight payload and approximately
4.25398 bpw including stored scale/draw metadata. Its MMA issue count is twice
NVFP4. This report establishes no coupled KLD or speed improvement.
