# Three-arm CF32 executor preflight

Status: CPU implementation complete; GPU execution not started.

The executor is `glm53_nvfp4/p8_coupled_cf32_executor.py`. It consumes the
sealed candidate-first two-arm runtime manifest and a second, explicit
execution seal. The only allowed order is coupled P8, then identity P8. Stock
serving is not tested; historical stock is context only and has no metric. A failed window or arm
raises before the next arm. No code path calls the older product `_campaign`
or `restore_if_safe`; the production user service, system timer and port 8000
are observed inactive before execution, before every arm/window, and after the
attempt. The root lock is acquired nonblocking for the whole attempt.

The execution seal binds the v9 image/attestation, the stock carrier receipt
and all 44 referenced shard hashes plus the three unindexed safetensors, all
identity and coupled rank sidecars, explicit postwrite and real loader receipts
for layers 3/20/22, CF32 role and teacher bytes, exactly regenerated launch
argv, source hashes, and the retained global capture inventory.

Storage authorization is a separately sealed, fresh external ledger generated
by `scripts/prepare_p8_coupled_cf32_storage_ledger.py`. It measures fixed
campaign-wide paths: all coupled worktrees, common Git growth, coupled files
outside those worktrees, Docker/containerd growth, image/build directories,
the fixture, three-layer output, historical quality root, and this capture
output. Growth is monotonic over projected component sizes. The historical
1,310,510,246-byte quality peak can cover only the unused portion after live
quality-root bytes, and only the active raw's actual capture-output overage.
The exact 1,268,157,440-byte next raw is prospectively charged before every
request, then the written peak is checked again before scoring. The ceiling is
30,000,000,000 bytes; there is no output-directory budget reset.

Each token file and its decoded token-value sequence are rehashed immediately
before every request and again during scoring. Each window is forced-decode
captured and scored immediately. Acceptance
requires exactly 2047 finite aligned KLD rows. Row 0 is recorded separately as
one-token prefill and only rows 1..2046 enter true-decode KLD. The exact raw
size and SHA-256 are durably bound by the score receipt before the sole
destructive operation: unlinking that canonical raw file. Metadata, scores,
retirement receipts and private request/response evidence remain. There is no
dense-32 capture and no speed loop.

Runtime logs must prove TP4/DCP1, no expert parallelism, no MTP, graphs,
B12X_MLA_SPARSE, NVFP4 MLA KV, all four capture hooks, and exact P8
weight/forward pairs for layers 3/20/22 and ranks 0..3. Final logs must contain the ordered 2047-row rank-0
completion marker for every CF32 window.

CPU validation:

```
PYTHONPATH=$PWD python3 -m pytest -q \
  tests/test_p8_coupled_cf32_executor.py \
  tests/test_p8_coupled_three_layer_runtime.py \
  tests/test_p8_coupled_cf32_analysis.py \
  tests/test_tail_v2_product_validation.py \
  tests/test_p8_decode_launcher.py \
  tests/test_p8_decode_protocol.py
```

Result: 66 passed. The destructive-path tests use a four-byte synthetic raw.
They prove durable score-before-retire ordering and that a receipt-write error
leaves the raw intact. A mocked lifecycle test injects an identity-arm failure
after coupled completes, proves there is no later arm, and makes any
restoration call fail the test.
The final broad `PYTHONPATH=$PWD python3 -m pytest -q tests/test_p8*.py`
regression suite passed 824 tests in 24.96 seconds. The three adjacent Tail-V2
capture/protocol files added another 39 passes in 0.95 seconds.

Remaining device gate: after all 12 sidecars and their real-loader receipts are
present, generate the prepared manifest and execution seal, review both hashes,
then explicitly invoke `python3 -m glm53_nvfp4.p8_coupled_cf32_executor execute`.
No GPU, image build, container, service, or production state was changed while
preparing this implementation.
