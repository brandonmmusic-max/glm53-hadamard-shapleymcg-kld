# Three-arm CF32 executor preflight

Status: CPU implementation complete; GPU execution not started.

The executor is `glm53_nvfp4/p8_coupled_cf32_executor.py`. It consumes the
sealed three-arm runtime manifest and a second, explicit execution seal. The
only allowed order is stock, identity P8, coupled P8. A failed window or arm
raises before the next arm. No code path calls the older product `_campaign`
or `restore_if_safe`; the production user service, system timer and port 8000
are observed inactive before execution, before every arm/window, and after the
attempt. The root lock is acquired nonblocking for the whole attempt.

The execution seal binds the v9 image/attestation, stock carrier config and
index, all identity and coupled rank sidecars, explicit postwrite and real
loader receipts for layers 3/20/22, CF32 role and teacher bytes, launch argv,
source hashes, and the retained global capture inventory. It reserves a
30,000,000,000-byte global future-capture budget before execution. This is a
global retained-inventory gate, not an output-directory reset.

Each window is forced-decode captured and scored immediately. Acceptance
requires exactly 2047 finite aligned KLD rows. Row 0 is recorded separately as
one-token prefill and only rows 1..2046 enter true-decode KLD. The exact raw
size and SHA-256 are durably bound by the score receipt before the sole
destructive operation: unlinking that canonical raw file. Metadata, scores,
retirement receipts and private request/response evidence remain. There is no
dense-32 capture and no speed loop.

Runtime logs must prove TP4/DCP1, no expert parallelism, no MTP, graphs,
B12X_MLA_SPARSE, NVFP4 MLA KV, all four capture hooks, and exact P8
weight/forward pairs for layers 3/20/22 and ranks 0..3. Stock MoE backend and
the available activation/checkpoint evidence are copied from emitted logs;
they are not inferred. Final logs must contain the ordered 2047-row rank-0
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
leaves the raw intact. A mocked lifecycle test injects an identity-arm failure,
proves coupled is never entered, and makes any restoration call fail the test.
The broader `PYTHONPATH=$PWD python3 -m pytest -q tests/test_p8*.py` regression
suite also passed: 821 passed in 24.74 seconds.

Remaining device gate: after all 12 sidecars and their real-loader receipts are
present, generate the prepared manifest and execution seal, review both hashes,
then explicitly invoke `python3 -m glm53_nvfp4.p8_coupled_cf32_executor execute`.
No GPU, image build, container, service, or production state was changed while
preparing this implementation.
