# Coupled carrier reference layout amendment

Date: 2026-09-05. Evidence level: CPU structural; no new GPU run.

## DECISIONS

1. Before the next device retry, canonicalize the reference quantizer payload
   from its exact `[rows, K/32, 32]` shape to the runtime wire `[rows, K]` shape.
   Apply the unchanged K32 permutation before reshaping. Require the expected
   input shape and uint8 dtype; do not flatten arbitrary tensors or reorder rows.
   This fixes a harness bug, not a numerical tolerance or kernel implementation.
2. Apply the same correction to the prefill reference cache: its former assignment
   of a blocked payload into `[prototype, 512]` was also shape-incompatible.
3. Preserve every prior failed device receipt. The first run's payload booleans
   alone were invalid byte tests. Diagnostic-v2 already compared flattened bytes:
   input 3/4096 mismatches, middle 4057/4096, scales 0/128 and 59/128 respectively.
   These failures remain unresolved. No claim of device closure follows from this
   amendment. Exact equality, numeric thresholds, fixture, seeds, and repeat count
   are unchanged. No teacher data or protected role was opened.

Validation: `python3 -m pytest -q tests/test_p8_coupled_m1_device_closure.py
tests/test_p8_coupled_prefill_device_closure.py`: 19 passed. New tests cover M1,
route and prototype shapes, exact byte preservation, one-bit negative control,
and rejection of an unexpected input shape.

Prior diagnostic receipt:
`/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-fixture-v1/m1-device-carrier-diagnostic-v2/carrier-failure.json`
SHA-256 `9cbdb80998a54601e1320f7944243be1e124670f9f3e8ef6df380f5602299d27`.

Runtime claim boundary: synthetic native P8 E4M3/UE8M0-K32, K4 payload 4.25 bpw
plus coupled metadata; no attention or KV path in this component test. P8 uses
`mxf8f6f4` at twice the NVFP4 MMA issue count. No KLD or speed result is produced.
