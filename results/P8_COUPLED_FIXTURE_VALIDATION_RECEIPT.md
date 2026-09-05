# Coupled P8 full-shape fixture validated in place

2026-09-05. Structural CPU evidence only; no GPU or model quality claim.

The first generation wrote the full sidecar but failed at the repository
package import during post-write validation. Recovery from source
`c76b251102413ad8d8187733bc4c4d3354856345` completed with exit zero and
`status=validated_existing_payload`. It did not regenerate or overwrite the
sidecar or design.

- Sidecar: 963,497,568 bytes, SHA256
  `c37ecf60ce9d5689292c92067494ed2df6d00875c727624dcaaaa4980efb2433`.
- Canonical design SHA256:
  `deb72317a1f860e769bfc681190a851cf0d261a460761612fc0044c5295ce74a`.
- Success receipt SHA256:
  `1bbdafe93f7a351087217e8061202d702e47e537c76d157d89fe9f4a1a5f5248`.
- Portable success receipt:
  `evidence/preparation/p8-coupled-fixture-v1/validation-receipt.json`.
- Preserved failure receipt:
  `evidence/preparation/p8-coupled-fixture-v1/generation-failure.json`.

All seeded tensor hashes were independently recomputed in bounded chunks and
matched the existing file. Header, offsets, whole-file hash, CPU runtime
schema, scale roles and fixed draw passed. Expert-0 gate/up/down decoding was
finite and exactly representable as E4M3 times UE8M0 for the tested values.
This last decode check is not an all-expert numerical correctness claim.

The command started while the external budget receipt was still current;
its conservative external charge was 4,000,000,000 bytes. Existing fixture
bytes plus that charge and the receipt allowance totaled 4,964,546,858 bytes,
under the 30,000,000,000-byte ceiling. Recovery wrote only the new receipt.

Ready inputs for the future device gate:

- Image: `sha256:77ae1c0b46c7f72b085b6ae3f8184f0f9df229bce5306d1849fb770501dc0d64`.
- Installed runtime manifest:
  `ff5ce6158e2d2134cedd1a451bc911fc3623c3ba93513916556a61833ee496d2`.
- Transform:
  `093d219b18ba32471adcee746442b1481c7ba5659bbea62b94f1a665d4343a12`.
- Fixture directory:
  `/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-fixture-v1/rank0-v1`.

No GPU launch is implied by readiness. Preserve the existing quality and speed
schedule; then require idle-device and current storage gates before M1 and
M2/M64/M65 closure. Attention/KV: not exercised here. Intended MoE: native P8
E4M3 weights/activations, UE8M0/32, K4 payload 4.25 bpw plus channel-scale
metadata. P8 uses twice NVFP4's MMA issue count. KLD, speed, CUDA-graph parity
and full-model serving remain untested for this coupled candidate.
