# P8 real layers 3/20/22 TP4 loader-ABI closure preflight

Date: 2026-09-05  
Scope: CPU-only implementation and tests. No GPU, kernel, service, image build,
model encode, or large write was performed.

## Prepared gate

`scripts/run_p8_coupled_real_sidecar_loader_closure.py` prepares one opt-in,
sequential GPU probe of four real TP4 sidecars for an explicit fixed
`--layer` in `{3,20,22}`. The default command only prints the original frozen
layer-3 protocol. Its digest remains
`bd9dbf3273445152ad93e743c2d7ca229c458ed5254045b35d269e2ff32a2b5f`,
so the prior layer-3 result and launch receipts remain immutable evidence.
Layer 20 and layer 22 each derive a separate protocol digest from the explicit
layer: `a17482a6721c8f83872a429138eb8d7a56632d3b5ec881ee12d7d52062de68e5`
and `ca8d5a391725d2cf1da058d54dca236de703f0716d04bad83d05edcc2605dec9`,
respectively. Execution and probe modes reject a missing or unsupported layer.

The probe uses immutable image
`sha256:ad6b26bf6d1f265d99b09383485ddef82a4acfaea43e28af46341ebb41da24e3`
and runtime manifest
`9a57438b3cefd022772bc471980fb0ece8c19087d08f02c875a73da2b6e392d1`.
It externally pins the V3 design to
`4ebb96dd9d555fc18f24fb5f4d380216e1de30327a68d7aae89fc10f41878695`
and the fixed draw0 transform to
`093d219b18ba32471adcee746442b1481c7ba5659bbea62b94f1a665d4343a12`.

For each rank in fixed order 0, 1, 2, 3 it:

1. verifies the source-exact postwrite PASS receipt, sidecar whole-file hash,
   and all eight tensor shapes, dtypes and hashes;
2. reopens and independently hashes all eight stored tensors;
3. constructs the actual image-v9 `P8NativeTPMoE` with layer/rank/design and
   transform pins, `full_coupled=True`, K4, N128 and the deterministic TP4
   owner;
4. hashes the seven weight/scale tensors after their actual runtime device
   materialization and compares them with the postwrite hashes;
5. verifies the stored draw0 byte tensor and separately hashes the runtime's
   regenerated FP16 sign vector; and
6. requires the descriptor carriers to alias the trellis stream storage,
   rejecting an identity or alternate-weight fallback.

Ranks are destroyed and the CUDA cache released between constructors. The
harness does not call `_compile` or `__call__`; no MoE/MMA kernel executes.
The unchanged constructor's scale-repacking implementation may launch device
work, so this is not labeled a no-kernel test.
Its evidence label is loader/ABI closure only—not numerical closure, CUDA graph
closure, KLD, throughput, serving, or full-model qualification.

The postwrite verifier was extended to record source-derived `shapes` and
`dtypes` for all tensors in each rank. This is required so the loader harness
does not infer ABI expectations from the sidecar it is testing.

## CPU validation

`PYTHONPATH=$PWD python3 -m pytest -q tests/test_p8*.py` passed **790 tests**.
Focused tests cover the CPU-only default, exact four-rank/file binding,
sidecar metadata/tensor corruption, fake-wrapper runtime materializations,
descriptor aliasing, and the networkless v9 launch command.

## Prepared invocation, not run

Run only after all four sidecars and the postwrite closure receipt exist, GPU 0
is idle, and the parent execution authority allows the probe:

```bash
python3 scripts/run_p8_coupled_real_sidecar_loader_closure.py --execute \
  --image sha256:ad6b26bf6d1f265d99b09383485ddef82a4acfaea43e28af46341ebb41da24e3 \
  --gpu-device 0 \
  --layer 3 \
  --sidecar /media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-three-layer-v1/sidecars/layer-003/p8-layer-003-tp4-rank-0.safetensors \
  --sidecar /media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-three-layer-v1/sidecars/layer-003/p8-layer-003-tp4-rank-1.safetensors \
  --sidecar /media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-three-layer-v1/sidecars/layer-003/p8-layer-003-tp4-rank-2.safetensors \
  --sidecar /media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-three-layer-v1/sidecars/layer-003/p8-layer-003-tp4-rank-3.safetensors \
  --postwrite /media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-three-layer-v1/receipts/layer-003-postwrite-closure.json \
  --design /home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-coupled-integration-v1/results/P8_COUPLED_SCALE_FLASH_EVIDENCE_PREPARATION_V3.json \
  --transform /home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-coupled-integration-v1/experiments/p8-coupled-transform-draw0-silu10-v1.json \
  --output /media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-three-layer-v1/receipts/layer-003-loader-closure-v1
```

A PASS closes the previously missing real all-rank loader gate, but the result
still writes `retirement_authorized: false`. Chunk retirement additionally
requires a fresh under-30-GB storage ledger and an explicit receipt naming the
exact chunk hashes being retired. It provides no permission to remove data by
itself.

For layer 20 or 22, change `--layer` and every zero-padded layer component in
the four sidecar paths, postwrite receipt, and fresh output directory together.
The selected layer is written into the protocol, launch receipt, probe result,
and every per-rank result. A sidecar or postwrite metadata layer mismatch is a
hard failure.
