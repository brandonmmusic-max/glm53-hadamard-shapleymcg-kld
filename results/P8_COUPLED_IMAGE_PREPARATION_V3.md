# P8 coupled image v3 preparation

Status: **prepared source only; retry image not built; device closure not run**.

Date: 2026-09-05

## V2 failure and causal diagnosis

The CPU-only v2 image build failed closed at verifier step 27. All installed
source hashes, parent phase hashes, import origins, call ABIs, Tail V2, FC2
resources, and Python compilation passed. The sole error was:

```text
candidate prefill_fc1 shared bytes mismatch: expected 65536, got 35328
```

The verifier used `getattr(cls, "shared_bytes")`. That observes an inherited
class attribute before `P8CoupledPrefillFC1Kernel.__init__` runs. The actual
kernel launch constructs `P8CoupledPrefillFC1Kernel()`, whose no-argument
constructor calls `super().__init__(full_coupled=True)` and sets:

| Constructed FC1 property | Exact value |
| --- | ---: |
| `tile_m` | 64 |
| `owned_n` | 128 |
| `stage_bytes` | 17,664 |
| `2 * stage_bytes` | 35,328 |
| `shared_bytes` | 65,536 |
| `shared_words` | 16,384 |
| `trellis_lut_offset` | 65,536 |

This was reproduced without a GPU in the preserved v2 intermediate image.
FC2 constructs to 34,816 bytes as expected.

## Region-bound proof

The 35,328-byte value is the double-buffered native-MMA pipeline extent, not
the full-coupled epilogue allocation. The kernel drains async copies, fences,
and synchronizes before aliasing that pipeline storage with the epilogue.
Within the epilogue the three live regions are contiguous and disjoint:

| FC1 region | Byte interval |
| --- | --- |
| gate FP16 | `[0, 16384)` |
| up FP16 | `[16384, 32768)` |
| transformed FP32 output | `[32768, 65536)` |

Therefore the required allocation remains exactly 65,536 bytes. V3 does not
relax it. The verifier now constructs each candidate using its exact declared
constructor ABI, checks all specialized instance attributes, recomputes the
FC1 intervals, and compares the entire resource contract to the manifest.

## Immutable retry separation

- Retry tag: `klc/glm53-p8-coupled:v3`.
- Source/build/verification schemas: v3.
- The v2 failed plan and log remain preserved at
  `/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-image-v2-build/`.
- A v3 retry must use a distinct fresh output directory and must not overwrite
  either the v2 evidence or an existing image tag.
- Parent, phase hashes and ABIs, repaired-runtime ancestor, candidate source
  hashes, Tail V2, launch policies, and stale-donor rejection remain
  fail-closed.

## Validation and boundary

- Focused builder/verifier tests cover the constructed-instance behavior and
  independently recompute the exact FC1 intervals.
- The v3 verifier and manifest were mounted read-only over the preserved v2
  intermediate image and run with `--runtime=runc`, `--network=none`, and
  `NVIDIA_VISIBLE_DEVICES=void`: **pass**, including constructed FC1
  `shared_words=16384` and every interval above.
- Builder, launch-ABI, and coupled-prefill-plan tests: **30 passed**.
- No runtime kernel source was changed for this correction.
- Image build: **not run**.
- GPU use: **none**.
- CUDA compilation, device resource acceptance, bit-exact closure,
  determinism, CUDA-graph parity, KLD, prefill speed, and decode speed remain
  unqualified.
