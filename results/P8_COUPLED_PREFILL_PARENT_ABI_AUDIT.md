# P8 coupled-prefill parent ABI audit

Date: 2026-09-05  
Scope: CPU/static only; no image build, CUDA compilation, GPU launch, model service, or performance/KLD measurement.  
Decision: the implementation at `4c73139964ad84ca18b9581a4a1cd1ac47536584` could not be integrated unchanged. Two invocation/launch-contract defects were found. Repair commit `53e0b45c57b9946ae3a7ef5dbdc376400d9a1a96` clears those two structural defects, but it is not device closure.

## Immutable identities

- Candidate implementation commit: `4c73139964ad84ca18b9581a4a1cd1ac47536584` (`Implement exact coupled P8 prefill kernels`).
- Candidate structural report commit: `37e85f5550e422045274bcc01824051a5b9ba190` (`Record coupled P8 prefill structural receipt`).
- Structural report SHA-256: `762b2e4fe780aa27299c6bd51c33c51b4bfd556ec1e0df581fff56be62f66e1a`.
- Immutable parent image: `klc/glm53-p8-tail-repair@sha256:0336113e0fff876cccf9e6ac5347528ae59f4ad894a0ce7cb4c4e90b4651a745`.
- Parent labels identify B12X commit `36bce2c1552ba2d47dc09f20a6f64fbfc8ec4ff8`, integration tree `12c426322cc5d239023b57a4bd5ab0e60c4302e0`, B12X base `30aafad96b7a78064651c4a5ac177791e7bdee30`, and vLLM overlay `df684ff47dcbf088b41311494fd20347a702e56a`.

## Active import origins and hashes

The parent was inspected with `--runtime=runc --network=none -e NVIDIA_VISIBLE_DEVICES=void --entrypoint /bin/bash`. Python resolved B12X from the source checkout, not `build/lib`:

| object | active origin | SHA-256 |
|---|---|---|
| `b12x` | `/opt/infernal-invocation/b12x/b12x/__init__.py` | `1a2c500fffb16f1fa94a458d4cde3602658b8975d0075c369ab97496c87ee6ab` |
| `dynamic.py` | `/opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/dynamic.py` | `7211657f8508dfbf12d2f4158562144fe86741d1f72ea32cc1dc2d6c3dc6b4d6` |
| phase 1 | `/opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/w4a8_phase1.py` | `7222df68bb6ad7ec5ca69cf7c4ffd5c3d6b1ddc240e6375c4b1e77bcc151909c` |
| phase 2 | `/opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/w4a8_phase2.py` | `be317f7f76153ff5f60d2d1cb76f15dae14e6252a109b2d448e7dd8852f66194` |

The source-tree and site-packages phase files are byte-identical. `build/lib` is absent from `sys.path` and contains stale donor variants: phase 1 `cb72c50dab933ee866103caf7c32f1a6cfb15df991fcdd8b753ac765551946b1`, phase 2 `ce31085628a8423468a453275f18c3322f8029d35afccb35f5e7060cbab11a7e`.

Those donor constructors accept neither `trellis_codebook`, `trellis_scaled`, nor `trellis_identity_boundary`; they must not replace the active parent phase files. An integration image should fail closed unless the two active parent hashes above match.

## Constructor ABI: compatible

The actual parent constructors are:

```text
W4A8MaterializedPhase1Kernel(*, fast_math=False, source_tile_m=128,
  deterministic_output=False, num_topk=1, activation='silu',
  trellis_bits=None, trellis_coupled=False, trellis_direct_lut=False,
  trellis_codebook='none', trellis_scaled=False,
  trellis_identity_boundary=False)

W4A8MaterializedPhase2Kernel(*, source_tile_m=128,
  deterministic_output=False, trellis_bits=None,
  trellis_direct_lut=False, trellis_codebook='none',
  trellis_scaled=False, trellis_identity_boundary=False)
```

`inspect.signature(...).bind(...)` accepted the exact keyword sets exercised by candidate `dynamic.py` at lines 1089-1136. The constructor itself is not the blocker.

## Defect 1: FC2 inherited the wrong call ABI

The actual parent phase-2 `__call__` accepts 16 arguments after `self`:

```text
intermediate_u32, down_rp, down_sfb_rp, scatter_output, token_map,
token_weights, task_expert, task_valid_rows, expert_tile_base, down_alpha,
global_scale, trellis_lut, intermediate_tiles, packed_output_tiles,
max_active_clusters, stream
```

Candidate `dynamic.py` lines 2880-2897 pass 17 arguments after `self`, adding `trellis_rotations` as the scale component after `trellis_lut`. Both `P8SmallMPhase2Kernel` and `P8CoupledPrefillFC2Kernel` require that component in their kernel bodies, but commit `4c731399` provided no `__call__` override. The call therefore resolved to the parent and an exact CPU signature bind failed with `TypeError: too many positional arguments` before any device work.

Required repair: `P8SmallMPhase2Kernel` must own an explicit `@cute.jit __call__` with the exact 17-argument dynamic-call ABI and forward `scale_component` between `trellis_lut` and `intermediate_tiles`.

## Defect 2: FC1 inherited the parent's two-CTA launch contract

The actual parent phase-1 `__call__` does accept the candidate's 18 arguments, including `trellis_rotations`, but launches:

```text
grid=(1, 1, max_active_clusters * Int32(2))
min_blocks_per_mp=2
```

`P8CoupledPrefillFC1Kernel` reports `shared_bytes == 65536`, and the candidate report explicitly plans one CTA/SM for this M64 FC1. At `4c731399` it inherited the parent's launch unchanged, so implementation and documented resource contract disagreed. This is a structural launch defect even before checking the device's exact residency limit.

Required repair: `P8CoupledPrefillFC1Kernel` must own its launch boundary and use `grid=(1, 1, max_active_clusters)` plus `min_blocks_per_mp=1`, while preserving the exact 18-argument parent-compatible interface.

## Post-repair CPU overlay result

The repaired candidate modules from commit `53e0b45c57b9946ae3a7ef5dbdc376400d9a1a96` were loaded under their canonical module names on top of the immutable parent in a CPU-only container.

| candidate | `__call__` owner | exact call bind | launch contract | shared bytes | observed file SHA-256 |
|---|---|---:|---|---:|---|
| `P8CoupledPrefillFC1Kernel` | candidate class | 18 args: pass | `grid.z=max_active_clusters`, `min_blocks_per_mp=1` | 65,536 | `c43517ea2bc3d33b66db720a184730970e503c28a4ab930b7696cdbc04617030` |
| `P8CoupledPrefillFC2Kernel` | `P8SmallMPhase2Kernel` | 17 args: pass | `grid.z=max_active_clusters*2`, `min_blocks_per_mp=2` | 34,816 | FC2 `7aed6e7c6f2b4195982e876d932891d4ed5fabd1ee04d8177d032328721d889c`; inherited call source `aefc234df70c4ee8ac823a7067a0691604b805b90ff77e317664212a86c9a81a` |

This clears only the two identified Python/CuTe call and launch-contract blockers. The repaired source is immutable at `53e0b45c`; an image must still pin and verify these file hashes before use.

## Integration image requirements

1. Pin the exact parent digest and assert active source origins and the phase hashes `7222df68...` / `be317f7f...` before copying.
2. Never copy the stale donor phase files `cb72c50d...` / `ce310856...`.
3. Copy candidate modules to both the active source checkout and site-packages, then verify byte equality and canonical import origin.
4. The minimum B12X candidate set is `dynamic.py`, `p8_h128_fc1.py`, `p8_small_m.py`, `p8_coupled_prefill_fc1.py`, `p8_coupled_prefill_fc2.py`, and `p8_coupled_topk.py`. The parent lacks the coupled-prefill and coupled-top-k modules. The wider product image must also include the already-qualified wrapper/hook sources it actually imports.
5. Re-run exact constructor binding, FC1 18-argument binding, FC2 17-argument binding, `__call__` ownership, launch-grid, and residency assertions inside the resulting image.

## Remaining device gates

- CuTe/SM120 compilation of both candidate kernels.
- Resource receipt: registers, actual shared bytes, spill/local memory, accepted `min_blocks_per_mp`, and achieved occupancy.
- Bit-exact kernel/pseudoquant closure, five-run determinism, CUDA-graph parity, and fail-closed runtime selection.
- Matched quality and performance gates. None is established by this audit.

## Reproduction commands

Parent ABI and origins:

```bash
docker run --rm --runtime=runc --network=none \
  -e NVIDIA_VISIBLE_DEVICES=void --entrypoint /bin/bash \
  sha256:0336113e0fff876cccf9e6ac5347528ae59f4ad894a0ce7cb4c4e90b4651a745 \
  -lc 'python - <<"PY"
import hashlib, inspect, sys
from b12x.moe._shared.kernels.w4a8_phase1 import W4A8MaterializedPhase1Kernel
from b12x.moe._shared.kernels.w4a8_phase2 import W4A8MaterializedPhase2Kernel
for cls in (W4A8MaterializedPhase1Kernel, W4A8MaterializedPhase2Kernel):
    path = inspect.getfile(cls)
    print(cls.__name__, path, inspect.signature(cls), inspect.signature(cls.__call__))
    print(hashlib.sha256(open(path, "rb").read()).hexdigest())
print("buildlib_on_path", any("build/lib" in p for p in sys.path))
PY'
```

The post-repair check mounted the candidate kernel directory read-only at `/audit`, loaded `p8_h128_fc1`, `p8_small_m`, `p8_coupled_prefill_fc1`, and `p8_coupled_prefill_fc2` via `importlib.util.spec_from_file_location` under their canonical module names, then used `inspect.signature(...).bind(None, *objects)` for 18 and 17 positional arguments respectively. It also inspected the defining class in each MRO and the `grid` / `min_blocks_per_mp` statements in the resolved `__call__` source.
