# P8 coupled image v2 preparation

Status: **superseded after a CPU-only image build failed closed; no image was
produced and device closure was not run**.

The later v2 build reached verifier step 27 and failed because the verifier
read FC1's inherited class-level `shared_bytes` value (35,328) instead of the
constructed M64/full-coupled instance value (65,536). The complete failed log
and plan are preserved under
`/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-image-v2-build/`.
The v3 receipt documents the bounded correction.

Date: 2026-09-05

Scope: CPU/static preparation only. No Docker build, CUDA compilation, GPU use,
model-service change, throughput run, or KLD run occurred.

## Decision and lineage

- The v1 preparation receipt remains at
  `results/P8_COUPLED_IMAGE_PREPARATION.md`; its dedicated image was never
  built.
- The repaired decode-plus-prefill recipe uses the new dedicated tag
  `klc/glm53-p8-coupled:v2` and refuses to overwrite an existing tag.
- Immutable parent image:
  `sha256:0336113e0fff876cccf9e6ac5347528ae59f4ad894a0ce7cb4c4e90b4651a745`.
- Required repaired-runtime ancestor:
  `53e0b45c57b9946ae3a7ef5dbdc376400d9a1a96`.
- Tail-V2 SHA-256:
  `494192195da43c46d99a684555fc10fd13a19e89288cb9f51da2536ccdf1f251`.
- The image remains `research-only-not-device-qualified`. Its device-support
  label is `decode-m1-prefill-m64-n128-unqualified`; this is a source/runtime
  coverage label, not a correctness or performance claim.

## Parent phase fail-closed contract

The Dockerfile checks the active parent phase files in both the editable B12X
tree and site-packages before installing candidate files. The in-image
verifier repeats the hash checks and asserts the exact Python call ABIs.

| Active parent phase | SHA-256 | Arguments after `self` |
| --- | --- | ---: |
| `w4a8_phase1.py` | `7222df68bb6ad7ec5ca69cf7c4ffd5c3d6b1ddc240e6375c4b1e77bcc151909c` | 18 |
| `w4a8_phase2.py` | `be317f7f76153ff5f60d2d1cb76f15dae14e6252a109b2d448e7dd8852f66194` | 16 |

The manifest explicitly rejects the stale `build/lib` donors with hashes
`cb72c50d...` and `ce310856...`; the verifier fails if `build/lib` is on
`sys.path`. No donor phase file is copied by the recipe.

## Repaired candidate contract

The v2 source set adds the exact coupled-prefill planner and both grouped
M64/N128 kernels. Candidate files are copied to both active B12X locations,
and the planner is installed under `/opt/p8-coupled-runtime`.

| Candidate | Call ABI | Owner and launch contract | Shared bytes |
| --- | ---: | --- | ---: |
| coupled FC1 | 18 args | candidate-owned; `grid.z=max_active_clusters`, `min_blocks_per_mp=1` | 65,536 |
| coupled FC2 | 17 args including `scale_component` | inherited repaired P8 call; `grid.z=max_active_clusters*2`, `min_blocks_per_mp=2` | 34,816 |

The in-image verifier checks exact source hashes, canonical import origins,
parent and candidate call signatures, `__call__` ownership, shared-memory
declarations, normalized launch-source clauses, Tail V2, and Python bytecode
compilation. These checks import source definitions but contain no GPU
initialization, kernel compilation, launch, or device query.

## Bounded footprint and validation

- Manifest source set: 14 files, 766,060 bytes total before Docker metadata.
- The builder still stages only declared sources plus the manifest; no model,
  calibration role, logits, capture, or unrelated repository data enters the
  context.
- `python3 -m py_compile scripts/build_p8_coupled_image.py runtime_patch/p8_coupled_image/verify_image.py tests/test_build_p8_coupled_image.py`: pass.
- `python3 -m pytest -q tests/test_build_p8_coupled_image.py`: **7 passed**.
- JSON parse, source hashes, and `git diff --check`: pass.
- Image build: **not run**.
- GPU use: **none**.

## Remaining gates

The v2 recipe is only a reproducible image input. It does not establish CuTe
SM120 compilation, resource acceptance, bit-exact device closure, five-run
determinism, CUDA-graph parity, KLD, prefill speed, or decode speed. Those
remain separate gates after the current campaign permits device work.
