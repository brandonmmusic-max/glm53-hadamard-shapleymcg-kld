# GLM-5.3 B12X FP8 NoPE audit and rebase kit

Status: CPU ABI closed; device/runtime integration **not qualified**.

This directory records the exact implementation lineage needed to add a
same-backend `fp8_ds_mla` control for GLM-5.3-Flash. It does not enable a GPU
path by itself and it must not be copied blindly into the active image.

## Finding

GLM-5.3-Flash uses an absorbed 512-wide MLA latent and
`qk_rope_head_dim == 0`. Its correct FP8 cache row is therefore exactly:

| byte range | field |
| --- | --- |
| `[0, 512)` | 512 E4M3FN values, four consecutive groups of 128 |
| `[512, 528)` | four little-endian FP32 scales, `amax(group) / 448` |

An all-zero group stores scale `1.0`. There is no RoPE field. The 656-byte
record currently returned by the active vLLM B12X backend is the GLM-NSA
geometry: it adds 128 BF16 RoPE bytes after the same 528-byte prefix. Padding a
NoPE model into that record does not establish a reader/writer ABI and is not a
correct fix.

The executable CPU contract is
`glm53_nvfp4/fp8_nope_abi.py`; `tests/test_fp8_nope_abi.py` pins the record
bytes, E4M3 encodings, FP32 scale bytes, zero-group rule, fail-closed model
selection, and 64-bit padded-page address arithmetic.

## Active-source audit (2026-09-05)

The inspected production-line sources were:

- B12X `30aafad96b7a78064651c4a5ac177791e7bdee30`
- vLLM `df684ff47dcbf088b41311494fd20347a702e56a`
- active vLLM B12X backend file SHA-256
  `0499c674b6890266b50fa0d5724dcfbb83cba3917714a6787e5dddc6feb65572`
- base runtime image
  `sha256:ed027a3a2ff93b9cf60c95f7adfaf676cabc8e040a28cffa7486a262c82fdfbe`
- P8 image
  `sha256:0f1eae9329965d68713857e4a5a12e9c5440c866b532e7ba288dc2ae4067fad9`

The current host and device contracts are internally fail-closed, but only for
the existing NoPE NVFP4 path:

1. `supports_combination` rejects head size 512 unless the cache dtype is
   `nvfp4_ds_mla` and the image-level NoPE opt-in is present.
2. `get_kv_cache_shape(..., fp8_ds_mla)` unconditionally allocates 656 bytes.
3. `ModelType.GLM_NOPE` in B12X traits accepts only `NVFP4_E4M3`, selects
   `ComputeMode.BF16`, and uses a 288/304-byte record.
4. The active FP8 writer is the stock GLM-NSA 512+scales+RoPE writer. There is
   no host dispatch to the 528-byte NoPE writer.
5. The reader has useful NoPE elision work for NVFP4, but its ARBITRARY_FP32
   branch still assumes the GLM-NSA 656-byte source stride. Consequently,
   changing only allocator shape or only `supports_combination` would be an
   unsafe half-port.

This explains the observed startup rejection. It does **not** prove that the
prior NVFP4-versus-FP8 KLD gap is caused by B12X: the successful four-window
control changed both backend and cache format to FlashInfer. A same-B12X
528-byte control is still required to isolate those variables.

## Pinned implementations

The four patches here are immutable reference commits, not a directly
applicable linear series. Their exact parents and hashes are in
`reference_manifest.json`.

- `reference-b12x-fp8-nope-f7c7fd9.patch` is the complete CuTeDSL
  `ModelType.GLM_NEXT` 528-byte implementation, including writer, reader,
  decode/prefill math, scratch planning, API validation, and device tests.
- `reference-vllm-fp8-nope-0c87882.patch` is its vLLM integration: model
  identity, 528-byte cache allocation, packaged writer dispatch, zero-width
  `k_pe`, and explicit model type on decode/extend.
- `reference-b12x-dual-format-c845ea1.patch` is the later reconciled B12X
  commit that preserves FP8 NoPE while adding the 288/304-byte NVFP4 path.
- `reference-vllm-dual-format-fc2904b.patch` is the matching vLLM dual-format
  integration.

Attribution: these patches are verbatim repository history from the local B12X
and vLLM integration worktrees. They are included to preserve authorship and
implementation provenance during the rebase; they are not a clean-room rewrite.

## Required rebase, in order

1. Rebase the *dual-format* B12X behavior onto `30aafad`, preserving the
   active NoPE NVFP4 per-token-scale work. Extend `GLM_NOPE` traits so
   ARBITRARY_FP32 selects FP8 compute and exactly 528 bytes; keep NVFP4 on its
   existing BF16 288/304 recipe.
2. Thread `traits.kv_gmem_stride` through every NoPE gather and page-stride
   calculation. In ARBITRARY_FP32 mode, copy 528 bytes and issue no RoPE copy.
3. Port the 128-thread writer from `f7c7fd9`: one 128-value group per warp,
   FP32 amax reduction, E4M3 RNE satfinite conversion, four FP32 scales, Int64
   page and row address arithmetic.
4. Rebase vLLM host integration onto `df684ff`, selecting GLM NoPE from model
   config (`kv_lora_rank=512`, `qk_rope_head_dim=0`) rather than width alone.
   Allocate 528 only for that identity and dispatch the packaged writer.
5. Preserve every active DCP/index-order change. The backend file is dirty
   relative to `df684ff`; resolve semantically, never by overwriting it with a
   reference snapshot.

## Qualification still required

No GPU service was launched for this audit. Before the environment opt-in can
be enabled, the rebased image still needs:

- B12X device test: writer output byte-equal to the CPU reference, including
  padded pages, negative graph slots, and an address beyond 2 GiB.
- Reader closure: B12X decode/prefill output against a dequantized torch
  reference for 528-byte rows, plus CUDA-graph replay and five-run bitwise
  determinism.
- vLLM allocation/writer test proving every NoPE cache page has semantic width
  528 and the physical page stride reaches B12X unchanged.
- The predeclared four-window same-backend KLD control. Only this result can
  separate the NVFP4 cache recipe from a backend/serving regression.

Do not add RoPE to repair this model: `qk_rope_head_dim == 0` is the model
contract. The candidate causal variables are the cache format/scales and the
reader/writer path, not omitted rotary position data.

