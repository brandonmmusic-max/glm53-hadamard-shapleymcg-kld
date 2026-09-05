# P8 coupled-scale encoder/sidecar implementation

Date: 2026-09-05
Scope: CPU-only implementation and reference validation. No GPU, encoder run,
kernel build, model service, or protected-role evaluation was performed.

## Outcome

This branch adds a separate fail-closed P8 coupled format. It preserves the
existing K4 procedural-MCG trellis plus UE8M0/32 weight payload at exactly
4.25 bpw, and adds TP-local FP16 coupled-scale metadata plus one draw byte per
expert. For the frozen Flash geometry (`H=4096`, `I=2048`, `E=288`, TP4), the
metadata rate is `0.003979859528718171` bpw and total storage is
`4.253979859528719` bpw.

The CPU reference implements the archived Luke/QSRT order:

```text
BF16 -> FP16 -> H512 -> FP16(suh) -> H128 -> optional E4M3/UE8M0-K32
FC1 physical pair -> 32-row atom interleave -> H128 -> gate/up svh
  -> H128 -> frozen signs -> clipped SwiGLU
  -> frozen signs -> H128 -> FP16(down suh) -> H128
  -> optional E4M3/UE8M0-K32
FC2 -> FP16 -> H128 -> down svh -> route sum -> H512
```

The new encoder transforms gate, up, and down weights into those coordinates,
builds the FC1 Hessian from the transformed A8 input carrier, builds the FC2
Hessian causally from the reconstructed gate/up candidate and transformed A8
middle carrier, and calls the repository's existing inter-group GPTQ-style
trellis feedback with static in-group activation order. LDLQ is explicitly
absent.

The chunk and TP4 sidecar ABIs explicitly freeze the otherwise easy-to-confuse
orders:

- FC1 trellis slots: `gate-up`
- FC1 UE8M0 scale planes: `up-gate` (the existing B12X ABI)
- coupled private scales: `gate_svh-up_svh-down_suh`

Preparation and encoding are restricted to layers 3, 20, and 22. The
procedural intermediate sign draw is frozen to the QSRT default draw zero for
all experts and layers; this implementation introduces no draw search.

## Validation

CPU test command:

```bash
python3 -m pytest -q \
  tests/test_p8_coupled_scale.py \
  tests/test_build_p8_coupled_scale_tp4_sidecars.py
```

Result: `10 passed in 0.79s`.

The seeded synthetic algebraic closure row measured max absolute error
`0.00037360191345214844` and output NMSE `3.0293630067567e-07`; this includes
the explicit BF16/FP16 storage casts and disables activation quantization.

The tests cover unquantized coupled/source expert closure with signed scales,
the exact BF16-to-FP16/H512/scale/H128 cast order, exact EXL3 scale loading and
hashing, rejection of non-shared vectors and wrong dimensions, frozen pilot
layers/draws, TP4 sharding, unchanged 4.25-bpw weight payload, the exact
0.0039798595-bpw production metadata rate, and rejection of changed shared
metadata. `py_compile` and Ruff also passed on every added file.

## Current source blocker (preserved, not papered over)

The locally available exact GLM-5.3 EXL3 scale files for layers 3, 20, and 22
belong to the full model (`H=6144`, `E=256`, `I=2048`). The P8 product under
test is GLM-5.3-Flash (`H=4096`, `E=288`, `I=2048`). A CPU read-only audit of
all three files returned the same fail-closed result:

```text
3  RuntimeError EXL3 scale source has 256 experts, expected 288
20 RuntimeError EXL3 scale source has 256 experts, expected 288
22 RuntimeError EXL3 scale source has 256 experts, expected 288
```

The implementation will not resize, repeat, truncate, or relabel those tensors.
Therefore this branch is ready to prepare the three-layer pilot once exact
Flash-compatible scale metadata is supplied or generated under a separately
frozen recipe; it has not produced a Flash checkpoint and makes no KLD claim.

## Files

- `glm53_nvfp4/p8_coupled_scale.py`: scale loader, exact transforms, and CPU
  source/coupled references.
- `glm53_nvfp4/prepare_p8_coupled_scale_v1.py`: immutable three-layer design
  generator and geometry gate.
- `glm53_nvfp4/quantize_p8_coupled_scale_layer.py`: transformed-carrier P8 K4
  layer encoder.
- `glm53_nvfp4/build_p8_coupled_scale_tp4_sidecars.py`: independent TP4
  sidecar writer; it does not weaken the identity-P8 loader.
- `tests/test_p8_coupled_scale.py` and
  `tests/test_build_p8_coupled_scale_tp4_sidecars.py`: CPU closure and ABI
  tests.

The mathematical/runtime hook derivation and source tensor receipts remain in
`results/P8_N64_COUPLED_SCALE_HOOK_AUDIT.md`.
