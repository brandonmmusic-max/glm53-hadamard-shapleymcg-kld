# P8 input prequant diagnostic receipt

Status: structural preparation only. No GPU, image build, service, or runtime
math change was performed by this work.

## Frozen observation

The v4 device closure preserves all 128 input UE8M0 scale bytes but differs in
2 of 4096 input payload bytes. The existing K32 wire permutation maps them as
follows:

| wire byte | logical channel | K32 block | device | CPU reference | CPU normalized value | E4M3 midpoint distance |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1293 | 1287 | 40 | `0x49` (4.5) | `0x4a` (5.0) | 4.75000524520874 | +0.00000524520874 from 4.75 |
| 2000 | 2004 | 62 | `0x5f` (30.0) | `0x60` (32.0) | 31.000001907348633 | +0.000001907348633 from 31.0 |

Both blocks use UE8M0 byte 115, or scale `2^-12`. This establishes that the
remaining observation is localized to tie-edge prequant arithmetic or
normalization. It does not establish compiler reassociation, FMA contraction,
or a faulty reference.

Evidence source:
`/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-fixture-v1/m1-device-closure-v4/carrier-failure.{json,npz}`.

## Diagnostic contract

The opt-in M1 full-coupled specialization records exactly 128 FP32 values
(512 bytes) through the MCG arm's otherwise compile-time-dead `trellis_lut`
operand. The trace is written after the payload quantizer returns, so trace
work cannot feed the observed payload:

1. logical K32 block 40, 32 raw prequant values;
2. block 40 multiplied by the quantizer's `ue8m0_to_output_scale` result;
3. logical K32 block 62, 32 raw prequant values;
4. block 62 multiplied by the same quantizer normalization primitive.

The serving default is `p8_input_prequant_diagnostic = False`; the trace
stores are therefore compile-time dead in the ordinary specialization. The
diagnostic is closure-ineligible and must not weaken or replace the original
byte-exact gate.

## Identity correction

The verified integration commit is
`b1745f43e688ed25e9278488a93e684979086594`. The earlier expanded string ending
in `...b61e8cb` is not a Git object and must not be used by a build recipe.
