# P8 full-coupled M1 middle-carrier divergence audit

Date: 2026-09-05

Scope: CPU/source audit and repair only. No GPU, image build, service, timer, or
production action was performed. Production remains stopped by the user's
maintenance directive.

## Immutable failure input

- `carrier-failure.npz`: `6c60560e4ca9eee1c86ab0be8d680d9b86abc91dd77fa8b895d21994db35a25a`
- `carrier-failure.json`: `9cbdb80998a54601e1320f7944243be1e124670f9f3e8ef6df380f5602299d27`
- Executed image: `sha256:77ae1c0b46c7f72b085b6ae3f8184f0f9df229bce5306d1849fb770501dc0d64`
- Image source manifest: `ff5ce6158e2d2134cedd1a451bc911fc3623c3ba93513916556a61833ee496d2`
- Protocol: `9556885c32385574530acf7b07c1d5636b46852ca9714c27883f98f17368efe1`

The observed gate was 3/4096 input payload bytes, 0/128 input scale bytes,
4057/4096 middle payload bytes, and 59/128 middle scale bytes different from
the CPU reference. The routed output cosine was about 0.032566 and relative
L2 about 1.43039.

## Localization

The three input bytes are not a sufficient explanation. Reconstructing the
device input carrier changes only 3/4096 values versus the reference carrier,
with maximum absolute delta 0.0009765625 and relative L2
0.0005159181891940534. Feeding that reconstructed carrier through the CPU
reference changed only 199/4096 predicted middle payload bytes and 0/128
middle scale bytes; the mismatch against the device remained 4057 payload and
59 scale bytes.

The physical row extraction is not the cause. In the M1 direct-route schedule,
`token_map[slot * 16]` is zero for all eight routes and each route's physical
row is `slot * 16`. `row_counts` and `expert_tile_base` are intentionally unused
and remain zero. Cross-expert row pairing, simple row orders, and adding or
removing the K32 payload permutation did not produce a close match; the best
cross-expert payload pairing still differed in 500/512 bytes.

The W13 UE8M0 repack/stage/consume geometry is exact. A CPU-only run inside the
immutable image repacked the fixture's logical `[E,1024,128]` scale plane with
`_e8m0_scale_to_w4a8_sfb_inplace`, then enumerated the specialized kernel's
`output_tile`, packed half, K128, row/N8, q, and K32-byte addresses for all eight
selected experts. Every consumed byte equaled its logical up/gate source byte.

## Confirmed defects and repair

Two independent source/index emulations found defects in the full-coupled FC1
branch of `p8_h128_fc1.py`.

1. The private `gate_svh`/`up_svh` lookup omitted `output_tile * 128` from
   `base_col`. Tile 0 was addressed correctly, while every one of the 256
   gate/up private-scale coordinates in each of tiles 1, 2, and 3 reused tile
   0's coordinate range. The repair adds the tile offset. Enumeration now
   covers gate coordinates 0..511 and up coordinates 0..511 exactly once.
2. The contiguous activation gather selected an `activated[source_slot]`
   variable before `shuffle_sync`. CUDA shuffle reads the selected variable as
   evaluated by the remote source lane, not the destination lane. The legacy
   mapping was exactly `[0:32], [96:128], [0:32], [96:128]`: 64/128 semantic
   positions wrong, with the middle 64 dropped and the outer quarters
   duplicated. The repair shuffles each fixed register slot uniformly from two
   adjacent producer lanes, then selects the first or second segment locally.
   CPU lane emulation now maps exactly to 0..127 with no duplicates or omissions.

Repaired source SHA256:
`499165e495b92bf03f70fcd631e4c45cf030585ad543c425395af57355db7dfa`.

## Validation

- Focused coupled M1/prefill/runtime suite: 54 passed.
- Full CPU suite with `PYTHONPATH=.`: 1326 passed, 2 failed. Both failures are
  stale source pins outside this repair: the v3 image manifest still pins the
  pre-repair `dynamic.py`, and `test_p4_native_kernel.py` pins an older P8
  wrapper. These are not numerical or mapping failures from this change.
- `git diff --check`: clean.

## Claim boundary and next device gate

This establishes two causal source defects and CPU-correct mappings. It does
not establish device closure. The next authorized GPU action should be one
immutable v4 image build followed by the same preregistered M1 carrier gate.
The image manifest must pin this repaired FC1 source. If the middle carrier
still fails broadly, the next discriminating diagnostic is a bounded raw FC1
gate/up FP16 capture before the first joint H128, compared per selected expert
against the existing CPU matmul. That separates procedural decode/MMA/weight
scales from the joint transform without weakening the existing carrier gate.
