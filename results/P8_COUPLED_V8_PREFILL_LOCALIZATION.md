# V8 prefill failure localized to rank-2 input indexing

Synthetic layer3/rank0 TP4-local M2, native coupled P8 MoE, E4M3 activations,
4.25 bpw plus channel metadata (~4.25398 effective). Attention/KV: N/A.
P8 mxf8f6f4 uses twice NVFP4's MMA issue count. No KLD or speed claim.

Image `sha256:b769436398e42cbf48d1df8f07a22ffbd069d63f742bc16780c126267f8ddf5c`
cleared the prior grouped FC2 compile failure. First numerical attempt failed.
The separately declared diagnostic retry retained exact byte gates and saved
the first failing carriers. M2 input payload: 8095/8192 mismatches; input scales:
44/256. Intermediate payload: 8113/8192; scales: 134/256. Shapes match and all
eight experts have two valid routes. This is not the earlier carrier-shape bug.

The prefill input loader used `a_input[row_base + col]` on a rank-2 CuTe
tensor. Scalar logical coordinates are colexicographic, not physical row-major
offsets. CPU-only reproduction using `x.T.contiguous().reshape(2,4096)` followed
by the unchanged H512, suh, H128 and FP8 quantizer exactly reproduces all 8192
observed input bytes and all 256 scales (zero mismatches). This pins the input
defect, not yet downstream correctness. M1 has a singleton token dimension
and therefore does not expose this indexing mistake.

Source repair: explicit `a_input[token_idx, col + offset]` for all four loads
in the grouped input helper. New image/device validation required. No gates
relaxed, no teacher data opened, production remains off.

Raw local evidence is under
`/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-fixture-v1/`
in `prefill-device-closure-v8/` and `prefill-device-closure-v8-localization/`.
The latter retains `first-carrier-failure.npz` and its count summary.
