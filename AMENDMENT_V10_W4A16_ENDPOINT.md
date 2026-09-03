# V10 pre-attribution endpoint clarification: W4A16 NVFP4

Date: 2026-09-03. This clarification is committed before the full-model H16
conditional-fit run, before selection wave 2, and before any Shapley coalition
score.

The pinned GLM stock checkpoint declares static 4-bit input activations in its
ModelOpt configuration, but the only locally qualified Humming execution path
identifies the loaded routed-expert method as `W4A16_NVFP4`. Its native FP4
activation-input path failed aligned execution locally and remains disabled and
not fully tested upstream. Consequently, every valid V8/V10 stock-versus-H16
NVFP4 KLD row is a packed-W4/BF16-activation endpoint, matching the earlier Qwen
W4A16 rotation experiment. Stock and H16 use the same loader, kernel family,
activation precision, tensor membership, and exact 4.5000076294-bpw routed
weight payload.

This supersedes the W4A4 endpoint label in
`AMENDMENT_V3_6BPW_RUNTIME.md`; that historical amendment remains unchanged.
The executable Shapley ladder is `{ModelOpt NVFP4 W4A16, B12X MXFP6 W6A8}` at
whole-routed-layer granularity. The 5.9583409627-bpw figure is an exact packed
routed-weight payload rate, not a common activation-bit rate. Therefore the
Shapley comparison estimates the combined effect of its selected weight and
native activation formats. The equal-cost uniform-depth control has exactly the
same 35 MXFP6 / 7 NVFP4 format counts, so it isolates allocation value at equal
packed bytes and equal activation-format composition. Uniform rotated NVFP4 is
a lower-rate secondary control, not an activation-matched 6-bpw control.

Runtime gates remain fail closed: NVFP4 logs must show the Humming backend and
`W4A16_NVFP4`; MXFP6 logs must show `source_format=mxfp6_w6a8` and
`act_fmt=e4m3`. No W4A4 claim is authorized by this campaign.
