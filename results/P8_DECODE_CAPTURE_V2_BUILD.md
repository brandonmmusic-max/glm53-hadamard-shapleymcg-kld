# P8 Model Runner V2 numerical-capture image

Status: **built and source-verified; GPU capture not yet tested**, 2026-09-05.
This is correctness infrastructure, not another speed or KLD result.

The separate image is
`sha256:071da1f9e9b24709e59a0959b97a8d524e82fca2fbf4de9a5dc2113a938c6b60`,
based directly on the measured N64-capable image
`sha256:6c08dffb4184c2704173a12909f4bbfaaa866351e55cbf03a2182741baf81141`.
The serving benchmark image and frozen cold-comparison sources are unchanged.

The hook captures pre-mask model logits inside the V2 sampler, after the model
forward graph, and forces the next original token. It does not install a V1
custom logits processor. One prompt token followed by 2,047 forced outputs
provides all 2,047 causal rows, including one prefill row and 2,046 decode rows.
The full conditional-fit panel will cover 32 windows and 65,504 prediction rows;
it has not yet been replayed with this image. Stored FP32 may be an exact
upcast of the model head's lower-precision output; metadata records its dtype.

## Structural evidence and review

- Both original sampler copies matched pinned vLLM `7f1e92bec13a05170ff78fd102d03c42487e4836`
  source SHA `0b56a1c80e2823235fc7df202c6784fd11b83ecaf80af745dceef5a143307711`
  before patching. Both patched copies now match
  `717bdd2203c8977205e16ca63385027f7e7ec8acd54afd962ada3e611cf7b958`.
- Image package hashes match the reviewed host sources. The executed inherited
  FC2 remains `a0c398e9d412672d1138c69a5c0c3677bb29b7215379c701062d29b8b9b9265f`;
  the different local donor was not copied into this image.
- Independent source review checked causal offsets, V2 startup dummy handling,
  request mappings, nonpadding M1 batches and forced-token override hazards.
  CPU tests cover these guards, partial artifacts and exact response checking.
- Independent builder review added a direct immutable parent reference and
  explicitly GPU-hidden `runc` verification. Fourteen mocked builder tests
  passed; 88 capture/protocol/archive/cold-runner tests also passed. These tests
  establish software safeguards, not real-device numerical closure.

The image was built without network access or GPU work. The post-build check
ran only `sha256sum`, with `--runtime=runc` and `NVIDIA_VISIBLE_DEVICES=void`.
No model service was launched or restarted by this build.

## Replay and receipts

```bash
python3 scripts/build_p8_decode_capture_image.py \
  --output /absolute/fresh/build-receipts \
  --tag klc/glm53-p8-capture:unique-version
```

The authenticated parent image must already be local; existing output paths
and capture tags are rejected. The builder writes its plan before execution,
preserves failures, and checks source identity again after building.

Portable plan, build receipt and image-source hashes are under
`evidence/opened/codec-v2/p8-decode-capture-image-v1/`.
Build receipt SHA-256 is
`fe2f1e455030e0fb8173fb9bb82d9f4389368f3d3846a08350aeadcfa6c0b786`.
The original build log remains under
`/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-smallm-scheduler-v1/decode-capture-image-v1/`;
its hash is recorded in that receipt.

Next, after the cold comparison releases the GPU lock, a separately sealed
same-image N128/N64 canary must pass before full-panel replay. Capture D2H
synchronization invalidates timing for this endpoint. Allocation remains
stopped pending the independent speed and numerical-closure gates. P8 uses
native E4M3 `mxf8f6f4`, with twice NVFP4's MMA issue count; this does not qualify
the separate P4/NVFP4 speed-class product. No protected roles were opened.
