# P8 forced-decode capture processor

`p8_decode_capture.processor:ForcedDecodeCaptureLogitsProcessor` implements the
custom `LogitsProcessor` ABI present at vLLM commit
`7f1e92bec13a05170ff78fd102d03c42487e4836`. It captures the first 154,880
unmodified FP32 logit columns before masking every row to the predeclared next
token. The server owns the capture root and exact conditional-fit allowlist;
requests cannot supply paths.

This component is **not directly usable by the current P8 serving image**. That
service selects Model Runner V2, while the pinned vLLM revision explicitly
marks custom logits processors unsupported by V2 in
`vllm/config/vllm.py:2429-2458`. Passing `--logits-processors` under automatic
runner selection changes the endpoint to Model Runner V1; forcing V2 makes
startup fail. Either outcome invalidates a V2/FULL-graph numerical closure.

The processor is therefore a tested reference implementation and a usable V1
component. Qualifying the current P8 endpoint requires porting the same
capture-before-mask and force-after-capture operations into the pinned V2
sampler without changing the model runner. Do not claim closure from this
processor on V1 against the V2 speed endpoint.

Server-owned configuration:

- `GLM53_P8_DECODE_CAPTURE_ROOT`: absolute, pre-existing output directory.
- `GLM53_P8_DECODE_CAPTURE_ALLOWED_WINDOW_IDS`: exact comma-separated
  `conditional-fit-NNNN` identifiers.
- `GLM53_P8_DECODE_CAPTURE_EXPECTED_OUTPUT_TOKENS`: expected forced length;
  `2047` by default and shorter only for a separately labelled canary.

Request `vllm_xargs` must contain exactly the five keys exported as
`REQUEST_KEYS` by `processor.py`. Token hashes use
`SHA256(np.asarray(ids, dtype="<i8").tobytes())`.

A complete request produces `<window>.logits.f32` followed by
`<window>.capture.json`. A final metadata file is the completion marker.
Interrupted or rejected captures retain `.partial`, `.inprogress.json`, and/or
`.failed.json` evidence and are never silently resumed or overwritten.
