# Model Runner V2 capture seam

The current P8 service reports `Using V2 Model Runner`. At pinned vLLM commit
`7f1e92bec13a05170ff78fd102d03c42487e4836`, configuring a custom logits
processor either selects V1 or makes forced-V2 startup fail. The V1 processor
in this package must therefore not be used for the P8 closure endpoint.

`sampler-v2-hook.patch` adds exactly three calls to the pinned V2 sampler:

1. construct `V2ForcedDecodeCapture` alongside the existing sampler state;
2. register every actual request before ordinary sampler registration; and
3. capture then force raw logits at the beginning of `Sampler.__call__`.

The pinned unmodified sampler source SHA-256 is
`0b56a1c80e2823235fc7df202c6784fd11b83ecaf80af745dceef5a143307711`.
The dedicated image build must verify that hash before applying the patch and
must preserve the patched source and hashes as receipts.

Runtime requirements:

- Add this `runtime_patch` directory to `PYTHONPATH`.
- Apply the patch to `vllm/v1/worker/gpu/sample/sampler.py` in the dedicated
  image. Do not pass `--logits-processors`.
- Set `VLLM_USE_V2_MODEL_RUNNER=1` and require the runtime receipt `Using V2
  Model Runner`.
- Set `GLM53_P8_DECODE_CAPTURE_V2=1` plus the three server-owned capture
  variables documented in `README.md`.
- Full-window closure requires `capture_start_output_len=0`. A short canary may
  use a nonzero start only when the dedicated endpoint explicitly sets
  `GLM53_P8_DECODE_CAPTURE_ALLOW_NONZERO_START=1`; that canary cannot qualify
  numerical closure.
- Keep Model Runner V2, the serving FULL CUDA graphs, TP4/DCP1, checkpoint,
  sidecars, and all other launch arguments identical between N128 and N64.

The seam runs after model logits are produced and outside the model CUDA graph.
It converts the untouched first 154,880 columns to stored little-endian FP32
for the binary capture, records the incoming tensor dtype and original width,
then masks the original tensor to the forced token. Stored FP32 does not imply
that the model head originally emitted FP32. It is a correctness-only endpoint:
D2H capture synchronization invalidates its timing for speed claims.

Startup profiling uses a synthetic `InputBatch.make_dummy` without registering
a real request. Passthrough is allowed only when request state is empty, every
request ID has the exact dummy prefix pattern, and every scheduled token is
marked padding. Any actual unregistered request fails closed.

For real requests, the hook additionally requires a one-row, non-padding,
non-speculative batch with `max_query_len=1`, consistent request-ID/index maps,
and no structured-output or thinking-budget state. These checks prevent later
sampler features or scheduler remapping from changing the forced sequence after
the raw-logit capture.
