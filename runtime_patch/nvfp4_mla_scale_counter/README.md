# NVFP4 MLA static-scale counter

Status: CPU/static prepared; no image build, GPU run, service change, or timing
measurement has been performed.

This opt-in diagnostic observes the exact 512-wide tensor presented to the
active GLM-5.3 NoPE NVFP4 cache writer. That point is after the calibrated
per-layer outer-scale division and after any DCP input gathering. The existing
writer and its 288-byte record remain unchanged:

```text
[0,256)   512 packed E2M1 values
[256,288) 32 E4M3 group scales, one per 16 values
```

For each layer and process rank, it records valid writer calls/tokens/groups,
the largest writer-domain absolute value, the largest `amax / 2688` ratio,
the largest requested E4M3 group scale (`amax / 6`), groups and tokens over
the E4M3 range, input zeros, all-zero groups/tokens, scales expected to round
to zero, and non-finite values/groups/tokens. Invalid CUDA-graph slots are
excluded with the writer's own slot-capacity rule.

Enable only in the diagnostic image or an equivalent runtime mount:

```bash
VLLM_NVFP4_MLA_SCALE_COUNTER=1
VLLM_NVFP4_MLA_COUNTER_CONTROL=/counter/control.json
VLLM_NVFP4_MLA_COUNTER_OUTPUT_DIR=/counter/receipts
```

The control file is atomic and sequence numbered. After ordinary server
warmup, issue `reset` immediately before the first measured request. Between
requests, issue `dump_reset`: the first writer invocation of the next request
dumps the completed request, resets all counters, and then counts the current
invocation for the newly named request. After the last request, issue `dump`
and send one disposable flush request; the dump occurs before that request is
observed and disarms the counter.

FULL CUDA graphs are supported explicitly: graph capture registers persistent
per-layer counter tensors behind a zero-valued device gate, so warmup does not
count. A hook at the outer `GPUModelRunner.execute_model` boundary processes
control commands before graph replay, while captured read-only reductions
update the armed tensors inside each replay. The control file is never polled
from inside a captured graph.

Example host commands:

```bash
python3 -m glm53_nvfp4.nvfp4_mla_counter_control issue \
  --control /counter/control.json --sequence 1 --action reset \
  --request-id conditional-fit-0032

python3 -m glm53_nvfp4.nvfp4_mla_counter_control issue \
  --control /counter/control.json --sequence 2 --action dump_reset \
  --dump-request-id conditional-fit-0032 \
  --request-id conditional-fit-0021

python3 -m glm53_nvfp4.nvfp4_mla_counter_control wait \
  --output-dir /counter/receipts --sequence 2 --ranks 0,1,2,3
```

The run must first pass the predeclared exact-logit canary in
`experiments/nvfp4-mla-scale-counter-cf32-v1.json`. Instrumentation adds CUDA
reductions and dump synchronization. Never use an instrumented process for a
prefill, decode, latency, or throughput claim.
