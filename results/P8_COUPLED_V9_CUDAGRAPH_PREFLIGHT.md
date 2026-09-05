# P8 v9 coupled CUDA-graph closure preflight

Status: **CPU/static preparation pass; device result absent.**

## Verdict

The existing index-order CUDA-graph probes do not call `P8NativeTPMoE` and
cannot qualify the coupled codec.  The v9 runtime image does not need rebuilding,
however: `scripts/run_p8_coupled_graph_device_closure.py` is a read-only
bind-mounted harness for the already-qualified eager image
`sha256:ad6b26bf6d1f265d99b09383485ddef82a4acfaea43e28af46341ebb41da24e3`.

Frozen graph protocol SHA256:
`3281b800664e7987732a681959c561b0c74905271e34a3da07e1ea004a0a2843`.

## What the device run must establish

- Exact v9 fixture, design, transform, runtime-manifest, eager-result and input
  payload identities are pinned before device work.
- The actual full-coupled wrapper is warmed outside capture, sampled eagerly,
  captured once per M in `{1,2,64,65}`, then replayed five times.
- Every replay must reproduce the CPU-reference E4M3 and UE8M0 carriers exactly.
- Every observable carrier, routed output and final output hash must be byte
  identical to the fresh eager observation and across all five replays.
- Capture failure or inaccessible/stale debug carriers is a failure, not a
  fallback pass.

The probe records component closure only.  Passing does not qualify KLD,
throughput, serving, TP4/all-rank behavior, real encoded weights, or full-model
CUDA-graph integration.

## Static validation

`PYTHONPATH="$PWD/src:$PWD" python3 -m pytest -q tests/test_p8_coupled_graph_device_closure.py`

Result at preparation: `10 passed`.

No GPU, service, model read, image build, or large allocation was performed by
this preflight.

## Opt-in device command

Use a fresh output directory and an idle SM120 GPU:

```bash
python3 scripts/run_p8_coupled_graph_device_closure.py --execute \
  --image sha256:ad6b26bf6d1f265d99b09383485ddef82a4acfaea43e28af46341ebb41da24e3 \
  --gpu-device 0 \
  --sidecar /media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-fixture-v1/rank0-v1/p8-synthetic-layer-003-tp4-rank-0.safetensors \
  --design /media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-fixture-v1/rank0-v1/synthetic-design.json \
  --transform /home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-coupled-integration-v1/experiments/p8-coupled-transform-draw0-silu10-v1.json \
  --output /media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-fixture-v1/cudagraph-device-closure-v9
```

The outer launcher verifies image and input identities, checks GPU idleness and
temperature, disables networking, mounts inputs read-only, hashes the exact
bind-mounted harness, and preserves launch/execution/result receipts.
