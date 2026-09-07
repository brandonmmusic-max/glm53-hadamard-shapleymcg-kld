# Current TrellisMX P8 serving recipe

Four SM120 RTX PRO 6000 96GB GPUs, TP4/DCP1, MTP3, native E4M3 P8 MoE,
B12X attention and PCIe collectives, calibrated NVFP4 MLA KV.
Default maximum context: **1,000,000 tokens**. This is within the base
model's declared 1,048,576 positions; retrieval quality at 1M is **Not tested**.
The release recipe itself has CPU syntax/Compose validation only until a
separate startup receipt is published. Existing quality and speed results
cover their recorded launches, not this larger context limit.

Published runtime: `verdictai/trellismx:glm53-flash-p8-rc5-20260907@sha256:609a5fc1cd7d994ba32d9c03626c414d315947eb9f13fab474a15bc8dfbe0129`.
Compose defaults to that immutable digest. Use the full repository checkout.
Set `MODEL_ROOT` to the required stock carrier,
and `P8_CHECKPOINT_ROOT` to the upgrades-only 17-K5/25-K4 encoded checkpoint
(containing `sidecars/`). Stock weights alone are not TrellisMX weights.
The runtime image does not contain or download the model. Existing encoded
expert files replace the corresponding carrier tensors at load time.

```bash
docker compose -f deploy/trellismx/compose.yaml config --quiet
docker compose -f deploy/trellismx/compose.yaml up -d
```

Default port is 8033; no host production services are changed. The container
binds all host interfaces: use a trusted network/firewall or authenticated
gateway before remote exposure. No API key is baked in. Prefix caching is on
for normal serving. Do not use cache-hit prefill to claim decode speed.
No custom GPU clock, power, or thermal override is performed.

On the campaign host, obtain `/run/lock/klc/model-stack.lock` and verify
production remains off and port 8000 unbound before launching. Do not run
alongside another GPU owner. Use a dedicated cache volume for this version;
cold compilation latency is not a throughput score.

The Dockerfile under `image-context/` reconstructs the exact 10 checked
runtime source files over the separately published pinned v11 base. The
published RC5 image itself is the existing tested image, retagged unchanged,
not a newly rebuilt image with implied GPU qualification. Cold source/JIT
rebuild performance is Not tested. Model weights are not in either image.

The image is a custom experimental derivative, not Local Inference Lab's
recommended runtime. Results and limitations:
[release report](../../results/RC5_RELEASE_RESULTS_20260907.md).
License: repository ShapleyMCG plus the respective upstream component
licenses. The general toolkit adapter contract does not make this GLM-specific
serving recipe compatible with arbitrary architectures.
