# TrellisMX P8 RC5 GLM-5.3-Flash

Existing verified RC5 runtime republished unchanged; new portable 1M-context launch is not GPU-qualified.

## Identity and status

- Release class: `experimental`
- Distribution role: `custom`
- Qualification: `implemented`
- Maintenance: `ephemeral`
- Model family: GLM-5.3-Flash
- Image and digest: `verdictai/trellismx:glm53-flash-p8-rc5-20260907@sha256:609a5fc1cd7d994ba32d9c03626c414d315947eb9f13fab474a15bc8dfbe0129`
- Recommended image/control: `voipmonitor/vllm@sha256:a298fe1cd207eaf97bd2ff2686716ed25b7009c09b36650eba732a4a7dc51512`

## Community runbook

- Wiki: https://github.com/local-inference-lab/rtx6kpro @ `94b71ac2a5f9c75f6b60dd1b6e6dffda492a4942`
- Runbook: [models/glm-5.3-flash.md](https://github.com/local-inference-lab/rtx6kpro/blob/94b71ac2a5f9c75f6b60dd1b6e6dffda492a4942/models/glm-5.3-flash.md)
- Relationship: Community source contract; does not qualify this custom P8 runtime.

## Based on

- Base image and digest: `verdictai/trellismx:glm53-flash-p8-base-v11-20260907@sha256:9846694e7dd3546f6ef3d259ecd2c129a5d54147662a76d32c5a3d15673f5e25`
- Base credit: Existing P8 v11 derivative by Brandon Music; inherits Local Inference Lab vLLM/B12X, ExLlamaV3, KQuant and QSRT. Component licenses and repository CITATION.cff apply.

## Build recipe

- Public recipe: https://github.com/brandonmmusic-max/glm53-hadamard-shapleymcg-kld/tree/a3e8f95847ffa5a7708f9b654906994a80393810/deploy/trellismx/image-context
- Recipe commit: `a3e8f95847ffa5a7708f9b654906994a80393810`
- Complete build command:
```bash
docker build --pull=false -t trellismx-rc5-rebuild deploy/trellismx/image-context
```

## Source commits, PRs, patches, and overlays

- **vllm:** https://github.com/local-inference-lab/vllm @ `6dc2f516688fe6f84c6994dcd20fddf296853a6c`; release `N/A`
  - Patch: deploy/trellismx/image-context/rootfs/opt/p8-coupled-runtime/p8_multirow_scratch.py — `sha256:c957986d0d0434a380a14c598695ee01a3274a20ec191ea1c885e48f496af4f7` — New file added to pinned base; full source supplied rather than a textual diff.; authors: Brandon Music; upstream derivations credited in repository CITATION.cff
  - Patch: deploy/trellismx/image-context/rootfs/opt/p8-coupled-runtime/p8_native_multirow_candidate.py — `sha256:dbc4cb653b4b139829709eb763652d38942945946ae1a19041f53dbe07a45bdc` — New file added to pinned base; full source supplied rather than a textual diff.; authors: Brandon Music; upstream derivations credited in repository CITATION.cff
  - Patch: deploy/trellismx/image-context/rootfs/opt/p8-mtp-bootstrap/sitecustomize.py — `sha256:797f66b76b9bce8e6f940a4b52da831dbb7576a22c32186aa9cf3831c746190c` — New file added to pinned base; full source supplied rather than a textual diff.; authors: Brandon Music; upstream derivations credited in repository CITATION.cff
  - Overlay: deploy/trellismx/image-context/rootfs/etc/python3.12/sitecustomize.py → /etc/python3.12/sitecustomize.py; sha256:2b6cbcb5edecf080ea21f5e7e876684d163e52d4766ec377a0c446c3f7d8d2ee → sha256:efed562351c003dbbc521f7482edbd940488ce1ecff16f5f58e4267ecb175e31; Exact checked RC5 runtime export; inherited base is separately digest pinned.
  - Overlay: deploy/trellismx/image-context/rootfs/opt/p8-coupled-runtime/p8_coupled_scales.py → /opt/p8-coupled-runtime/p8_coupled_scales.py; sha256:5e9220740c30bfbcb47ab0ea0857f820562b95e6a0797ff0fb0868eb6f84a0f2 → sha256:5e9220740c30bfbcb47ab0ea0857f820562b95e6a0797ff0fb0868eb6f84a0f2; Exact checked RC5 runtime export; inherited base is separately digest pinned.
  - Overlay: deploy/trellismx/image-context/rootfs/opt/p8-coupled-runtime/p8_native_kernel.py → /opt/p8-coupled-runtime/p8_native_kernel.py; sha256:e16bf26f244dbbedab6d43c1c756cf0d41a690cbd7c5be124b698791e768cd92 → sha256:dbc4cb653b4b139829709eb763652d38942945946ae1a19041f53dbe07a45bdc; Exact checked RC5 runtime export; inherited base is separately digest pinned.
  - Overlay: deploy/trellismx/image-context/rootfs/opt/p8-coupled-runtime/p8_smallm_schedule.py → /opt/p8-coupled-runtime/p8_smallm_schedule.py; sha256:804fae60a3b2e5fdea31d0fc1f343e545a45a4a3e859d617bc667bf3ea706df6 → sha256:b5d56e800c7760269608b34fa06533181d9b431aeab535e644a4d5e4620b66e2; Exact checked RC5 runtime export; inherited base is separately digest pinned.
- **b12x:** https://github.com/local-inference-lab/b12x @ `36bce2c1552ba2d47dc09f20a6f64fbfc8ec4ff8`; release `N/A`
  - Overlay: deploy/trellismx/image-context/rootfs/opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/dynamic.py → /opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/dynamic.py; sha256:8b26ab7a7b193bb78fd400cd3bf639a42069bcabfffbd29d038858d625cc179e → sha256:1283909e371eace07784fd6f628ac83997377dba649a28b09e058a26dcac8b75; Exact checked RC5 runtime export; inherited base is separately digest pinned.
  - Overlay: deploy/trellismx/image-context/rootfs/opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/p8_h128_fc1.py → /opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/p8_h128_fc1.py; sha256:97dc8c1899bab187918143da1dda7b3397d70fdf710810c79070ed970907ba40 → sha256:a38691a411e98202eabb0d2ee0085aaa98ffc71b4b9eb63a628def064cbd291f; Exact checked RC5 runtime export; inherited base is separately digest pinned.
  - Overlay: deploy/trellismx/image-context/rootfs/opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/p8_small_m.py → /opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/p8_small_m.py; sha256:478c5400d1efd7fd7c301ae4298a139e3c75b95928dc176173fe5d495e48f285 → sha256:8530dc32fb17ee1eaaf770810ce5999ecda0489b0bd2b04a5b7b8e8d749812cb; Exact checked RC5 runtime export; inherited base is separately digest pinned.

### Package changes
- None in this publication: the existing RC5 image was retagged unchanged.

### Build arguments
- BASE_IMAGE=verdictai/trellismx:glm53-flash-p8-base-v11-20260907@sha256:9846694e7dd3546f6ef3d259ecd2c129a5d54147662a76d32c5a3d15673f5e25

### Environment defaults
- Inherited exact base defaults; PYTHONPATH in Dockerfile; runtime flags in serve.sh.

### Entrypoint changes
- Published image unchanged. Compose runs /release/serve.sh and defaults to max-model-len 1000000.

- Result tree: `N/A`
- Integration patch: `N/A`

## Changes from the base image

### Inherited
- vLLM/B12X serving, calibrated NVFP4 MLA KV, P8 v11 base and dependency stack.

### Introduced
- Exact RC5 FC1-broadcast/mixed-rate sources; unchanged RC5 image published with portable Compose and 1M launch default.

### Compatibility impact
- Four SM120 GPUs, TP4/DCP1, explicit GLM carrier plus K4/K5 routed sidecars; not an arbitrary-model server.

## Tested configuration

### RC5 qualitative profile launches (not new 1M recipe)
- Hardware: 4x RTX PRO 6000 96GB SM120
- Topology: PCIe; no NVLink
- Power/clocks: 300W/GPU; +6000 memory offset; not changed by recipe
- Driver/runtime: 610.57.04; CUDA 13.3.1.008; PyTorch 2.13.0a0+9186a08; NCCL 2.31.2
- Engine: 6dc2f516688fe6f84c6994dcd20fddf296853a6c plus pinned image overlays
- Model/quant: 17-K5 / 25-K4 manifest in evidence/p8-full-coupled-campaign-20260906/phase1/k5-only-checkpoint-manifest.json; P8 E4M3 activations/MMA; 4.6587417643 routed stored bpw
- Parallelism: TP4/DCP1
- KV/speculation: nvfp4_ds_mla; per-launch capacity in runtime receipts; MTP3 greedy
- Graph/scheduler: CUDA graphs; exact profile launch in evidence; 8192 batch tokens; 32 sequences; context differs by profile
- Cache/JIT: Prefix caching enabled for qualitative runs; separate cache-disabled speed run
- Launch command:
```bash
https://github.com/brandonmmusic-max/glm53-hadamard-shapleymcg-kld/tree/main/evidence/rc5-release-20260907
```

## Validation results

### Commands
- `docker run --rm --network none --entrypoint /bin/bash verdictai/trellismx:glm53-flash-p8-rc5-20260907 -lc "/opt/venv/bin/python /opt/p8-coupled-runtime/verify_release.py"`
- `bash -n deploy/trellismx/serve.sh`
- `docker compose -f deploy/trellismx/compose.yaml config --quiet`
- `docker buildx imagetools inspect verdictai/trellismx:glm53-flash-p8-rc5-20260907@sha256:609a5fc1cd7d994ba32d9c03626c414d315947eb9f13fab474a15bc8dfbe0129`

### Results
- **smoke / Embedded runtime source integrity; CPU inspection without GPUs:** passed. Conditions: verdictai/trellismx:glm53-flash-p8-rc5-20260907@sha256:609a5fc1cd7d994ba32d9c03626c414d315947eb9f13fab474a15bc8dfbe0129. Measurement: Embedded SHA256 verifier. Result: 10 files pass; public registry digest equals working image identity. Conclusion: Source identity only; no new GPU or 1M-context qualification. Evidence: https://github.com/brandonmmusic-max/glm53-hadamard-shapleymcg-kld/tree/main/deploy/trellismx (`sha256:7d4890acda0c8bfc9fb940ea6c8bdd6992b97fed4bbdb324b0e5b604924609d7`).
- **smoke / CPU-only source reconstruction:** passed. Conditions: Pinned public v11 base; no network and no GPUs; exact exported RC5 source files. Measurement: docker build --pull=false --network=none deploy/trellismx/image-context. Result: Build completed; embedded verifier passed all 10 files. Local rebuild manifest list sha256:e1514e8478d89a16ae2151db655a7a96ad4befd3640e421bc2547010a46d39e9; this is not the published RC5 image identity.. Conclusion: Source reconstruction and hash closure, not serving or performance qualification.. Evidence: https://github.com/brandonmmusic-max/glm53-hadamard-shapleymcg-kld/tree/a3e8f95847ffa5a7708f9b654906994a80393810/deploy/trellismx/image-context (`sha256:ada01610f8a83dfe1a8922e2092fb6de8de1645b2a21535872158a9304480a9f`).

## Performance claims

- N/A

## Known limitations

- P8 uses E4M3 mxf8f6f4 MMA, 2x NVFP4 issue count for equal dimensions. Historical 0.034181 KLD is not a new RC5 measurement.
- LAVD official 28/30 includes six near scores and two token caps; Hotel 28/30 has two non-token-cap failures.

## Untested configurations

- New 1M launch startup/accuracy, independent cold GPU/JIT rebuild, universal architecture support.

## Unsupported configurations

- Automatic arbitrary-model BF16 conversion by this serving recipe; claiming P8 has native NVFP4 compute throughput.

## Support and issue routing

- Support owner: Brandon Music
- Contact: https://github.com/brandonmmusic-max
- Support commitment: `ephemeral`
- Support thread: https://github.com/brandonmmusic-max/glm53-hadamard-shapleymcg-kld/issues/5
- Thread status: `active`
- Issue tracker: https://github.com/brandonmmusic-max/glm53-hadamard-shapleymcg-kld/issues
- Upstream escalation: Keep reports in this thread. Escalate upstream only after reproduction on the recommended image or a minimal reproducer identifies the responsible source change.
- Superseded by: `N/A`

## Publication record

- Machine-readable record: https://github.com/brandonmmusic-max/glm53-hadamard-shapleymcg-kld/blob/main/deploy/trellismx/image-record.json
- Main-channel link: N/A
- Automated listing: `not-applicable`
- Maintainer approval: N/A
