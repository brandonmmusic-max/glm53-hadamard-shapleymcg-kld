# Coupled P8 image v3: completed CPU build

Date: 2026-09-05. Evidence level: structural. GPU use: none.

- Source commit: `ee5240641ea26d48ba324b07b546b1bc7f4d0e1d`.
- Image: `sha256:77ae1c0b46c7f72b085b6ae3f8184f0f9df229bce5306d1849fb770501dc0d64`.
- Tag: `klc/glm53-p8-coupled:v3`.
- Installed manifest SHA256: `ff5ce6158e2d2134cedd1a451bc911fc3623c3ba93513916556a61833ee496d2`.
- Build receipt SHA256: `403b073ad557e75da02d5253ff250d6c787424235dd31d6a1ef069509250d493`.
- Evidence directory: `/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-image-v3-build`.

The builder exited zero. The final image verifier returned `status=pass`,
`errors=[]`; installed hashes, exact parent identity, constructor/call ABI,
constructed FC1/FC2 resource contracts, import origins and Python compilation
passed. Independent `docker image inspect` matched the image ID; a CPU-only
`sha256sum` invocation inside that immutable image matched the manifest hash.

The frozen build reused the existing clean detached worktree named
`bmxfp4-glm53-p8-coupled-image-v2`, switched to the source commit above; its
directory name is not the candidate identity. The separate v2 staged context,
failure receipt and logs were preserved. V3 fixed a class-versus-instance
verifier error without modifying runtime kernel sources or relaxing the
65,536-byte FC1 allocation requirement.

Replay after preparing a clean worktree at the source commit (use a fresh
output directory and do not overwrite an existing tag):

```bash
python3 scripts/build_p8_coupled_image.py \
  --output /absolute/fresh/p8-coupled-image-v3-build \
  --source-commit ee5240641ea26d48ba324b07b546b1bc7f4d0e1d --execute
```

This is not CUDA compilation, device closure, graph parity, KLD, speed, or
production qualification. Attention backend/KV dtype: not exercised. Intended
MoE path: coupled native P8 M1 decode and M64/N128 materialized prefill;
weights/activations E4M3 with UE8M0/32, K4 payload 4.25 bpw plus channel-scale
metadata. P8 mxf8f6f4 requires twice NVFP4's MMA issue count. No performance
conclusion is supported by this build receipt.
