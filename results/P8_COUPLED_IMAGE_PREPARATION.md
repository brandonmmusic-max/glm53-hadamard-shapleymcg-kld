# P8 coupled image preparation

Status: **prepared only; image not built**.

## Immutable base and scope

- Parent image: `sha256:0336113e0fff876cccf9e6ac5347528ae59f4ad894a0ce7cb4c4e90b4651a745`
- Parent local tag observed during the CPU-only audit: `klc/glm53-p8-tail-repair:v2`
- Dedicated future tag: `klc/glm53-p8-coupled:v1`
- The dedicated tag was absent at preparation time. The builder refuses to
  overwrite it if it later exists.
- Required integration ancestor: `eded4c349f0bec82634ccf0fa0749be833665670`
- The future builder requires the caller to supply the exact clean-worktree
  `HEAD`; it embeds both that commit and its tree hash in image labels and the
  receipt. The recipe does not pin a V2 encoder-design receipt or sidecar.
- The image is labeled `research-only-not-device-qualified` and
  `m1-n128-only-unqualified`. No KLD, throughput, prefill, decode, CUDA graph,
  or device-closure claim is made by this preparation.

## Parent inspection receipts

The immutable parent was inspected with `docker run --runtime=runc`,
`--network=none`, and `NVIDIA_VISIBLE_DEVICES=void`. It resolved:

- `b12x.moe._shared.kernels.dynamic` from
  `/opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/dynamic.py`;
- `sitecustomize` from `/usr/lib/python3.12/sitecustomize.py`;
- `PYTHONPATH` as
  `/opt/exllamav3:/opt/infernal-invocation/vllm:/opt/infernal-invocation/b12x`.

The Dockerfile checks these inherited bytes *before* any copy:

| Inherited target | Parent SHA256 |
| --- | --- |
| `/usr/lib/python3.12/sitecustomize.py` | `43d81125d92376b1a69d53a71126a041cc9a18d8080e92dea0a2ae23be138b1e` |
| B12X `dynamic.py`, source and site copies | `7211657f8508dfbf12d2f4158562144fe86741d1f72ea32cc1dc2d6c3dc6b4d6` |
| B12X `p8_small_m.py`, source and site copies | `a0c398e9d412672d1138c69a5c0c3677bb29b7215379c701062d29b8b9b9265f` |
| B12X `p8_narrow_fc1.py`, source and site copies | `84d28a0a72b474ce3fde4eeb11a371bfb1c58e843928382992aa99457409f8d1` |
| Tail-V2 `kpool_compress.py` in the imported vLLM tree | `494192195da43c46d99a684555fc10fd13a19e89288cb9f51da2536ccdf1f251` |

The parent also had no `p8_h128_fc1.py`, `p8_coupled_topk.py`, or
`/opt/p8-coupled-runtime`; those absence checks are part of the Dockerfile.

## Prepared implementation

- `scripts/build_p8_coupled_image.py` defaults to preparation only. An actual
  build requires the explicit `--execute` flag.
- The builder validates the parent ID, a clean exact source commit, the audited
  integration ancestor, every manifest source hash, Tail V2, and tag absence.
- It stages only declared source files into a minimal context, so model files,
  role data, results, and unrelated worktree content are not sent to Docker.
- The Dockerfile copies the coupled top-level runtime into
  `/opt/p8-coupled-runtime`, replaces the actual imported `sitecustomize`, and
  copies all five B12X kernel sources to both the editable source tree and the
  site-packages mirror.
- The in-image verifier checks every installed byte, `py_compile`s every
  installed Python source, verifies Tail V2 again, and proves the expected
  import origin for every coupled module. The future builder repeats this with
  a CPU-only `runc` container and validates image labels after the build.

Prepared-only invocation after committing all intended integration inputs:

```bash
python3 scripts/build_p8_coupled_image.py \
  --output /absolute/fresh/preflight-directory \
  --source-commit <full-clean-HEAD>
```

The same invocation with `--execute` is intentionally deferred until the
baseline run permits image building.

## CPU/static validation completed

- `python3 -m py_compile scripts/build_p8_coupled_image.py runtime_patch/p8_coupled_image/verify_image.py`: pass.
- `python3 -m pytest -q tests/test_build_p8_coupled_image.py`: **4 passed**.
- Manifest verification: all **11** declared image inputs matched their pinned
  SHA256 values.
- Dedicated future tag check: absent.
- Docker build: **not run**.
- GPU use: **none**.
