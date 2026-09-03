# Third-party notices

This repository records experiments that interoperate with third-party models
and software. Those components are not relicensed by the ShapleyMCG License.

- GLM-5.3-Flash BF16: `zai-org/GLM-5.3-Flash-BF16`; model terms apply.
- GLM-5.3-Flash NVFP4 carrier: `local-inference-lab/GLM-5.3-Flash-NVFP4`;
  model terms apply.
- vLLM: Apache-2.0.
- B12X: Apache-2.0; the license accompanying the modified runtime sources is
  retained at `runtime_patch/b12x_h16/LICENSE.b12x`.
- InstantTensor: see its upstream repository.
- PyTorch: BSD-style license.
- NVIDIA CUDA, ModelOpt, and container components: NVIDIA terms apply.

The `runtime_patch/b12x_h16` files are modifications of the pinned B12X
runtime sources. Their upstream notices and the ShapleyMCG attribution must be
retained when redistributed.
