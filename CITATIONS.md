# Citations, inspirations, and method lineage

This file names the ideas, papers, formats, data, and code that this campaign
builds on. `THIRD_PARTY_NOTICES.md` carries the license terms; this file
carries the intellectual credit. If something here inspired a design choice,
it is listed even when no code was reused.

## Lineage in one line

QTIP (trellis-coded quantization with incoherence processing) → ExLlamaV3 /
EXL3 (procedural MCG codebook, cyclic bitstream, LDLQ) → KQuant/QSRT and the
b12x W4A8 trellis path (trellis decoded to E4M3 inside an FP8 MoE kernel,
Luke Alonso) → this repository (procedural MCG decoded in-register to E4M3
with a physical UE8M0/32 scale plane consumed by `mxf8f6f4`, MoE small-M
scheduling, end-to-end teacher-KLD Shapley allocation, sealed-role
methodology).

## Quantization methods

- Marcellin, M. W., and Fischer, T. R. "Trellis coded quantization of
  memoryless and Gauss-Markov sources." IEEE Transactions on Communications
  38(1), 1990. The origin of trellis-coded quantization.
- Viterbi, A. J. "Error bounds for convolutional codes and an asymptotically
  optimum decoding algorithm." IEEE Transactions on Information Theory 13(2),
  1967. The encoder's search.
- Tseng, A., Sun, Q., Hou, D., and De Sa, C. "QTIP: Quantization with Trellises
  and Incoherence Processing." NeurIPS 2024. arXiv:2406.11235.
  https://arxiv.org/abs/2406.11235 — the bitshift trellis, compute-based
  procedural codes, and incoherence processing that EXL3 and this codec
  descend from.
- Chee, J., Cai, Y., Kuleshov, V., and De Sa, C. "QuIP: 2-Bit Quantization of
  Large Language Models With Guarantees." NeurIPS 2023. arXiv:2307.13304 —
  LDLQ and the incoherence argument.
- Tseng, A., Chee, J., Sun, Q., Kuleshov, V., and De Sa, C. "QuIP#: Even
  Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks."
  ICML 2024. arXiv:2402.04396 — randomized Hadamard incoherence processing.
- Frantar, E., Ashkboos, S., Hoefler, T., and Alistarh, D. "GPTQ: Accurate
  Post-Training Quantization for Generative Pre-trained Transformers." ICLR
  2023. arXiv:2210.17323 — the Hessian-based error feedback used by the P8
  encoder ("GPTQ-style inter-group feedback").
- turboderp. ExLlamaV3 and the EXL3 format.
  https://github.com/turboderp-org/exllamav3 — the procedural MCG codebook
  constants, cyclic bitstream layout, and tensor-core tile convention ported
  here under `LICENSE.exllamav3`; the EXL3 4-bpw GLM-5.3-Flash checkpoint is
  the serving and quality comparator throughout.
- Ashkboos, S., Mohtashami, A., Croci, M., Li, B., Jaggi, M., Alistarh, D.,
  Hoefler, T., and Hensman, J. "QuaRot: Outlier-Free 4-Bit Inference in
  Rotated LLMs." NeurIPS 2024. arXiv:2404.00456 — rotation folded into
  weights with an online Hadamard on the down-projection input; the model for
  the runtime activation transforms tested here.
- Liu, Z., Zhao, C., Fedorov, I., Soran, B., Choudhary, D., Krishnamoorthi,
  R., Chandra, V., Tian, Y., and Blankevoort, T. "SpinQuant: LLM Quantization
  with Learned Rotations." arXiv:2405.16406, 2024 — learned orthogonal
  rotations, the precedent for the Cayley-learned block rotations in V3.

## Formats, instructions, and hardware

- Rouhani, B. D., et al. "Microscaling Data Formats for Deep Learning."
  arXiv:2310.10537, 2023, and the Open Compute Project "OCP Microscaling
  Formats (MX) Specification v1.0," 2023 — MXFP8/MXFP6/MXFP4 with UE8M0 block
  scales; the P8 scale plane is an MX plane.
- NVIDIA. NVFP4 (E2M1 with UE4M3 per-16 scales) and the PTX ISA warp-level
  block-scaled MMA (`mma.sync ... kind::mxf8f6f4` and `kind::mxf4nvf4`).
  https://docs.nvidia.com/cuda/parallel-thread-execution/ — the instructions
  the P8 and P4 prologues target.
- NVIDIA CUTLASS and CuTe DSL; NVIDIA TensorRT Model Optimizer (ModelOpt) —
  the NVFP4 checkpoint format and packer used for every NVFP4 arm.

## Allocation, statistics, and methodology

- Shapley, L. S. "A Value for n-Person Games." Contributions to the Theory of
  Games II, 1953.
- Castro, J., Gómez, D., and Tejada, J. "Polynomial calculation of the Shapley
  value based on sampling." Computers & Operations Research 36(5), 2009 —
  permutation sampling; the antithetic-permutation estimator here is a
  variance-reduced variant.
- Zhao, J., Derakhshan, A., Hyman, J. K., Dong, J., Abdu Jyothi, S., and
  Harris, I. "CoopQ: Cooperative Game Inspired Layerwise Mixed Precision
  Quantization for LLMs." arXiv:2509.15455, 2025; Findings of ACL 2026 —
  prior use of Shapley estimation for layerwise mixed-precision allocation in
  dense LLMs. ShapleyMCG differs in valuing whole routed MoE layers by
  end-to-end teacher KLD under exact byte budgets, but CoopQ is the closest
  published precedent and should be cited whenever the allocation is described.
- Related MoE allocation work: "Q-Strata: Hierarchical Bit Allocation for
  Mixed-Precision Quantization of Mixture-of-Experts LLMs," arXiv:2608.30564,
  2026; "BitsMoE: Efficient Spectral Energy-Guided Bit Allocation for MoE LLM
  Quantization," arXiv:2606.00079, 2026.
- Kullback, S., and Leibler, R. A. "On Information and Sufficiency." Annals of
  Mathematical Statistics 22(1), 1951 — the primary metric,
  `KL(teacher || student)`.
- Efron, B. "Better Bootstrap Confidence Intervals." Journal of the American
  Statistical Association 82(397), 1987 — the BCa intervals used for every
  paired comparison.
- Nosek, B. A., Ebersole, C. R., DeHaven, A. C., and Mellor, D. T. "The
  preregistration revolution." PNAS 115(11), 2018 — the practice behind the
  sealed plans, hashed analysis code, and role separation used here.

## Models and data

- Z.ai / zai-org. GLM-5.3-Flash and GLM-5.3-Flash-BF16.
  https://huggingface.co/zai-org/GLM-5.3-Flash-BF16 — the model; model terms
  apply.
- Moonshot AI. "Kimi Linear: An Expressive, Efficient Attention Architecture."
  arXiv:2510.26692, 2025 — Kimi Delta Attention, the linear-attention layers
  in GLM-5.3-Flash's hybrid architecture.
- DeepSeek-AI. DeepSeek Sparse Attention (DeepSeek-V3.2-Exp technical report,
  2025). https://github.com/deepseek-ai/DeepSeek-V3.2-Exp — the sparse
  attention indexer family used by the full-attention layers.
- Lasby, M., Lazarevich, I., Sinnadurai, N., Lie, S., Ioannou, Y., and
  Thangarasa, V. "REAP the Experts: Why Pruning Prevails for One-Shot MoE
  Compression." arXiv:2510.13999, 2025; ICLR 2026.
  https://github.com/CerebrasResearch/reap — the REAP calibration corpus
  family used for Hessians and fit roles.
- Music, B. M. "GLM-5.3-Flash BF16 Teacher Full-Vocabulary Logits."
  https://huggingface.co/datasets/brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits
  — the pinned teacher (BF16 on 4x B200, FP32 logits, 640 role windows plus
  25 sealed final windows).
- local-inference-lab/GLM-5.3-Flash-NVFP4 and LibertAIDAI/GLM-5.3-Flash-NVFP4
  — the stock NVFP4 carriers and comparators.

## Software, kernels, and people

- Luke Alonso. b12x (https://github.com/lukealonso/b12x, mirrored at
  https://github.com/local-inference-lab/b12x), KQuant, and QSRT. The August
  2026 b12x W4A8 trellis path decodes the QSRT trellis to E4M3 inside the FP8
  MoE kernel with a T12 table and coupled H512/H128 incoherence; this
  repository's P8 producer/MMA separation follows that design and replaces
  the table with the procedural MCG law and a physical MX scale plane. The
  KQuant/QSRT snapshot carried no license file; see `THIRD_PARTY_NOTICES.md`.
- Martin Vit. Co-author of the b12x GLM-5.2 SQG W4A8 stack (144 commits in
  the archived coupled worktree, August 2026), including the integration
  merges the coupled-incoherence archive audit relies on.
- Michel Belleau, derek, and MadeBy561 — additional b12x contributors in the
  archived worktree.
- The vLLM project (Kwon, W., et al. "Efficient Memory Management for Large
  Language Model Serving with PagedAttention," SOSP 2023, arXiv:2309.06180)
  and the local-inference-lab/vllm fork that hosts the Humming NVFP4 MoE
  backend, the B12X sparse-MLA attention, and the KDA/DSA GLM-5.3 support.
- FlashInfer (Ye, Z., et al. "FlashInfer: Efficient and Customizable Attention
  Engine for LLM Inference Serving," arXiv:2501.01005, 2025) and the
  voipmonitor/flashinfer fork — the FP8 sparse-MLA attention path.
- InstantTensor, upstream https://github.com/scitix/InstantTensor, used
  through the voipmonitor/InstantTensor fork — the indexed loader that made
  matched BF16 overlays and fail-closed candidate loading possible.
- cstechdev — the NoPE-capable GLM-5.3 serving image used for the first NVFP4
  bring-up.
- PyTorch, safetensors, NumPy, SciPy, PyArrow.

## Related work not used

- cnygaard/glq — lattice and trellis-coded PTQ with fused CUDA kernels
  (https://github.com/cnygaard/glq).
- "BCJR-QAT: A Differentiable Relaxation of Trellis-Coded Weight
  Quantization," arXiv:2605.10655, 2026.
- "HARP: Hadamard-Preconditioned Adaptive Rotation Processor for Extreme LLM
  Quantization," arXiv:2605.29843, 2026.

## Citing this repository

See `CITATION.cff`. The method and license authority is
https://github.com/brandonmmusic-max/shapleymcg.
