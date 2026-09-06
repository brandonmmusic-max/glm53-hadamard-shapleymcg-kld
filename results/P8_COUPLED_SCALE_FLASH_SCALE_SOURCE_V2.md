# P8 coupled-scale Flash source and preparation audit

Date: 2026-09-05

Status: CPU-only preparation complete; no encoding, GPU work, build, service,
runtime, or KLD job was launched.

## Verdict

The target-Flash scale source was already present locally. The sealed
checkpoint at
`/home/brandonmusic/models/GLM-5.3-Flash-EXL3-4bpw` contains the exact
E288/H4096/I2048 FP16 `suh`/`svh` vectors for layers 3, 20, and 22. It is a
stronger and less ambiguous preparation source than resizing the full GLM-5.3
H6144/E256 vectors or attempting a new CPU approximation of the GPU GSS.

The new loader accepts that indexed checkpoint directly and fails closed on a
bad top-level seal, config/index hash drift, wrong K4/MCG semantics, wrong
Flash geometry, missing shard receipts, changed scale payload bytes, or broken
gate/up-SU and down-SV sharing.

The immutable preparation record is
`results/P8_COUPLED_SCALE_FLASH_EVIDENCE_PREPARATION_V1.json`, SHA-256
`396c5b8cbad7476a48cd540444e8e3e58de2cb15bff516ab479ee77dc1efdcdc`.
It binds the BF16 index, sealed REAP capture, role file, exact EXL3 checkpoint
receipt, encoder/reference sources, three frozen layers, K4/4.25-bpw weight
ABI, and separate `0.003979859528718171`-bpw coupled metadata.

## What actually produced these scale roles

The evidence does **not** support describing the production Flash vectors as
the current QSRT wrapper's `apply_out_scales=False` regularizer. The actual
published GLM-5.3 preparation path is the target-neutral R10/v31 numeric core
plus the GLM-5.3 adapter:

1. The selected profile is fixed to `energy_balanced` permutation and
   `per128-grid`; selection and confirmation rows are explicitly excluded from
   profile choice (`glm53_mcg_preparation.py:58-85`).
2. On `fit`, routed P2 mass, gate/input diagonals, candidate middle diagonals,
   and down output energy are accumulated per expert; layer-shared statistics
   are mass-weighted (`glm53_mcg_preparation.py:126-181`).
3. The fixed scale family and deterministic signs are assigned with one
   gate/up input vector and one down output vector shared by the layer, while
   gate/up output and down input vectors remain expert-private
   (`glm53_mcg_preparation.py:185-220`).
4. Absolute v31 normalization computes private gate/up relative output RMS,
   then the shared gate/up input magnitude as a mass-weighted geometric mean
   of post-right-H128 row RMS. A private beta residual stays on each gate/up
   `svh` (`absolute_v31.py:469-513`).
5. Down `svh` is the mass-weighted shared relative-output profile; each down
   `suh` receives its own post-right-H128 absolute row RMS
   (`absolute_v31.py:515-537`).
6. A pinned per-matrix GSS is run at the actual rate, and its scalar is folded
   only into the private side before FP16 storage
   (`glm53_mcg_preparation.py:358-400`).
7. Full gate/up Hessians are routed-P2 uncentered covariances. Down Hessians are
   recomputed after exact gate/up K4 decode on conditional-fit rows; the factor
   cache is cleared before the down encode
   (`glm53_prepared_backend.py:490-548`).

The Flash release identifies the same hash-verified R10 Python closure and
pinned v31 numeric core (`glm-5.3-flash-exl3-4bpw-release/README.md:28-46`).

## Local evidence and exact identities

- Materialization receipt schema:
  `quant-pipeline.glm53-k4-materialization-receipt.v1`
- Materialization receipt SHA-256:
  `092be1ffa8db66bf02d4c370d0433a57aa48d4a6e5ce89723ef6a3bb7ca32643`
- Source model revision:
  `a6c167b62691b2bac901344b65cb651a70f53e43`
- Index SHA-256:
  `2f64d21c67c90bbafeb36c4e9b2f06f54063ed439e9f7cf95962d425a1d8515d`
- Local REAP capture SHA-256:
  `6a057e03808b2a7bc04bd6c757862f3954b166eba31fbc362c91cc966cd2e4b8`
- Capture geometry and roles: layers 3..44, H4096, E288, top-k 8, roles
  `fit`, `conditional-fit`, `selection`, `confirmation`.
- Scale metadata per layer: 3,555,328 bytes.
- Layer 3 shared gate/up `suh`:
  `244f05112c2507a2b681202bba8113e599420d7aa664e69f488752dd8b9b8e56`
- Layer 3 shared down `svh`:
  `90047fad1b5b77c9549a5688f021083fddebc9bdd878f8d0020e0c40b8386615`
- Layer 20 shared gate/up `suh`:
  `22b733ca9e43aad8828943c82b72814c70a5e799ef546ae527c8345583b9f8af`
- Layer 20 shared down `svh`:
  `7abc9646efb0451dc2145de101b78152f257107a9bb34b81a7852a848fdfcaaf`
- Layer 22 shared gate/up `suh`:
  `90796aaf01cd8cb9b97589378fa3d8835bfab10c035fcec10285ded9cd3142cb`
- Layer 22 shared down `svh`:
  `1dacd0f5e064bc0e7852f495b24dbc6927ea72e5c888ce12f783444496182d56`

All 1,728 scale tensors per layer were read from only the indexed shards named
by the model index. Every selected tensor's raw payload hash matched its sealed
shard receipt, gate/up `suh` was byte-identical across all 288 experts and both
projections, and down `svh` was byte-identical across all 288 experts.

## Why no new CPU fit was fabricated

The raw REAP capture needed to replay profile statistics is present and sealed.
However, a faithful fresh fit also requires the pinned v31 block primitives,
all target BF16 triplets, the fixed profile/seed receipt, and a per-matrix K4
GSS against the actual MCG quantizer. The archived implementation performs
that GSS and subsequent encode on CUDA. No complete local
`preparation.safetensors` / `profile-decision-statistics.safetensors` set was
found for layers 3, 20, and 22 in the searched roots.

Accordingly, this branch does not call a weight-only CPU heuristic a refit. It
imports the already-fitted, exact target-Flash vectors from the sealed K4
checkpoint. If that checkpoint or its receipts are absent or drift, preparation
stops. A fresh refit remains a separate authorized GPU campaign, not a hidden
fallback.

## ABI and next executable step

The existing P8 stream is unchanged: K4 procedural MCG trellis plus UE8M0/32
weights remains exactly 4.25 bpw. Coupled FP16 scales/draw metadata remain a
separate approximately 0.004-bpw sidecar. The encoder entrypoint can now take
the indexed Flash checkpoint as `--exl3-scales`; it validates the same sealed
source identity embedded in the preparation record before any CUDA allocation.
Only layers 3, 20, and 22 are accepted.
