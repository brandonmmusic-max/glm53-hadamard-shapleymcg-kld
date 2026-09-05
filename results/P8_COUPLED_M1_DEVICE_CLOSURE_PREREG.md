# P8 full-coupled M1 SM120 device-closure preregistration

Status: **harness prepared; no GPU execution performed**.

Protocol SHA-256:
`9556885c32385574530acf7b07c1d5636b46852ca9714c27883f98f17368efe1`.

The executable protocol is frozen in
`scripts/run_p8_coupled_m1_device_closure.py`. Its default invocation only
prints the protocol. Device execution requires the explicit `--execute` flag,
an immutable image ID, exact sidecar/design/transform hashes, the hash of the
runtime source manifest installed in that image, a selected idle GPU, and a
fresh absolute evidence directory.

## Decision question and claim boundary

Question: for one synthetic layer-3/rank-0 GLM-5.3-Flash TP-local M1 input, does
the repaired full-coupled P8 wrapper execute K4 procedural-MCG weights with
physical UE8M0/32 scales through the native E4M3 `mxf8f6f4` path while matching
the observable activation carrier bytes and an exact quantized CPU reference?

The intended evidence level is a developmental GPU smoke/numerical closure.
It is not end-to-end KLD, prefill, throughput, CUDA-graph, serving, all-rank,
all-layer, final-role, or independent-reproduction evidence. The five eager
runs are correlated determinism repeats of one prepared image/sidecar, not five
independent experimental units.

P8 remains a quality product: `mxf8f6f4` requires twice the MMA issue count of
NVFP4 for equal K. This harness does not measure that cost.

## Frozen payload and topology

- Seed: `20260905128`, generated on CPU after all runtime construction effects
  are excluded.
- Shape: M1, H4096, local I512, E288, top-k 8, TP rank 0, layer 3.
- Expert IDs, in reducer order: `0,1,17,63,127,191,255,287`.
- Runtime: the actual `P8NativeTPMoE`, with `small_m_scheduler=True`, N128,
  deterministic output, full debug capture, capped SiLU, and no forced mode.
- Required resolved dispatch: exactly `small_m=True`, `materialized=True`,
  `fc1_tile_n=128`, `tile_m=16`, `fused_scratch_zero=False`.
- Full-coupled schema, external transform hash, H512/H128 identities, K4,
  procedural MCG alpha2, E4M3, UE8M0-K32, and `ldlq=false` are mandatory.
- All five scale roles must be finite, nonzero, non-unit, and contain both
  positive and negative values. An identity or positive-only scale fixture
  cannot pass.
- The transformed input carrier must differ from a raw-input E4M3 carrier, and
  each selected expert's coupled down-input carrier must differ from a plain
  post-SiLU carrier. These are explicit anti-identity checks.

No model is encoded by the harness. The reference reader uses safetensors
slices for the eight selected experts and refuses a fallback that materializes
an entire CPU weight plane. The actual wrapper necessarily loads the supplied
single TP-rank sidecar when the future device run is authorized; no such load
or allocation occurred during this preparation.

## Frozen exact gates

Any failure stops the closure claim; output similarity cannot excuse a byte or
identity failure.

1. The immutable image resolves locally to the supplied image ID. The image's
   installed runtime-manifest hash equals the supplied hash.
2. Every actually imported coupled runtime source hashes to its entry in that
   manifest. This intentionally waits for and pins the repaired post-race,
   post-constructor-ABI image rather than the obsolete prepared image.
3. Sidecar, design and transform file hashes match their supplied identities;
   sidecar metadata links the same design/transform hashes and full-coupled
   schema.
4. Selected weight streams are packed INT16 K4 geometry. Every selected weight
   scale is a finite raw UE8M0 byte, no code is 255, and both W13 and W2 exercise
   non-identity scale codes. CPU decode must reconstruct each weight exactly as
   E4M3 bytes multiplied by its UE8M0 power-of-two scale.
5. Device debug buffers must be exactly `packed_a`, `scale_flat`,
   `intermediate_u32`, `route_output`, `token_map`, `row_counts`, and
   `expert_tile_base`. Missing or unexpected buffers fail closed.
6. The input E4M3 payload, after the frozen B12X within-K32 permutation, and all
   128 UE8M0 bytes must match the CPU transformed reference bit-for-bit.
7. For all eight physical direct-route rows (`slot * 16`), all 512 post-FC1
   E4M3 bytes and all sixteen UE8M0 bytes must match the exact quantized CPU
   reference bit-for-bit.
8. Every per-route FP32 output and final BF16 output must be finite. Each must
   have cosine strictly greater than `0.995` and relative L2 strictly less than
   `0.12` versus the exact quantized CPU reference. Max absolute error is also
   reported but is not a decision threshold. These thresholds preserve the
   existing native-P8 device-closure convention.
9. All five eager runs must have identical final-output bits and identical
   observable activation-carrier hashes.

The CPU reference includes both activation quantizers and the physical codec:

```text
BF16 -> FP16 -> H512 -> signed suh -> H128
     -> E4M3/UE8M0-K32
     -> decoded gate/up E4M3*UE8M0 MMA reference -> FP16
     -> joint H128 -> private svh -> H128 -> fixed signs -> capped SiLU
     -> fixed signs -> H128 -> private down suh -> H128
     -> E4M3/UE8M0-K32
     -> decoded down E4M3*UE8M0 MMA reference -> FP16
     -> H128 -> shared down svh
     -> ordered weighted FP32 route sum -> H512 -> BF16
```

## Observable and inaccessible seams

The present wrapper exposes enough to test both activation carriers exactly,
the route output numerically, dispatch, and the final output. It does not expose
the physical gate/up FP16 MMA results, values between the two joint H128s, the
raw physical down FP16 MMA result, or the reducer's pre-H512 FP32 sum.

One known validity pressure is deliberate: the device capped-SiLU uses its
frozen approximate reciprocal while the CPU reference uses PyTorch sigmoid.
The middle E4M3/UE8M0 byte gate tests whether those implementations close after
the specified native quantization. If they do not, the run is a failure; the
gate must not be weakened after observing the result. A future retry would
first preregister either a proven bit-exact CPU mirror or a new debug seam that
isolates the activation implementation from the surrounding transform/quant
path.

Therefore this protocol does **not** claim bitwise closure at those inaccessible
internal seams. If that stronger claim becomes required, the exact additional
debug-only hooks are:

- `fc1_physical_fp16[8,2,512]` before the joint transform;
- `fc1_joint_fp32[8,1024]` after each named H128/scale/sign boundary;
- `down_physical_fp16[8,4096]` before output H128/down-svh;
- `reducer_pre_h512_fp32[1,4096]` after ordered weighting and before H512.

The harness must fail—not substitute identity, an inherited phase, or a looser
output-only rule—if any currently required observable is absent. No runtime
source was changed to add the optional hooks in this preparation.

## Future opt-in command shape

Only after the repaired image and coupled sidecar exist and their exact hashes
are inserted:

```bash
python3 scripts/run_p8_coupled_m1_device_closure.py --execute \
  --image sha256:<64-hex> \
  --gpu-device <idle-GPU-index-or-UUID> \
  --sidecar /absolute/p8-layer-003-tp4-rank-0.safetensors \
  --sidecar-sha256 <64-hex> \
  --design /absolute/design.json --design-sha256 <64-hex> \
  --transform /absolute/transform.json --transform-sha256 <64-hex> \
  --runtime-manifest-sha256 <64-hex> \
  --output /absolute/fresh/evidence-directory
```

The outer harness performs CPU-only image/manifest checks first, refuses a
non-idle selected GPU or startup temperature above 75 C, changes no service,
then launches one network-disabled single-GPU container. Raw launch, stdout,
stderr, result and execution receipts are preserved. A failing probe remains a
failure receipt.

## Preparation validation

- `python3 -m py_compile scripts/run_p8_coupled_m1_device_closure.py`: pass.
- `python3 -m pytest -q tests/test_p8_coupled_m1_device_closure.py`: **6 passed**.
- GPU execution: **Not tested**.
- Image build: **Not performed by this task**.
- Model encoding, service mutation, KLD and performance measurement: **Not tested**.
