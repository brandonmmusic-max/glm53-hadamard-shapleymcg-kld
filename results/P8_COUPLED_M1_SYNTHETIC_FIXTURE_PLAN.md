# P8 coupled M1 synthetic full-shape fixture plan

Status: **generator prepared; full fixture not generated**.

This is a developmental device-closure fixture, not a quantized model and not
quality evidence. It reads no model, calibration, Hessian, REAP, teacher-logit,
EXL3, or V3 preparation artifact. Its only existing input is the repository's
610-byte frozen draw-0 transform receipt, SHA-256
`093d219b18ba32471adcee746442b1481c7ba5659bbea62b94f1a665d4343a12`.

The synthetic design, sidecar, and receipt use their own
`glm53.p8-coupled.synthetic-m1-fixture-*` schemas and explicitly record
`synthetic=true` and `quality_claim_allowed=false`. They must not be inserted
into the real three-layer V3 preparation or any KLD comparison.

## Frozen shape and bytes

- Layer 3, TP rank 0 of 4, E288, H4096, local I512, K4.
- Deterministic seed `20260905129`.
- K4 streams: counter-addressed SplitMix64 low-16-bit words, independent fixed
  tags for W13 and W2. The bits are decoded only through the frozen procedural
  MCG-alpha2 finite-E4M3 law.
- UE8M0 planes: repeating raw codes `126,127,128,129`, i.e. non-identity and
  never the reserved code 255.
- All five suh/svh roles: repeating signed nonzero, non-unit FP16 values
  `{0.625,-0.75,0.875,-1.125,1.25,-1.375,0.5,-1.5}` with role offsets.
- Fixed coupled sign draw 0 and exact H512/H128 transform contract.

Exact payload accounting:

| tensor | bytes |
|---|---:|
| `w13_trellis` | 603,979,776 |
| `w2_trellis` | 301,989,888 |
| `w13_scale_ue8m0` | 37,748,736 |
| `w2_scale_ue8m0` | 18,874,368 |
| stored scale/draw metadata tensors | 901,408 |
| total tensor payload | **963,494,176** |

The safetensors header is computed before generation and is included in the
default plan output. The exact full file is below 965 MB. Weight payload remains
exactly 4.25 bpw; stored scale/draw metadata is 0.003979859528718171 bpw, with
the separately regenerated 3,072 sign bytes included in the accounted rate.

## Streaming and budget gates

The generator writes one `.partial` file directly in safetensors layout using
at most 1 MiB output chunks. The SplitMix work array is bounded by the same
chunk geometry. It never creates a BF16 plane, a dense decoded plane, or a
second codec payload. Tensor hashes are accumulated during the write; the
fixed-length placeholder hashes are replaced in-place in the header, then the
file is atomically renamed. A second bounded read verifies every stored tensor
hash.

Full generation is opt-in. It requires:

- a new absolute output directory inside an existing absolute budget root;
- existing regular files under that budget root plus the forecast sidecar,
  design, and 1 MiB receipt allowance to remain at or below 30,000,000,000 B;
- filesystem free bytes at least equal to those forecast writes;
- the exact repository transform receipt hash;
- no pre-existing output or `.partial` file.

No large fixture has been generated in this preparation task.

## CPU validation and device linkage

The unit test generates only E2/H512/I128, which preserves legitimate H512 and
H128 alignment while remaining small. It verifies the streaming header patch,
all raw tensor hashes, safetensors shapes, runtime
`validate_coupled_component`, signed/non-unit scale roles, and sampled
non-identity UE8M0 codes. It does not claim the E2 fixture can enter the actual
M1 wrapper, whose small-M contract correctly requires E288.

Future full fixture generation:

```bash
python3 scripts/generate_p8_coupled_m1_synthetic_fixture.py --generate \
  --budget-root /absolute/synthetic-campaign-root \
  --output-dir /absolute/synthetic-campaign-root/rank0-fixture-v1
```

The resulting `receipt.json` prints the complete command for
`scripts/run_p8_coupled_m1_device_closure.py`, leaving only the immutable
repaired image ID, installed runtime-manifest hash, and idle SM120 GPU to pin.
That later device run remains responsible for actual wrapper construction,
kernel compilation, exact activation-carrier closure, output closure, and
five-run determinism.

## Claim boundary

A successful synthetic-device run can establish only that one frozen
full-shaped byte pattern traverses the intended layer-3/rank-0 M1 coupled
wrapper and closes against the reference. It cannot establish model quality,
KLD, real checkpoint compatibility, prefill, throughput, CUDA-graph behavior,
TP4 collectives, or production serving.

Preparation validation:

- generator `py_compile`: pass;
- small streaming counterpart and policy tests: 5 passed;
- combined fixture, closure, ABI, and coupled-runtime tests: 39 passed;
- full 963,497,568-byte sidecar generation: not run;
- GPU, image build, model read, and service action: not run.
