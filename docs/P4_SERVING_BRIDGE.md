# Native-RNE P4 encoder and GLM TP4 sidecars

The bridge connects the one-matrix `glm53-p4-mcg-matrix.v1` codec to the
six physical tensors consumed by the GLM P4 MoE endpoint. The serving schema
is `glm53-p4-mcg-tp-rank.v2`. Its law is exactly
`procedural-mcg-alpha1-rne-e2m1`: uint32 wrapping MCG arithmetic, one binary16
RNE addition, then finite-saturating E2M1 RNE with signed zero preserved.
It does not accept the kernel prototype's v1 lower-magnitude tie rule.

The code in this change is CPU-validated structural evidence. Actual CUDA
Viterbi quantization, P4 kernel execution, serving closure, GLM KLD and speed
have **not** been measured by this bridge's tests. The target P4 instruction
is `mxf4nvf4 m16n8k64` with E2M1 operands and E4M3/16 scales. P8 instead uses
`mxf8f6f4 m16n8k32` at twice the MMA issue count for the same logical K span;
P8 is a quality path and P4 is the intended NVFP4 speed path. Neither opcode
selection nor these CPU tests proves throughput.

## Encoder and calibration

`p4_encoder.encoder_state_lut()` generates an **offline** 65,536-byte E4M3
label table for KQuant's Viterbi encoder. Every label is numerically E2M1 and
has exactly the native nibble's zero sign. The complete table matches the
independent scalar IEEE-half/RNE oracle at all 65,536 states. The table is
not stored in matrices or rank sidecars. Runtime state-table storage is zero;
the largest fixed numeric alphabet here is 126 E4M3 scale values.

`encode_p4_matrix(weight, hessian, ...)` calls the existing
`_encode_nvfp4_with_gptq_feedback` path with this fixed law. Viterbi chooses
closed paths in each native 16-column group. The full calibration Hessian
propagates the chosen group's output error into later groups. Activation
ordering stays within each physical group; stored streams return to native
column order. The Cholesky inverse and static permutation are computed once
and reused across scale-refinement passes. This is the repository's existing
GPTQ-style feedback, with no LDLQ or BlockLDLQ call.

After each refinement pass, block and global scales are jointly refit by
fixed-code weight SSE, projected onto the 126 strictly positive E4M3 values
and a positive FP32 global. This scale subobjective is explicitly different
from the full-Hessian feedback objective; the implementation does not claim
that every scale update minimizes the complete output quadratic. Reports
include per-projection weight NMSE, the full-Hessian output NMSE, scale
history, Hessian identity, law identity and the actual refinement count.
KLD remains the acceptance metric for the calibrated experiment.

`quantize_p4_layer` requires pinned BF16 shard hashes, an index hash, fit-role
hash, and REAP capture-manifest hash. It additionally requires the exact
resolved path, byte count and SHA-256 for **each materialized local** hidden,
expert-ID and route-weight capture file. These local pins cover sparse
fit-only captures independently of the remote/full capture manifest. All
three files are verified once before any CUDA transfer or selection, and
the verified identities are included in every matrix receipt and checked
again on resume. Same-length corruption, different paths and incomplete
pin sets fail closed. It opens only the `fit` role using
`LayerCapture(..., sampling_strategy="domain-balanced")`. Gate/up share the
route-weighted Hessian of the candidate's E2M1/E4M3 activation conversion.
Down uses the Hessian of quantized SwiGLU activations from the encoded gate
and up matrices. The encoder's activation QDQ matches the explicit P4
producer policy: per-16 max/6, positive E4M3 saturation, native E2M1 RNE;
zero activation groups may have zero SFA. These are calibration inputs, not
new evidence that the activation path is qualified on GLM.

The checked selection adapter rejects non-cyclic paths or backend labels
that disagree with the native decoder. The CPU sensitivity test substitutes
a small deterministic candidate-path selector for CUDA Viterbi and exercises
the **actual GPTQ feedback function**: changing the Hessian's off-diagonal
entries changes the next group's inputs and selected path. A cached inverse
produces identical outputs to fresh preparation. This proves the Hessian
is connected to selection; it does not replace a real calibrated GPU run.

## Physical tensor ABI

Let `E` be the global expert count, `H` the hidden width, `I` the full
intermediate width, and `J=I/4`. Production GLM uses all 288 experts. The
synthetic fixture uses three experts to make every field cheap to inspect.
`H` is a multiple of 64 and `I` a multiple of 256, so both GEMMs have K64
alignment after TP4 sharding.

| Tensor | Type | Shape | Ordering |
|---|---|---|---|
| `w13_trellis` | little-endian I16 | `[2,E,H/16,J/16,64]` | `[gate,up]`, global expert, K tile, N tile, stream words |
| `w2_trellis` | little-endian I16 | `[E,J/16,H/16,64]` | global expert, K tile, N tile, stream words |
| `w13_scale_e4m3` | U8 | `[2,E,J,H/16]` | `[gate,up]`, global expert, native N row, K16 group |
| `w2_scale_e4m3` | U8 | `[E,H,J/16]` | global expert, native N row, K16 group |
| `w13_global_scale` | F32 | `[2,E]` | `[gate,up]`, global expert |
| `w2_global_scale` | F32 | `[E]` | global expert |

**All** W13 planes use `[gate,up]`. The P8 B12X scale plane's `[up;gate]`
repack is not copied into P4. Rank `r` owns intermediate indices
`[r*J,(r+1)*J)`: those are gate/up **rows** and down **columns**. Experts are
not divided between ranks. No branch symbols are reconstructed or changed
during sharding: a cyclic state boundary stays inside its original 16x16
tile. Each runtime scale plane is logical row-major E4M3/16; the P4 prologue
loads physical scales from it. Any kernel staging or swizzle is not persisted
in these sidecars.

Scale bytes must be `0x01..0x7e`. Zero, sign-bit and NaN scale codes are
rejected. The positive FP32 global is always present, even if it is one. It
must multiply that projection's accumulation; folding it away without a
compatible re-encode would change the weights. Runtime must reject v1 and
P8 metadata, wrong ranks/designs, wrong shapes, malformed inventories and
hidden codebook tensors. Whole-file hashes in the serving manifest bind
every rank before allocation/loading.

## Exact stored rate

A rank holds `W=3*E*H*J` logical weights. Stream bytes are `W/2`, scale bytes
are `W/16`, and the three projection globals consume `12*E` bytes:

```text
stream plus block scales = 4.5 bpw
rank payload             = 4.5 + 32/(H*J) bpw
rank file                = payload + 8*(8 + padded_header_bytes)/W bpw
```

The complete TP4 export charges **four copies** of every FP32 global:
`48*E` bytes per layer. It is therefore `4.5 + 128/(H*I)` payload bpw across
the unique full-layer weights, not exactly 4.5. No runtime LUT, hidden tail
state, decoded weight cache or alternate law selector is omitted. The JSON
manifest and encoding receipts are auxiliary metadata, reported separately
from the weight files. `rank_accounting` reports integer bytes and reduced
rational bpw, including each complete safetensors header.

## Export, resume and replay

The calibrated encoder writes each expert projection as a matrix sidecar
and immediate source-bound receipt. A matrix manifest can cover a disjoint
expert range. `build_p4_tp4_sidecars` accepts several such manifests but
requires every expert and all three projections exactly once. It rejects
mixed source designs, inconsistent shapes, untrusted file hashes and legacy
law labels. No source matrix is read before its pinned file hash is checked
by the matrix decoder.

The optimized exporter loads each source expert once and slices it into all
unfinished ranks in memory. It avoids four separate matrix reads/hashes and
does not decode or refit the weights. Ranks are serialized in deterministic
order. `export-inputs.json` binds the input manifests before output begins.
`--resume` requires that same input plan and verifies every finished rank's
immutable receipt, file hash, canonical container and accounting. It skips
only those verified ranks. Failed partial files and completed prefix ranks
are retained; existing finished weights are never overwritten. Per-stage
timings and I/O counters go into separate receipts, so they cannot perturb
the deterministic rank bytes. Tests compare the optimized export byte for
byte with the direct serial assembler and cover a deliberately interrupted
prefix followed by resume.

The serving manifest is `glm53-p4-mcg-tp4-manifest.v1` with
`source_design_sha256`, `codec_schema=glm53-p4-mcg-tp-rank.v2`,
`world_size=4`, `layers`, and `entries` containing layer/rank, exact basename,
bytes and SHA-256. Its per-rank accounting includes all six tensors. The
serving integration can consume it through `GLM53_P4_NATIVE_MANIFEST` along
with the sidecar directory and the source design.

CPU replay, without touching GPU devices or any data roles:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest -q tests/test_p4_serving_bridge.py tests/test_p4_codec.py tests/test_trellis_nvfp4.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python3 -m glm53_nvfp4.p4_serving_fixture evidence/opened/codec-v2/p4-bridge-v2/fixture
```

To generate a second fixture in a **new** directory, append `--write` to the
fixture command. Its three synthetic experts have different gate/up/down
streams, scales and globals. It covers all four rank shards, every physical
projection, native nibble and scale bytes, route weights, clipped SwiGLU,
FC2 activation quantization, and exact TP4 summation. It preserves full
CPU stage/output hashes. Source inputs and expected outputs are public
synthetic data; no confirmation or final logits are accessed.

Authorized calibrated build entry points, after the parent schedules GPU
time and freezes a real design:

```bash
python3 -m glm53_nvfp4.quantize_p4_layer --source BF16_ROOT --source-index INDEX_JSON --roles FIT_ROLES_JSON --capture-root REAP_CAPTURE --design P4_DESIGN_JSON --layer 3 --expert-start 0 --expert-end 288 --output-dir MATRIX_DIR --device cuda:0
python3 -m glm53_nvfp4.build_p4_tp4_sidecars --matrix-manifest MATRIX_DIR/matrices.json --design P4_DESIGN_JSON --output-dir RANK_DIR
```

The design schema is `glm53-p4-viterbi-rne-encoder-design.v1`, with `law`,
`ldlq=false`, `objective=viterbi-with-full-hessian-gptq-feedback`,
`source_precision=BF16`, `source_index_sha256`, `source_file_sha256` mapping,
`fit_roles_sha256`, `capture_manifest_sha256`, `capture_files_by_layer`, `data_role=fit`,
`sampling_strategy=domain-balanced`, `samples`, `layers`, `experts=288`,
`hidden`, `intermediate`, `scale_refinement_iterations`, `search_grid`,
`tailbite_context` in 1..128, `percdamp` and `column_block`. All paths in this
example are placeholders; no calibrated GPU build is included in this receipt.

`capture_files_by_layer` is keyed by canonical decimal layer strings, e.g.
`"3"`, `"4"`, through `"44"`, and must cover **exactly every declared layer**.
Each layer contains exactly `hidden_bf16`, `topk_ids_u16le`, and
`topk_weights_f32le`, each with `path` (resolved absolute filename), `bytes`
(positive integer) and `sha256` (lowercase hex). This permits one immutable
design to cover the entire 42-layer build while binding every layer to its
own local materialized capture files. Missing/extra layers, noncanonical keys
such as `"03"`, malformed pins and wrong-layer path substitutions fail closed.
The selector validates all declared pin structures but hashes/reads only the
requested layer's files.

For compatibility, the previous top-level `capture_files` form is accepted
only when `layers` is exactly the single requested layer. The two forms may
never coexist, even if their contents agree. A multi-layer design may not
reuse the old unkeyed form.

## Attribution

ExLlamaV3 (turboderp and contributors) supplies the ported MCG constants,
cyclic K4 stream and tile/lane layout; its MIT terms remain in
`LICENSE.exllamav3`. KQuant/QSRT (Luke Alonso and contributors, audited
snapshot `104dd9233f850a3955f4991bea68b07dd34deeb8`) supplies the Viterbi
backend and prior codebook workflow. That snapshot's license remains
unverified. The B12X `w4a8_trellis` pipeline is itself a QSRT port and informs
the neighboring producer/consumer kernel. This bridge adds the RNE encoder
adapter, versioned six-tensor interchange, exact TP4 packing/accounting,
calibration wiring, export/resume receipts and independent CPU fixtures.
It does not relicense those sources or claim their procedural codec as new.
