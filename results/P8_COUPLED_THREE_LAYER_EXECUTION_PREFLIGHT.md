# Coupled P8 three-layer execution preflight

Date: 2026-09-05  
Scope: read-only CPU preparation plus this report. No GPU, model tensor, scale
tensor, calibration tensor, service, image, or large-output read/write was
performed.

## Verdict

The existing encoder and TP4 packer commands are recoverable and the V3 design
still binds the intended fixed candidate: layers 3, 20, 22; draw zero; capped
Flash SiLU; H512/suh/H128 and signed H128/down-suh/H128 carriers; K4 procedural
MCG E4M3 with UE8M0/32; domain-balanced fit data; full-Hessian GPTQ-style
feedback between native groups; static order within each native 16-column
group; two scale-refit iterations; and no LDLQ.

The pilot is **not launch-cleared yet**. The old campaign-wide storage receipt
has expired and predates subsequent fixture/image/Git growth. More importantly,
the packer writes final sidecars directly, hashes them, but does not reopen and
semantically/hash-validate every stored weight tensor. The runtime validates
weight shapes and coupled scale hashes, not all four weight payload hashes.
Chunks therefore cannot yet be called safely redundant or retired solely from
the current packer receipt.

## Frozen V3 identities rechecked

Preparation record:
`results/P8_COUPLED_SCALE_FLASH_EVIDENCE_PREPARATION_V3.json`, SHA256
`4ebb96dd9d555fc18f24fb5f4d380216e1de30327a68d7aae89fc10f41878695`.
The proposal still pins that hash and transform hash. The record has external
hash identity rather than an embedded self-seal.

| V3 input | Recorded/current SHA256 | Result |
|---|---|---|
| encoder `quantize_p8_coupled_scale_layer.py` | `0c1900e3425b310bf4f669d02d6b75d52769aa04ad2a1544b4c8904072f90bc1` | match |
| coupled reference `p8_coupled_scale.py` | `ea48c69dab08f6f16aa0f31bbdded1a27e85a0e0809aaee5e50568bb739d1e66` | match |
| trellis codec `trellis_mxf.py` | `bb4309ef2be5f008c8091c8be44e4596f367c50fffcdeac9d7ebe1f1a0bef07c` | match |
| activation quantizer `canary_mxfp6_reap.py` | `ee1628523fb1b38d10e2232e31b528a8986c355d3d29cd5ece247ca3e11fc81d` | match |
| draw0/capped-SiLU transform | `093d219b18ba32471adcee746442b1481c7ba5659bbea62b94f1a665d4343a12` | match |
| BF16 source index | `e6007bd58fb7e07f9fe69544257ee2713f252ef5855bbf685b48c991d524ef0f` | match |
| roles V5 | `85cb6f8d29151863457830e41a141743feabcd69706a231d89cd01a3e02175c9` | match |
| capture manifest file | `f1a6fe7b8828b3461e81ee533d417dd1524355ef6a205145850116560197f81a` | match |
| capture canonical self-seal | `6a057e03808b2a7bc04bd6c757862f3954b166eba31fbc362c91cc966cd2e4b8` | match |
| EXL3 materialization canonical self-seal | `092be1ffa8db66bf02d4c370d0433a57aa48d4a6e5ce89723ef6a3bb7ca32643` | match |
| EXL3 index/config/quantization config | `2f64d21c...`, `4f5341e0...`, `1e5cdf56...` | all match their receipt |

No EXL3 weight shard was opened. V3 already records all 1,728 scale-tensor
hashes per layer and the three aggregate scale hashes. At actual encoder start,
`_validate_design` reopens only the indexed scale tensors, verifies their shard
receipts and payload hashes, enforces E288/H4096/I2048 and shared-scale
semantics, and compares their source seal to V3 before CUDA setup
(`p8_coupled_scale.py:328-456`, `quantize_p8_coupled_scale_layer.py:60-121,
154-159`). That deferred launch-time check is mandatory.

The TP4 packer itself is not a V3 input. Its current execution identity must be
added to the execution seal:

```text
glm53_nvfp4/build_p8_coupled_scale_tp4_sidecars.py
d8d0ebdc1f5621af2a195b929ca70a1cd5b9b0990feea80262ae388567fcd2e6
```

## Existing algorithm path (do not change for this pilot)

The encoder's fixed behavior is explicit in source:

- K4/E4M3/MCG, UE8M0/32 and two scale-refit iterations:
  `quantize_p8_coupled_scale_layer.py:47-57`.
- V3 source and activation-quantizer hash validation, fixed layers and draw0:
  lines 60-121.
- fit-only, 256-sample, domain-balanced per-expert carriers:
  lines 163-171.
- exact transformed weights and quantized input carrier before the FC1 Hessian:
  lines 217-231.
- reconstructed gate/up candidate feeding the transformed quantized middle
  carrier and causal down Hessian: lines 233-244.
- exact decode closure for every projection before serialization: lines
  245-269.
- receipt labels full-Hessian inter-group feedback, static in-group order,
  refit=2 and `ldlq=false`: lines 357-402.

V3's candidate storage labels remain: weight payload 4.25 bpw; stored
scale/draw metadata 0.003979859528718171 bpw; physical tensor total
4.253979859528719 bpw; 4.253993422896774 bpw when also charging regenerated
runtime sign storage. Safetensors headers and JSON receipts are additional.

## Command sheet

These commands are preparation records, not authority to launch. Replace only
`<PHYSICAL_GPU_ID>` after an idle/thermal/storage execution gate. With
`CUDA_VISIBLE_DEVICES` set to that one physical GPU, the encoder must retain
`--device cuda:0`. Do not add `--dense-output`.

The already-created V3 record can be reproduced by the existing preparer with
the following exact inputs. Do **not** rerun this command against the existing
output; it intentionally refuses overwrite and would reopen EXL3 scale shards.

```bash
PYTHONPATH=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-coupled-integration-v1 \
python3 -m glm53_nvfp4.prepare_p8_coupled_scale_v1 \
  --source-index /media/brandonmusic/klcstore/bmxfp4-glm53/downloads/GLM-5.3-Flash-BF16/model.safetensors.index.json \
  --capture-root /media/brandonmusic/klcstore/bmxfp4-glm53/teacher/calibration/main-ep4-full \
  --roles /media/brandonmusic/klcstore/bmxfp4-glm53/roles/roles-v5.json \
  --exl3-checkpoint /home/brandonmusic/models/GLM-5.3-Flash-EXL3-4bpw \
  --output /home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-coupled-integration-v1/results/P8_COUPLED_SCALE_FLASH_EVIDENCE_PREPARATION_V3.json
```

For each row of the fixed matrix below, invoke the existing encoder as follows:

```bash
CUDA_VISIBLE_DEVICES=<PHYSICAL_GPU_ID> \
PYTHONPATH=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-coupled-integration-v1 \
python3 -m glm53_nvfp4.quantize_p8_coupled_scale_layer \
  --source /media/brandonmusic/klcstore/bmxfp4-glm53/downloads/GLM-5.3-Flash-BF16 \
  --source-index /media/brandonmusic/klcstore/bmxfp4-glm53/downloads/GLM-5.3-Flash-BF16/model.safetensors.index.json \
  --capture-root /media/brandonmusic/klcstore/bmxfp4-glm53/teacher/calibration/main-ep4-full \
  --roles /media/brandonmusic/klcstore/bmxfp4-glm53/roles/roles-v5.json \
  --design /home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-coupled-integration-v1/results/P8_COUPLED_SCALE_FLASH_EVIDENCE_PREPARATION_V3.json \
  --exl3-scales /home/brandonmusic/models/GLM-5.3-Flash-EXL3-4bpw \
  --codec-output <CODEC_OUTPUT> \
  --receipt <CHUNK_RECEIPT> \
  --layer <LAYER> --expert-start <START> --expert-end <END> \
  --samples 256 --device cuda:0
```

Fixed expansion matrix, in execution order:

| layer | start:end | `<CODEC_OUTPUT>` below `p8-coupled-three-layer-v1/` | `<CHUNK_RECEIPT>` below the same root |
|---:|---:|---|---|
| 3 | 0:72 | `chunks/layer-003/p8-coupled-layer-003-experts-000-072.safetensors` | `receipts/chunks/layer-003-experts-000-072.json` |
| 3 | 72:144 | `chunks/layer-003/p8-coupled-layer-003-experts-072-144.safetensors` | `receipts/chunks/layer-003-experts-072-144.json` |
| 3 | 144:216 | `chunks/layer-003/p8-coupled-layer-003-experts-144-216.safetensors` | `receipts/chunks/layer-003-experts-144-216.json` |
| 3 | 216:288 | `chunks/layer-003/p8-coupled-layer-003-experts-216-288.safetensors` | `receipts/chunks/layer-003-experts-216-288.json` |
| 20 | 0:72 | `chunks/layer-020/p8-coupled-layer-020-experts-000-072.safetensors` | `receipts/chunks/layer-020-experts-000-072.json` |
| 20 | 72:144 | `chunks/layer-020/p8-coupled-layer-020-experts-072-144.safetensors` | `receipts/chunks/layer-020-experts-072-144.json` |
| 20 | 144:216 | `chunks/layer-020/p8-coupled-layer-020-experts-144-216.safetensors` | `receipts/chunks/layer-020-experts-144-216.json` |
| 20 | 216:288 | `chunks/layer-020/p8-coupled-layer-020-experts-216-288.safetensors` | `receipts/chunks/layer-020-experts-216-288.json` |
| 22 | 0:72 | `chunks/layer-022/p8-coupled-layer-022-experts-000-072.safetensors` | `receipts/chunks/layer-022-experts-000-072.json` |
| 22 | 72:144 | `chunks/layer-022/p8-coupled-layer-022-experts-072-144.safetensors` | `receipts/chunks/layer-022-experts-072-144.json` |
| 22 | 144:216 | `chunks/layer-022/p8-coupled-layer-022-experts-144-216.safetensors` | `receipts/chunks/layer-022-experts-144-216.json` |
| 22 | 216:288 | `chunks/layer-022/p8-coupled-layer-022-experts-216-288.safetensors` | `receipts/chunks/layer-022-experts-216-288.json` |

Every relative output above is rooted at:

```text
/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-three-layer-v1
```

That root was absent during this audit. Existing identity/rotated P8 sidecars
elsewhere are different candidates and must not be reused.

After all four chunks for one layer exist and their receipts close, use the
existing packer. Layer 3's exact expansion is:

```bash
PYTHONPATH=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-coupled-integration-v1 \
python3 -m glm53_nvfp4.build_p8_coupled_scale_tp4_sidecars \
  --chunk /media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-three-layer-v1/chunks/layer-003/p8-coupled-layer-003-experts-000-072.safetensors \
  --chunk /media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-three-layer-v1/chunks/layer-003/p8-coupled-layer-003-experts-072-144.safetensors \
  --chunk /media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-three-layer-v1/chunks/layer-003/p8-coupled-layer-003-experts-144-216.safetensors \
  --chunk /media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-three-layer-v1/chunks/layer-003/p8-coupled-layer-003-experts-216-288.safetensors \
  --output-dir /media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-three-layer-v1/sidecars/layer-003 \
  --receipt /media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-three-layer-v1/receipts/layer-003-sidecars.json \
  --layer 3 --world-size 4
```

For layers 20 and 22, substitute every zero-padded layer component and the
numeric `--layer` consistently; the four expert ranges remain identical. This
is argument substitution only—the packer and format are unchanged.

## Disk and memory peak accounting

For E288/H4096/I2048, one layer has 7,247,757,312 routed weights.

### Encoder chunks

For one 72-expert command:

- K4+UE8M0 weight tensor data: 962,592,768 bytes;
- coupled tensor data including repeated shared scales and 72 draw bytes:
  901,192 bytes;
- minimum file data: **963,493,960 bytes**, plus its safetensors header.

Four chunks therefore retain **3,853,975,840 tensor bytes**, plus four headers.
Comparable existing 72-expert P8 chunks have approximately 60 KiB headers;
the exact coupled header has not yet been emitted. The encoder holds roughly
963,493,960 bytes of completed chunk tensors in CPU memory before `save_file`,
in addition to per-expert CPU/GPU workspaces. Its receipt records observed CUDA
peak after execution, but there is no preregistered host-memory ceiling.

### TP4 packing

Each TP rank contains 962,592,768 weight bytes and 901,408 stored scale/draw
bytes: **963,494,176 tensor bytes**. Four ranks retain **3,853,976,704 tensor
bytes**, plus four headers and one receipt. While `_load_rank_coupled` builds a
rank, the retained sliced inputs are approximately 963,543,328 bytes and the
assembled result is 963,494,176 bytes, a **1,927,037,504-byte lower bound** on
simultaneously live CPU tensors before stack/cat temporaries and serializer
workspace. It processes ranks serially, so this is not multiplied by four.

The minimum on-disk coexistence of one layer's chunks and final TP4 sidecars is
therefore **7,707,952,544 bytes plus eight safetensors headers and receipts**.
The older ledger's 7,707,963,392-byte estimate is close but is not an exact
writer bound: its chunk subtotal omitted the 288 stored draw bytes, and it did
not derive the coupled chunk headers or serializer behavior. Both Python
entrypoints write directly to final paths rather than a named `.partial`; a
failed serializer can leave an output path which a retry then refuses to
overwrite. No second full on-disk tensor plane is visible in Python source, but
the Rust `safetensors` serializer's filesystem behavior has not been measured
for this full shape.

The prior sequential proposal forecast a 15,415,939,072-byte layer-artifact
peak at layer 22, then 20,074,902,837 bytes including its historical external
bound and fixture. That left 9,925,097,163 bytes, but the external receipt is
expired and the header/serializer bound above was not closed. A fresh
campaign-wide ledger must count the now-present 963,505,370-byte synthetic
fixture directory, current worktree/Git/image/cache growth, any active capture
peak, exact per-layer observed files, and a serializer/receipt allowance before
each layer. Raw filesystem free space was 150,828,048,384 bytes at this audit;
that does not relax the 30,000,000,000-byte campaign ceiling.

## Sequential retirement: conditionally valid, not yet executable

The fixed order 3, 20, 22 can remain under the ceiling by retaining final
sidecars and retiring the corresponding four chunks after each layer. Before a
chunk can be retired, all of these gates must exist for that layer:

1. Four chunk receipts whose ordered ranges cover exactly 0:72, 72:144,
   144:216 and 216:288 and agree on V3 design and EXL3 scale-source hashes.
2. Four final sidecars and a packer receipt agreeing on layer, TP4, 4.25 bpw,
   draw0, V3 design, exact scale source, complete source list and whole-file
   SHA256.
3. Independent post-write reopening which checks every sidecar's schema,
   shapes, stored tensor SHA fields, whole-file SHA and source receipt. The
   current packer does not provide this after `save_file`.
4. Real runtime-loader closure for every layer/rank with external V3 and
   transform pins. The current synthetic device harness is layer 3/rank 0 and
   cannot stand in for twelve real sidecars.
5. Durable retention of the V3 record, transform, all chunk/packer/closure
   receipts and logs. Removing chunks sacrifices cheap byte-level replay, so
   it must be recorded as storage retirement, not treated as if the inputs
   never existed.
6. A fresh storage ledger taken after verification and before the next layer.

No current committed command closes gates 3 and 4 for all twelve real
sidecars. Therefore this audit supports the *architecture* of sequential
packing, but does not authorize deletion or claim that the chunks are already
redundant.

## Remaining preflight blockers

1. The proposal status is explicitly `proposal-before-encoding-not-execution-authority`.
   An execution seal still must pin GPU, image/runtime, current packer SHA,
   output root and fresh global storage receipt.
2. Refresh the conservative campaign-wide storage ledger immediately before
   each layer; the earlier receipt expired at Unix 1788639928.
3. Add or identify a committed post-write all-tensor verifier and real
   layer/rank runtime-closure path before allowing chunk retirement.
4. Predeclare handling of direct-write failures: quarantine and inspect a
   residual output; never delete/retry it silently.
5. Confirm no conflicting GPU quality/service job and select an idle physical
   GPU. This audit deliberately did not inspect or allocate GPUs.
6. Let the encoder perform its mandatory launch-time EXL3 scale-shard and BF16
   model/capture validation; none of those large payloads was read here.
7. Preserve `PYTHONPATH` explicitly. A diagnostic bare `pytest` invocation
   failed collection with `ModuleNotFoundError: glm53_nvfp4`; the correct
   command below passed.

CPU/static check executed without model data or GPU:

```bash
PYTHONPATH=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-coupled-integration-v1 \
python3 -m pytest -q \
  tests/test_p8_coupled_scale.py \
  tests/test_build_p8_coupled_scale_tp4_sidecars.py \
  tests/test_p8_full_coupled_runtime.py
```

Result: **30 passed in 0.96 seconds**. This is CPU/static ABI evidence only,
not real-sidecar, device, KLD, or throughput closure.
