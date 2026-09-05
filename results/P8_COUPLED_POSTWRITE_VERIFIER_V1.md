# P8 coupled real-layer postwrite closure

Date: 2026-09-05  
Scope: CPU/source audit and small synthetic safetensors tests only. No GPU,
service, image build, model encode, or large write was performed.

## Result

The smallest missing implementation in the three-layer execution preflight was
an independent verifier after TP4 serialization. It is now implemented in
`glm53_nvfp4/verify_p8_coupled_scale_tp4_sidecars.py` without changing the
encoder or packer.

For every one of the four TP ranks, the verifier independently rebuilds the
expected rank tensors and metadata from the four immutable encoder chunks,
reopens the serialized sidecar, and requires:

- the packer receipt header and rank inventory to match the frozen TP4 product;
- exact source-chunk, design, scale-source, byte-count, shape, and bpw fields;
- exact serialized metadata, tensor inventory, dtype, shape, value and tensor
  SHA256 for all eight tensors; and
- one common chunk/design/scale identity across all ranks.

The emitted closure receipt remains deliberately bounded. It says the
postwrite source-exact structural gate passed, but records
`runtime_loader_closure: not tested` and `retirement_authorized: false`.
Therefore it cannot independently authorize deletion.

Focused corruption tests prove that changing a tensor or serialized metadata
is rejected even when the whole-file hash in the packer receipt is updated.
The combined existing/new CPU suite passed: **40 passed**.

## Frozen inputs and current storage

The encoder and packer were not modified:

| file | SHA256 |
|---|---|
| `quantize_p8_coupled_scale_layer.py` | `0c1900e3425b310bf4f669d02d6b75d52769aa04ad2a1544b4c8904072f90bc1` |
| `build_p8_coupled_scale_tp4_sidecars.py` | `d8d0ebdc1f5621af2a195b929ca70a1cd5b9b0990feea80262ae388567fcd2e6` |
| V3 preparation record | `4ebb96dd9d555fc18f24fb5f4d380216e1de30327a68d7aae89fc10f41878695` |

At this audit, NVMe had 150,820,573,184 free bytes and klcstore had
79,915,155,456 free bytes. The three-layer target root occupied 2,719 bytes;
the retained coupled fixture occupied 964,856,268 bytes.

One complete layer is 3,853,975,840 chunk tensor bytes plus 3,853,976,704
sidecar tensor bytes. With verified sequential retirement, the final three
sidecar layers, one current chunk layer, and the fixture require approximately
16.38 GB before headers, receipts, serializer workspace, and caches. Even if
all three layers' chunks coexist with all three sidecar layers, the tensors and
fixture total approximately 24.09 GB, leaving about 5.91 GB of the 30 GB
campaign cap before those extras. Sequential retirement is therefore the safer
schedule, subject to the gates below.

## First real encode command prepared, not launched

This is the first frozen matrix cell, layer 3 experts 0:72. It is not execution
authority: choose an idle physical GPU, write the execution seal, perform a
fresh storage check, and prove both output paths absent before launch.

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
  --codec-output /media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-three-layer-v1/chunks/layer-003/p8-coupled-layer-003-experts-000-072.safetensors \
  --receipt /media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-three-layer-v1/receipts/chunks/layer-003-experts-000-072.json \
  --layer 3 --expert-start 0 --expert-end 72 \
  --samples 256 --device cuda:0
```

## Postwrite invocation and retirement boundary

After all four layer-3 chunks and four packed sidecars exist, run:

```bash
PYTHONPATH=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-coupled-integration-v1 \
python3 -m glm53_nvfp4.verify_p8_coupled_scale_tp4_sidecars \
  --chunk /media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-three-layer-v1/chunks/layer-003/p8-coupled-layer-003-experts-000-072.safetensors \
  --chunk /media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-three-layer-v1/chunks/layer-003/p8-coupled-layer-003-experts-072-144.safetensors \
  --chunk /media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-three-layer-v1/chunks/layer-003/p8-coupled-layer-003-experts-144-216.safetensors \
  --chunk /media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-three-layer-v1/chunks/layer-003/p8-coupled-layer-003-experts-216-288.safetensors \
  --sidecar /media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-three-layer-v1/sidecars/layer-003/p8-layer-003-tp4-rank-0.safetensors \
  --sidecar /media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-three-layer-v1/sidecars/layer-003/p8-layer-003-tp4-rank-1.safetensors \
  --sidecar /media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-three-layer-v1/sidecars/layer-003/p8-layer-003-tp4-rank-2.safetensors \
  --sidecar /media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-three-layer-v1/sidecars/layer-003/p8-layer-003-tp4-rank-3.safetensors \
  --packer-receipt /media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-three-layer-v1/receipts/layer-003-sidecars.json \
  --receipt /media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-three-layer-v1/receipts/layer-003-postwrite-closure.json \
  --layer 3
```

Chunk retirement becomes eligible only after: this verifier passes; the real runtime
loader closes all four ranks for the layer; a fresh storage ledger passes; and
an explicit retirement receipt identifies the exact chunk hashes removed.
Until then, no chunk is redundant and none is authorized for deletion.
