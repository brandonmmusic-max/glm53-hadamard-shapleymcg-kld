# V3 storage cleanup receipt

- Date: 2026-09-03 America/New_York
- Scope: campaign-local superseded V2 quantized tensor payload only
- Removed path: `/media/brandonmusic/klcstore/bmxfp4-glm53/chunks`
- Pre-removal payload: 169 files, 171257337652 bytes (`du -sh`: 160G)
- Reason: V3 uses the stock NVFP4 carrier plus `chunks-v3/had16` and the separate learned-rotation chunk tree; no V3 script, record, plan, or live process referenced the V2 path.
- Preserved: BF16 source, teacher logits and calibration captures, V3 chunks, compact Hessians, all experiment evidence, and the lightweight V2 candidate metadata/index directory.
- V2 metadata hashes retained in the candidate directory:
  - `OVERLAY.json`: `0c8d00d4f5278242b53c845011ead6e7a3d371f5cb1ac3352dd8f59e66eef3cf`
  - `config.json`: `676382abd1e90a6c85f0c8f33d45441ecd45fd514fd7b63ce5610e732d8e4996`
  - `model.safetensors.index.json`: `7a59df020e308e020ec8587692014e878fcfb65e0664c41712b3253489702a66`
- Recoverability: the deleted tensor payload is not recoverable locally, but it can be regenerated from the preserved BF16 source and calibration/evidence inputs.
- Post-removal filesystem readings:
  - `/media/brandonmusic/klcstore`: 468779331584 bytes available (`df -h`: 437G), up from 308027191296 bytes in the pre-cleanup audit.
  - `/home/brandonmusic` root filesystem: 215525388288 bytes available (`df -h`: 201G); unchanged apart from normal live-campaign growth.
- Forward allocation: all V3 MXFP6 layer payloads are routed to `/media/brandonmusic/klcstore/bmxfp4-glm53/mxfp6-v3`; only the already-running learned-rotation chunks remain on the root filesystem.
