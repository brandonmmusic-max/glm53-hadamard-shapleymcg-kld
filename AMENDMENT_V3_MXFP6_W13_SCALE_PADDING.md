# V3 MXFP6 w13 virtual-TP scale-padding correction

Date: 2026-09-03. Fail-only loader diagnostics established that the exact group-32 `up_proj.weight_scale` tensor stored as `(2048, 128)` reached the mixed loader as `(2048, 256)`, while the B12X fused parameter correctly expected 128 groups. The virtual-TP preload had padded the scale axis to the stock NVFP4 group-16 width before B12X dispatch. The repeated canary failed before readiness or inference.

For only the B12X MXFP6 MoE method and only `w13_weight_scale`, the bridge now recognizes the exact 2x padded condition and narrows the last axis to the declared parameter width before invoking the unchanged vLLM loader. Any other geometry follows the original path and retains fail-only diagnostics. Packed FP6 codes, the retained scale values, layer assignments, rotations, roles, metrics, and decisions do not change. A patch-active marker is emitted once per worker. The one-layer canary remains mandatory before bulk quantization.
